from btcy_holter import *


class BeatAmplitudeThresholds:
    median_amp:             Final[List] = [0.3, 15]  # mV
    min_amp:                Final[int] = 10
    max_amp:                Final[int] = 10
    avg_ratio:              Final[int] = 1 / 4


class HighBeatThresholds:
    buffer_size:            Final[int] = 10
    n_shift:                Final[int] = 20
    n_var:                  Final[int] = 10
    excited_peak_threshold: Final[int] = 20
    length_segment:         Final[int] = 3


class RemoveBeats(
        pt.Algorithm
):
    
    QRS_OFFSET:             float = 0.1             # seconds
    MAX_SEGMENT_PROCESS:    Final[float] = 5        # minutes
    
    SYMBOL_LEVEL:   Final[List[str]] = [df.HolterSymbols.N.value, df.HolterSymbols.IVCD.value, df.HolterSymbols.SVE.value, df.HolterSymbols.VE.value]
    
    def __init__(
            self,
            is_process_event:   bool,
            data_structure:     sr.AIPredictionResult,
            **kwargs
    ) -> None:
        try:
            super(RemoveBeats, self).__init__(
                    data_structure=data_structure,
                    is_hes_process=is_process_event
            )

            self.data_structure.lead_off = np.zeros_like(self.data_structure.beat)
            self.data_structure.lead_off_frames = np.zeros_like(self.data_structure.ecg_signal)

            self.lead_off_frames = np.zeros_like(self.data_structure.ecg_signal)

            self.rhythm_class = cf.RHYTHMS_DATASTORE['classes']

            frame = np.arange(-int(self.QRS_OFFSET * self.data_structure.sampling_rate),
                              int(self.QRS_OFFSET * self.data_structure.sampling_rate))[None, :]

            self.ecg_frame = self._get_beat_frames(self.data_structure.ecg_signal, self.data_structure.beat, frame)
            self.beat_min_amp = np.min(self.ecg_frame, axis=-1)
            self.beat_max_amp = np.max(self.ecg_frame, axis=-1)

            # region param
            try:
                self.max_sample_process = kwargs['max_sample_process']
            except (Exception,):
                self.max_sample_process = (5 * df.SECOND_IN_MINUTE) * self.data_structure.sampling_rate

            try:
                self.plot_result = kwargs['plot_result']
            except (Exception,):
                self.plot_result = False

            try:
                self.using_qrs_detection = kwargs['using_qrs_detection']
            except (Exception,):
                self.using_qrs_detection = True

            self.ba_thr = BeatAmplitudeThresholds()
            self.hb_thr = HighBeatThresholds()
            self.hb_thr.length_segment = self.hb_thr.length_segment * self.data_structure.sampling_rate

            self.beats_remove_by_alg = deepcopy(self.data_structure.beat)
            self.beats_remove_by_qrs = deepcopy(self.data_structure.beat)
            if self.plot_result:
                self.beats_copy = deepcopy(self.data_structure.beat)
                self.symbols_copy = deepcopy(self.data_structure.symbol)
            # endregion param

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    def _get_frame(
            self,
            len_signal:         int,
            num_of_elements:    int
    ) -> NDArray:
        
        frame_index = np.arange(0, len_signal, num_of_elements)
        try:
            if len_signal % num_of_elements == 0:
                frame_index = np.concatenate((frame_index, [len_signal]))
    
            frame_index = np.asarray([np.arange(frame_index[i], frame_index[i + 1])
                                      for i in range(len(frame_index) - 1)])
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return frame_index

    def _get_beat_frames(
            self,
            buf_ecg:    NDArray,
            beat:       NDArray,
            frame:      NDArray
    ) -> NDArray:

        result = list()
        try:
            b_frame = frame + beat.reshape(-1, 1)
            b_frame[b_frame < 0] = 0
            b_frame[b_frame >= len(buf_ecg)] = len(buf_ecg) - 1
            result = buf_ecg[b_frame]

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return result

    def __check_high_beat_by_var(
            self,
    ) -> NDArray:

        ecg_signal = deepcopy(self.data_structure.ecg_signal)
        ecg_signal = ut.butter_highpass_filter(ecg_signal, 0.5, self.data_structure.sampling_rate)
        
        lead_off = np.zeros_like(ecg_signal, dtype=int)

        try:
            count_var = self.hb_thr.n_var * self.hb_thr.n_shift

            frame_buf_index = self._get_frame(
                len_signal=len(ecg_signal),
                num_of_elements=self.hb_thr.n_shift
            )
            var_values = np.var(ecg_signal[frame_buf_index], axis=-1)

            frame_var_index = self._get_frame(
                len_signal=len(var_values),
                num_of_elements=self.hb_thr.n_var
            )
            var_values = var_values[frame_var_index]

            median_amps = np.max(var_values, axis=-1) - np.min(var_values, axis=-1)
            index_high_peak = np.flatnonzero(median_amps > self.hb_thr.excited_peak_threshold)

            if len(index_high_peak) > 0:
                ind = np.hstack(frame_var_index[index_high_peak])
                lead_off_samples = np.hstack(frame_buf_index[ind])
                lead_off_samples = lead_off_samples[::count_var] // self.hb_thr.length_segment
                lead_off_samples = lead_off_samples.astype(int)

                index_lead_off = np.hstack([
                    np.arange(x * self.hb_thr.length_segment, (x + 1) * self.hb_thr.length_segment)
                    for x in lead_off_samples
                ])

                index_lead_off = index_lead_off[np.logical_and(index_lead_off > 0, index_lead_off < len(ecg_signal))]
                lead_off[index_lead_off] = 1

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return lead_off

    def _check_high_beat(
            self,
            max_amp_beat: NDArray,
            min_amp_beat: NDArray
    ) -> NDArray:
        
        cond1 = np.abs(max_amp_beat - min_amp_beat) < self.ba_thr.median_amp[1]
        cond2 = np.abs(min_amp_beat) < self.ba_thr.min_amp
        cond3 = np.abs(max_amp_beat) < self.ba_thr.max_amp

        return cond1 & cond2 & cond3

    # @df.timeit
    def __remove_high_beats(
            self
    ) -> None:
        try:
            if len(self.data_structure.beat) <= df.LIMIT_BEAT_SAMPLE_IN_SIGNAL:
                return
            
            ind_invalid_beat = np.flatnonzero(~np.asarray(list(map(
                self._check_high_beat,
                self.beat_min_amp,
                self.beat_max_amp
            ))))
            if len(ind_invalid_beat) == 0:
                return

            index = np.flatnonzero(np.isin(
                    self.data_structure.symbol[ind_invalid_beat],
                    df.VALID_BEAT_TYPE + [df.HolterSymbols.MARKED.value]
            ))
            if len(index) == 0:
                return
            
            self.data_structure.beat            = np.delete(self.data_structure.beat, ind_invalid_beat[index])
            self.data_structure.symbol          = np.delete(self.data_structure.symbol, ind_invalid_beat[index])
            self.data_structure.rhythm          = np.delete(self.data_structure.rhythm, ind_invalid_beat[index])
            self.data_structure.beat_channel    = np.delete(self.data_structure.beat_channel, ind_invalid_beat[index])
            self.data_structure.lead_off        = np.delete(self.data_structure.lead_off, ind_invalid_beat[index])

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    def _check_low_beat_by_qrs_model(
            self,
            beats:              NDArray,
            index_beats:        NDArray
    ) -> NDArray:
        try:
            if not self.using_qrs_detection:
                return index_beats
                
            from btcy_holter.ai_core.core.qrs_detection import QRSDetection
            
            qrs_func = QRSDetection()
            qrs_predict = qrs_func.prediction(
                    beat=beats[index_beats],
                    ecg_signal=self.data_structure.ecg_signal,
                    ecg_signal_fs=self.data_structure.sampling_rate
            )
            
            index_beats = index_beats[np.flatnonzero(qrs_predict == qrs_func.datastore['classes']['NO_QRS'])]
            if len(index_beats) > 0:
                self.beats_remove_by_qrs = beats[index_beats]

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return index_beats

    def _check_low_beat_in_artifact_region(
            self,
            beat_ind:           NDArray,
            syms:               NDArray,
            amp_threshold:      float,
            beat_type:          str,
            index_low_beats:    List
    ) -> List:
        try:
            index_other_beats = np.flatnonzero(~np.isin(syms, [beat_type, df.HolterBeatTypes.MARKED.value]))
            if len(index_other_beats) > 0 and amp_threshold is not None:
                index_other_beats = beat_ind[index_other_beats]
                avg_amp_beat = np.abs(self.beat_min_amp[index_other_beats] - self.beat_max_amp[index_other_beats])
                index_del = index_other_beats[np.flatnonzero(avg_amp_beat < amp_threshold)]
                if len(index_del) == 0:
                    return index_low_beats
                
                group_index = df.get_group_from_index_event(index_del, is_region=True)
                valid_group = list(filter(
                        lambda x: df.calculate_duration(
                                beats=self.data_structure.beat,
                                index=group_index,
                                sampling_rate=self.data_structure.sampling_rate
                        ) >= df.CRITERIA['OTHER']['duration'],
                        group_index
                ))
                if len(valid_group) == 0:
                    return index_low_beats
                
                other_index = list(chain.from_iterable(map(
                        lambda y: np.arange(y[0], y[-1] + 1),
                        valid_group
                )))
                index_del = np.delete(index_del, np.isin(index_del, np.array(other_index)))
                index_low_beats.extend(index_del)

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return index_low_beats

    # @df.timeit
    def __remove_low_beats(
            self
    ) -> None:
        try:
            total_segments = max(1, len(self.data_structure.ecg_signal) // self.max_sample_process)

            index_low_beats = list()
            for index_seg in range(total_segments):
                beat_ranges = df.get_indices_within_range(
                        self.data_structure.beat,
                        start=index_seg * self.max_sample_process,
                        stop=(index_seg + 1) * self.max_sample_process
                )
                if len(beat_ranges) == 0:
                    continue

                beat_channel = self.data_structure.beat_channel[beat_ranges]
                for chn in np.unique(beat_channel):
                    beat_ind = beat_ranges[np.flatnonzero(beat_channel == chn)]
                    syms = self.data_structure.symbol[beat_ind]

                    ind = None
                    beat_type = None
                    for beat_type in self.SYMBOL_LEVEL:
                        ind = np.flatnonzero(syms == beat_type)
                        if len(ind) > 0:
                            break
                            
                    amp_threshold = None
                    if len(ind) > df.LIMIT_BEAT_SAMPLE_IN_SIGNAL:
                        min_amp = self.beat_min_amp[beat_ind[ind]]
                        max_amp = self.beat_max_amp[beat_ind[ind]]
                        idx = np.flatnonzero(np.logical_and(
                                np.abs(min_amp) <= self.ba_thr.min_amp,
                                np.abs(max_amp) <= self.ba_thr.max_amp
                        ))
                        if len(idx) > df.LIMIT_BEAT_SAMPLE_IN_SIGNAL:
                            ind = ind[idx]
                            avg_amp_beat = np.abs(min_amp[idx] - max_amp[idx])
                            amp_threshold = np.quantile(avg_amp_beat, 0.8) * self.ba_thr.avg_ratio
                            index_low_beats.extend(beat_ind[ind[np.flatnonzero(avg_amp_beat < amp_threshold)]])

                    index_low_beats = self._check_low_beat_in_artifact_region(
                            beat_ind=beat_ind,
                            syms=syms,
                            amp_threshold=amp_threshold,
                            beat_type=beat_type,
                            index_low_beats=index_low_beats
                    )

            if len(index_low_beats) > 0:
                index_low_beats = np.array(index_low_beats)
                self.beats_remove_by_alg = self.data_structure.beat[index_low_beats]
                index_low_beats = self._check_low_beat_by_qrs_model(
                        beats=self.data_structure.beat,
                        index_beats=index_low_beats
                )

                self.data_structure.beat            = np.delete(self.data_structure.beat, index_low_beats)
                self.data_structure.symbol          = np.delete(self.data_structure.symbol, index_low_beats)
                self.data_structure.rhythm          = np.delete(self.data_structure.rhythm, index_low_beats)
                self.data_structure.beat_channel    = np.delete(self.data_structure.beat_channel, index_low_beats)
                self.data_structure.lead_off        = np.delete(self.data_structure.lead_off, index_low_beats)

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def __mark_high_beat_to_noise(
            self
    ) -> None:
        try:
            self.data_structure.lead_off_frames = self.__check_high_beat_by_var()
            index_in_lead_off = np.flatnonzero(self.data_structure.lead_off_frames[self.data_structure.beat])
            index_in_lead_off_sig = np.flatnonzero(self.data_structure.lead_off_frames)

            index_lead_off = list()
            if len(index_in_lead_off) > 0:
                index_in_lead_off = np.unique((index_in_lead_off.reshape((-1, 1)) + np.arange(-1, 2)).flatten())

                index = np.flatnonzero(np.logical_and(
                    index_in_lead_off >= 0,
                    index_in_lead_off <= len(self.data_structure.symbol) - 1)
                )
                index_lead_off = index_in_lead_off[index]

            elif len(index_in_lead_off_sig) > 0:
                group = np.split(index_in_lead_off_sig, np.flatnonzero(np.diff(index_in_lead_off) != 1) + 1)
                group = np.array(list(map(
                    lambda x: np.flatnonzero(np.logical_and(
                            self.data_structure.beat[:-1] <= x[0],
                            self.data_structure.beat[1:] >= x[-1]
                    )),
                    group
                )))
                index_lead_off = (np.arange(0, 2)[:, None] + group).flatten()

            if len(index_lead_off) > 0:
                self.data_structure.symbol[index_lead_off] = df.HolterSymbols.OTHER.value
                self.data_structure.lead_off[index_lead_off] = 1

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    # @df.timeit
    def __mark_pause_lead_off(
            self
    ) -> None:
        try:
            ind_noise = np.flatnonzero(np.logical_and(
                self.data_structure.rhythm == self.rhythm_class['OTHER'],
                np.array(list(map(lambda x: x in df.VALID_BEAT_TYPE, self.data_structure.symbol)))
            ))
            if len(ind_noise) == 0:
                return
            
            group = df.get_group_from_index_event(ind_noise)
            len_group = list(filter(
                lambda x: x[1] <= df.LIMIT_BEAT_SAMPLE_IN_SIGNAL,
                map(lambda y, z: [z, len(y)], group, range(len(group)))
            ))
            if len(len_group) == 0:
                return
            
            valid_len_group = np.hstack(([group[x] for x in np.array(len_group)[:, 0]]))
            i = np.flatnonzero(self.data_structure.lead_off[valid_len_group] == 0)
            if len(i) == 0:
                return
            
            self.data_structure.rhythm[valid_len_group[i]] = self.rhythm_class['SINUS']

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def review(
            self
    ):
        try:
            def bbox_color(x):
                return dict(boxstyle='round', fc=df.BEAT_COLORS[x], ec=df.BEAT_COLORS_EC[x])

            fig, (axis) = plt.subplots(2, 1, sharex='all', sharey='all')
            fig.suptitle(f'[POST-PROCESSING] REMOVE LOW BEATS - CH{self.data_structure.channel}')
            fig.subplots_adjust(hspace=0, wspace=0, left=0.02, bottom=0.06, top=0.96, right=0.99)

            y_max = 2
            axis[0].plot(self.data_structure.ecg_signal)
            axis[0].vlines(self.beats_copy, ymin=-y_max - 0.3, ymax=y_max - 0.2, lw=0.5, color='r', linestyles='dotted')
            [axis[0].annotate(x, xy=(self.beats_copy[i], y_max - 0.2), xycoords='data',
                              textcoords='data', bbox=bbox_color(x))
             for i, x in enumerate(self.symbols_copy)]

            axis[1].set_title('SUMMARIES - BEAT LEAD OFF')
            axis[1].plot(self.data_structure.ecg_signal)
            axis[1].vlines(self.beats_copy, ymin=-y_max - 0.3, ymax=y_max - 0.2, lw=0.5, color='r', linestyles='dotted')
            [axis[1].annotate(x, xy=(self.data_structure.beat[i], y_max - 0.2), xycoords='data', textcoords='data',
                              bbox=bbox_color(x))
             for i, x in enumerate(self.data_structure.symbol)]

            ind_lead_off = np.flatnonzero(self.data_structure.lead_off == 1)
            group = np.split(ind_lead_off, np.flatnonzero(np.diff(ind_lead_off) != 1) + 1)
            for gr in filter(lambda x: len(x) > 0, group):
                axis[1].axvspan(*self.data_structure.beat[gr[[0, -1]]], color='yellow')

            if len(self.beats_remove_by_alg) < len(self.beats_copy):
                axis[1].plot(self.beats_remove_by_alg, self.data_structure.ecg_signal[self.beats_remove_by_alg],
                             'r*', label='beats_remove_by_alg')

            if len(self.beats_remove_by_qrs) < len(self.beats_copy):
                axis[1].plot(self.beats_remove_by_qrs, self.data_structure.ecg_signal[self.beats_remove_by_qrs],
                             'ko', label='beats_remove_by_qrs')

            axis[0].set_xlim(0, len(self.data_structure.ecg_signal))
            axis[0].set_ylim(-y_max, y_max + 0.5)

            plt.legend(loc='lower right')
            plt.show()
            plt.close()

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def process(
            self
    ) -> sr.AIPredictionResult:

        try:
            s = np.unique(self.data_structure.symbol)
            if len(s) > 0 and s[0] == df.HolterSymbols.MARKED.value:
                return self.data_structure
            
            self.__remove_high_beats()
            if len(self.data_structure.beat) == 0:
                return self.data_structure
            
            self.__remove_low_beats()
            
            self.__mark_high_beat_to_noise()
            
            self.__mark_pause_lead_off()

            self.plot_result and self.review()

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return self.data_structure
