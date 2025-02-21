from btcy_holter import *
from btcy_holter.algs.remove_beats import (
    BeatAmplitudeThresholds
)


class PausedDetection(
        pt.Algorithm
):
    MARKED_ID:               Final[int] = 1
    
    IDLE:                    Final[int] = 0
    PAUSE1:                  Final[int] = 1  # one missing QRS complex
    PAUSE2:                  Final[int] = 2  # two missing QRS complex
    ASYSTOLE:                Final[int] = 4  # – no QRS activity (QRS complex) for at least 5 seconds
    
    RATIO_TO_P_WAVE_DETECT:  Final[float] = 1 / 3
    RATIO_TO_T_WAVE_DETECT:  Final[float] = 1 - RATIO_TO_P_WAVE_DETECT

    COUNT_BEAT_PAUSE_VALID:  Final[int] = 2
    
    BEAT_AROUND:             Final[int] = 5
    BEAT_OFFSET:             Final[float] = 0.1  # second

    def __init__(
            self,
            data_structure:     sr.AIPredictionResult,
            record_config:      sr.RecordConfigurations,
            is_hes_process:     bool = False,
    ) -> None:
        
        try:
            super(PausedDetection, self).__init__(
                    data_structure=data_structure,
                    record_config=record_config,
                    is_hes_process=is_hes_process
            )

            self.sync_beat_type()
            self.max_bpm: Final[float] = df.SECOND_IN_MINUTE / (self.record_config.pause_threshold / df.MILLISECOND)
            self.min_bpm: Final[float] = df.SECOND_IN_MINUTE / (df.THRESHOLD_MAX_PAUSE_RR_INTERVALS / df.MILLISECOND)

            self.region = None
            self.pause_events = np.zeros_like(self.data_structure.beat)
 
            self.beat_amp_threshold = deepcopy(BeatAmplitudeThresholds())

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    def _get_ecg_signal_based_on_beat_channels(
            self,
            ecg_signals: NDArray
    ) -> NDArray:
        
        ecg_channel = np.array([])
        try:
            if self.data_structure.beat_channel is None:
                st.get_error_exception('BEAT Channel is None')
                
            if len(np.unique(self.data_structure.beat_channel)) == 1:
                channel = df.find_most_frequency_occurring_values(self.data_structure.beat_channel)
                ecg_channel = ecg_signals[:, channel]
            else:
                group_channel = np.split(
                        np.arange(len(self.data_structure.beat_channel)),
                        np.flatnonzero(np.abs(np.diff(self.data_structure.beat_channel)) != 0) + 1
                )
                
                ecg_channel = ecg_signals[:, df.find_most_frequency_occurring_values(self.data_structure.beat_channel)]
                for i, gr in enumerate(group_channel):
                    bg = self.data_structure.beat[gr[0]] if i != 0 else 0
                    en = self.data_structure.beat[gr[-1]] if i != len(group_channel) - 1 else len(group_channel) - 1
                    ecg_channel[bg: en] = ecg_signals[bg: en, self.data_structure.beat_channel[gr[0]]]
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
        return ecg_channel
    
    def _preprocess(
            self,
            ecg_signal: NDArray
    ) -> [NDArray, NDArray]:
        
        ecg_bp = None
        ecg_filter = None
        try:
            
            lp_ecg = ut.butter_lowpass_filter(
                    ecg_signal,
                    cutoff=40,
                    fs=self.record_config.sampling_rate
            )
            
            ecg_filter = ut.butter_highpass_filter(
                    lp_ecg,
                    cutoff=0.5,
                    fs=self.record_config.sampling_rate
            )
            
            ecg_bp = ut.butter_highpass_filter(
                    lp_ecg,
                    cutoff=2,
                    fs=self.record_config.sampling_rate
            )
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return ecg_filter, ecg_bp
    
    def _get_ecg_signals(
            self
    ) -> None:
        try:
            ecg_signal = None
            if self.data_structure.ecg_signal is not None:
                ecg_signal = deepcopy(self.data_structure.ecg_signal)
            
            elif df.check_hea_file(self.record_config.record_path):
                record = wf.rdrecord(self.record_config.record_path)
                ecg_signals = np.nan_to_num(record.p_signal)
                ecg_signal = self._get_ecg_signal_based_on_beat_channels(ecg_signals)
            
            elif df.check_dat_file(self.record_config.record_path):
                ecg_signals = ut.get_data_from_dat(self.record_config.record_path, self.record_config)
                ecg_signal = self._get_ecg_signal_based_on_beat_channels(ecg_signals)
            
            else:
                st.get_error_exception(error='Missing ECG Signal.')
            
            self.ecg_signal, self.ecg_bandpass = self._preprocess(ecg_signal)
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    def _determine_low_amplitude_threshold(
            self,
    ) -> None:
        try:
            beat_valid = np.flatnonzero(np.isin(
                    self.data_structure.symbol, 
                    [df.HolterSymbols.N.value, df.HolterSymbols.IVCD.value]
            ))
            if len(beat_valid) == 0:
                return
            
            offset_samples = int(self.BEAT_OFFSET * self.record_config.sampling_rate)
            frames = np.arange(-offset_samples, offset_samples)[None, :]
            frames = frames + self.data_structure.beat[beat_valid].reshape(-1, 1)
            frames[frames < 0] = 0
            frames[frames >= len(self.ecg_signal)] = len(self.ecg_signal) - 1

            ecg_frame = self.ecg_signal[frames]
            mean_amp = np.abs(np.min(ecg_frame, axis=-1) - np.max(ecg_frame, axis=-1))
            quad_amp = np.quantile(mean_amp, 0.82)
            
            check_amp = quad_amp <= 0.2
            check_percentage = (np.count_nonzero(mean_amp < 0.22) / len(beat_valid)) >= 0.7  # 70%
            if not (check_amp and check_percentage):
                self.beat_amp_threshold.median_amp[0] = min([
                    quad_amp / 2,
                    self.beat_amp_threshold.median_amp[0] * 2
                ])
                
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            
    def _is_low_amplitude(
            self,
            begin:  int,
            end:    int
    ) -> bool:
        
        is_low_amplitude = False
        try:
            mean_amp = np.abs(np.min(self.ecg_bandpass[begin: end]) - np.max(self.ecg_bandpass[begin: end]))
            
            #
            cond1 = mean_amp <= self.beat_amp_threshold.median_amp[0]
            
            #
            cond2 = np.abs(np.min(self.ecg_bandpass[begin: end])) < self.beat_amp_threshold.min_amp
            
            is_low_amplitude = cond1 & cond2
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return is_low_amplitude
    
    def _is_not_zero_line(
            self,
            begin:  int,
            end:    int
    ) -> bool:
        
        is_not_zero_line = False
        try:
            
            asc_sig = np.round(np.sort(np.abs(self.ecg_signal[begin: end])), 3)
            
            # 60%, meaning 60% of the signal amplitude is above 0.0018mV.
            cond1 = asc_sig[int(0.6 * len(asc_sig))] >= 0.0018
            
            # 30%, meaning he total number of zero values in the signal is less than 30% of the signal.
            zero_burden = (len(asc_sig) - np.count_nonzero(asc_sig)) / len(asc_sig)
            cond2 = zero_burden <= 0.3

            is_not_zero_line = cond1 & cond2
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return is_not_zero_line
    
    def _is_pause_valid(
            self,
            index:      NDArray
    ) -> bool:
        """
        Analyzes the ECG morphology within a specified range to determine
        if a low beat or zero line condition is met.
        """
        
        is_pause_valid = False
        try:
            index_around = index + np.arange(-self.BEAT_AROUND, self.BEAT_AROUND + 1)
            index_around = index_around[np.flatnonzero(np.logical_and(
                    index_around >= 0,
                    index_around < len(self.data_structure.beat)
            ))]
            if index_around.size == 0:
                return is_pause_valid
            
            index_around = np.column_stack([index_around[:-1], index_around[1:]])
            index_around = index_around[np.flatnonzero(index_around[:, 0] != index)]
            if index_around.size == 0:
                return is_pause_valid
            
            if len(index_around) > 0:
                index_around = index_around[np.flatnonzero(np.logical_and(
                        np.isin(self.data_structure.symbol[index_around][:, 0], df.VALID_BEAT_TYPE),
                        np.isin(self.data_structure.symbol[index_around][:, 1], df.VALID_BEAT_TYPE),
                ))]
                if index_around.size == 0:
                    return is_pause_valid
                
                rr = np.mean(np.diff(self.data_structure.beat[index_around], axis=-1))
                begin = self.data_structure.beat[index] + int(rr * self.RATIO_TO_T_WAVE_DETECT)
                end = self.data_structure.beat[index + 1] - int(rr * self.RATIO_TO_P_WAVE_DETECT)
                
                if begin >= end:
                    begin = self.data_structure.beat[index] + int(self.record_config.sampling_rate * self.BEAT_OFFSET)
                    end = self.data_structure.beat[index + 1] - int(self.record_config.sampling_rate * self.BEAT_OFFSET)

            else:
                begin = self.data_structure.beat[index] + int(self.record_config.sampling_rate * self.BEAT_OFFSET)
                end = self.data_structure.beat[index + 1] - int(self.record_config.sampling_rate * self.BEAT_OFFSET)

            is_low_amplitude = self._is_low_amplitude(begin, end)
            is_not_zero_line = self._is_not_zero_line(begin, end)
            is_pause_valid = is_low_amplitude & is_not_zero_line

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return is_pause_valid
    
    def _get_invalid_region(
            self
    ) -> NDArray:
        
        region = np.ones_like(self.data_structure.beat)
        try:
            region |= self.data_structure.lead_off
            
            # region Ignore ARTIFACT
            for (hes_id, rhythm_id) in [
                [df.HOLTER_ARTIFACT, cf.RHYTHMS_DATASTORE['classes']['OTHER']],
            ]:
                if self.is_hes_process:
                    index = df.check_hes_event(self.data_structure.rhythm, hes_id)
                else:
                    index = self.data_structure.rhythm == rhythm_id
                
                index = np.flatnonzero(index)
                if len(index) > 0:
                    region[index] = self.INVALID
            # endregion Ignore ARTIFACT
            
            # region Ignore INVALID BEAT TYPE
            index = np.flatnonzero(np.isin(self.data_structure.symbol, df.INVALID_BEAT_TYPE))
            if len(index) > 0:
                region[index] = self.INVALID
            # endregion Ignore INVALID BEAT TYPE

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return region
    
    def _detect(
            self,
    ) -> None:
        
        try:
            group_valid_region = df.get_group_from_index_event(np.flatnonzero(self.region != self.INVALID))
            for index in group_valid_region:
                if len(index) < self.COUNT_BEAT_PAUSE_VALID:
                    continue
                
                hr = df.SECOND_IN_MINUTE * self.record_config.sampling_rate / np.diff(self.data_structure.beat[index])
                index_pause_valid_by_hr = np.flatnonzero(np.logical_and(
                        hr >= self.min_bpm,
                        hr <= self.max_bpm
                ))
                if len(index_pause_valid_by_hr) > 0:
                    self.pause_events[index[index_pause_valid_by_hr]] = self.MARKED_ID

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    def _verify(
            self,
    ) -> None:
        
        try:
            ind_pause = np.flatnonzero(self.pause_events == self.MARKED_ID)
            self.pause_events[ind_pause] = 0
            if len(ind_pause) == 0:
                return
            
            ind_pause = np.array(list(filter(self._is_pause_valid, ind_pause)))
            if len(ind_pause) == 0:
                return
            
            self.pause_events[ind_pause[:, None]] |= df.HOLTER_PAUSE
            pass
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    def _plot(
            self
    ) -> None:
        try:
            fig, (axis) = plt.subplots(2, 1, figsize=(19.2, 10.08), sharex='all')
            fig.suptitle(f'Pause Detection - {self.record_config.record_path}', fontsize=13)
            fig.subplots_adjust(hspace=0, wspace=0.2, left=0.04, right=0.98, bottom=0.05, top=0.93)
            
            i = 0
            times = np.arange(len(self.ecg_signal)) / self.record_config.sampling_rate
            axis[i].plot(times, self.ecg_signal, label='ECG Signal')
            
            index_valid_region = self.data_structure.beat[np.flatnonzero(self.region != self.INVALID)]
            amp_pos = min(np.min(self.ecg_signal[index_valid_region]), df.MAX_STRIP_LEN)
            axis[i].vlines(
                    self.data_structure.beat / self.record_config.sampling_rate,
                    ymin=-amp_pos - 0.3,
                    ymax=amp_pos,
                    lw=0.5,
                    color='r',
                    linestyles='dotted'
            )

            [
                axis[i].annotate(
                        sym,
                        xy=(beat / self.record_config.sampling_rate, amp_pos),
                        xycoords='data',
                        textcoords='data',
                        bbox=dict(
                                boxstyle='round',
                                fc=df.BEAT_COLORS.get(sym, 'white'),
                                ec=df.BEAT_COLORS_EC.get(sym, 'blue')
                        )
                )
                for beat, sym in zip(self.data_structure.beat, self.data_structure.symbol)
            ]
            
            group_pause_events = df.get_group_index_event(self.pause_events, df.HOLTER_PAUSE)
            for index in group_pause_events:
                axis[i].axvspan(
                        self.data_structure.beat[index[0]] / self.record_config.sampling_rate,
                        self.data_structure.beat[index[-1]] / self.record_config.sampling_rate,
                        color='yellow',
                        alpha=0.5
                )
            
            group_invalid_events = df.get_group_index_event(self.region, self.INVALID)
            for index in group_invalid_events:
                axis[i].axvspan(
                        self.data_structure.beat[index[0]] / self.record_config.sampling_rate,
                        self.data_structure.beat[index[-1]] / self.record_config.sampling_rate,
                        color='gray',
                        alpha=0.5
                )
            
            axis[i].set_ylabel('Amplitude (mV)', color='red', fontsize=12)
            axis[i].grid(color='gray', linestyle='--', linewidth=0.5)

            i += 1
            axis[i].set_xlabel('Times (s)', color='red', fontsize=12)
            axis[i].set_ylabel('Heat Rate (bpm)', color='red', fontsize=12)

            rr_intervals = np.diff(self.data_structure.beat) / self.record_config.sampling_rate
            hr = df.SECOND_IN_MINUTE / rr_intervals
            hr_pos = np.diff(self.data_structure.beat) / 2 + self.data_structure.beat[:-1]
            hr_pos = hr_pos / self.record_config.sampling_rate
            
            axis[i].plot(hr_pos, hr,  marker='o', linestyle='-', color='green', alpha=0.3)
            axis[i].axhline(y=self.max_bpm, color='blue')
            axis[i].annotate(
                    f'Pause Threshold: {self.min_bpm} - {self.max_bpm} bpm',
                    xy=(0.1, self.max_bpm),
                    xycoords=('axes fraction', 'data'),
                    textcoords=('axes fraction', 'data'),
                    bbox=dict(boxstyle="round", fc="white"),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="angle,angleA=0,angleB=90,rad=10",
                    )
            )
            
            axis[i].grid(color='gray', linestyle='--', linewidth=0.5)
            axis[i].set_xlim(times[0], times[-1])
            plt.show()
            plt.close()
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    def process(
            self,
            plot:  bool = False
    ) -> [NDArray, NDArray]:
        try:
            self.region = self._get_invalid_region()
            if np.all(self.region == self.INVALID):
                return self.pause_events
            
            self._get_ecg_signals()
            self._determine_low_amplitude_threshold()
            self._detect()
            self._verify()
            
            plot and cf.DEBUG_MODE and self._plot()
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return self.pause_events
