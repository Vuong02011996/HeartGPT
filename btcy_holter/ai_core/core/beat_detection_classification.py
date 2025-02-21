from btcy_holter import *


class BeatsDetectionAndClassification(
    ut.TFServing
):
    _TFSERVER_TIMEOUT:  Final[float] = 60.0
    
    def __init__(
            self,
            is_process_event: bool = False,
            **kwargs
    ) -> None:
        try:
            super(BeatsDetectionAndClassification, self).__init__(
                model_spec=cf.TF_BEAT_MODEL,
                timeout_sec=self._TFSERVER_TIMEOUT,
                datastore=cf.BEATS_DATASTORE,
                is_process_event=is_process_event
            )
            self._label_len:             Final[int] = self.datastore['feature_len'] // self.datastore['NUM_BLOCK']
            self._min_rr_interval:       Final[float] = cf.MIN_RR_INTERVAL * self.datastore['sampling_rate']
            
            self.max_sample:            Final[int] = cf.MAX_SAMPLE_PROCESS * self.datastore['sampling_rate']
            self._beat_offset_frame:    Any = self.datastore['OFFSET_FRAME_BEAT']

            try:
                self._log_process_time = kwargs['log_process_time']
            except (Exception,):
                self._log_process_time = False
            
            try:
                self._beat_bandpass_filter = self.datastore['BANDPASS_FILTER']
            except (Exception,):
                st.get_error_exception('Missing BANDPASS_FILTER in the configuration')
            
            try:
                self._beat_clip_range = self.datastore['R_PEAK_AMPLITUDE']
            except (Exception,):
                self._beat_clip_range = None

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _pre_process(
            self,
            ecg_signal:         NDArray,
            ecg_signal_fs:      int,
            max_sample_process: int
    ) -> NDArray:
        
        pre_ecg_signal = ecg_signal.copy()
        try:
            if len(pre_ecg_signal) < max_sample_process:
                pre_ecg_signal = np.concatenate([pre_ecg_signal, np.zeros(max_sample_process - len(pre_ecg_signal))])
            
            if ecg_signal_fs != self.datastore['sampling_rate']:
                pre_ecg_signal, _ = resample_sig(
                    pre_ecg_signal,
                    ecg_signal_fs,
                    self.datastore['sampling_rate']
                )
            
            redundant = len(pre_ecg_signal) % self.datastore['feature_len']
            if redundant:
                pre_ecg_signal = np.concatenate([pre_ecg_signal, np.zeros(self.datastore['feature_len'] - redundant)])
            
            pre_ecg_signal = ut.butter_bandpass_filter(
                    pre_ecg_signal,
                    *self._beat_bandpass_filter,
                    self.datastore['sampling_rate']
            )
            
            if self._beat_clip_range is not None:
                pre_ecg_signal = np.clip(pre_ecg_signal, *self._beat_clip_range)
            
            if self.datastore['bwr']:
                pre_ecg_signal = ut.bwr_smooth(
                        pre_ecg_signal,
                        fs=self.datastore['sampling_rate']
                )
            
            if self.datastore['norm']:
                pre_ecg_signal = ut.norm(
                        pre_ecg_signal,
                        window_len=int(self.datastore['NUM_NORM'] * self.datastore['sampling_rate'])
                )
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return pre_ecg_signal
    
    @staticmethod
    def _find_max_amp(
            symbols:    NDArray,
            amps:       NDArray,
            index:      Any
    ) -> [int, str]:
        index = np.arange(index.start(0), min(index.end(0), len(symbols)))
        max_amp = index[np.argmax(amps[index])]
        
        unique, counts = np.unique(symbols[index], return_counts=True)
        most_common_idx = np.argmax(counts)
        sym = unique[most_common_idx]
        
        return max_amp, sym
    
    # @df.timeit
    def _post_process(
            self,
            beats:      NDArray,
            symbols:    NDArray,
            amps:       NDArray
    ) -> Any:
        
        beats_valid = None
        symbols_valid = None
        amps_valid = None
        
        try:
            group_beats = np.zeros_like(beats)
            group_beats[np.flatnonzero(np.diff(beats) <= self._min_rr_interval) + 1] = 1
            group_beats = group_beats.astype(str)
            
            frame = [(
                [0, b.start(), idx]
                if min(b.end(), len(beats)) - b.start() == 1
                else [1, *self._find_max_amp(symbols, amps, b)])
                for idx, b in enumerate(re.finditer('01+|0', ''.join(group_beats)))]
            
            frame = np.asarray(frame)
            len_frame = len(frame)
            amps_valid = np.zeros(len_frame, dtype=float)
            beats_valid = np.zeros(len_frame, dtype=int)
            symbols_valid = np.zeros(len_frame, dtype=str)
            
            idx = np.flatnonzero(np.array(frame[:, 0], dtype=int) == 0)
            tmp_index = np.array(frame[idx, 1], dtype=int)
            org_index = np.array(frame[idx, -1], dtype=int)
            
            amps_valid[org_index] = amps[tmp_index]
            beats_valid[org_index] = beats[tmp_index]
            symbols_valid[org_index] = symbols[tmp_index]
            
            org_index = np.delete(np.arange(len(beats_valid)), idx)
            idx = np.flatnonzero(np.array(frame[:, 0], dtype=int) == 1)
            tmp_index = np.array(frame[idx, 1], dtype=int)
            
            amps_valid[org_index] = amps[tmp_index]
            beats_valid[org_index] = beats[tmp_index]
            symbols_valid[org_index] = np.array(frame[idx, -1])
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return beats_valid, symbols_valid, amps_valid
    
    # @df.timeit
    def prediction(
            self,
            ecg_signal:     NDArray,
            ecg_signal_fs:  int
    ) -> Any:
        
        total_beat = list()
        total_symbol = list()
        
        try:
            start_process_time = time.monotonic()
            process_time = dict()
            
            beat_inv = {i: k for k, i in self.datastore['classes'].items()}
            
            sample_length = self.standard_event_length * ecg_signal_fs
            max_sample_process = sample_length * int(self.max_sample_process / df.SECOND_IN_MINUTE)
            if (
                    len(ecg_signal) is not None
                    and max_sample_process != len(ecg_signal)
                    and self.is_process_event
            ):
                if len(ecg_signal) % sample_length == 0:
                    max_sample_process = int(len(ecg_signal) // sample_length) * sample_length
                else:
                    max_sample_process = (int(len(ecg_signal) // sample_length) + 1) * sample_length
            
            fs_ratio = self.datastore['sampling_rate'] / ecg_signal_fs
            pre_ecg_signal = self._pre_process(
                    ecg_signal=ecg_signal,
                    ecg_signal_fs=ecg_signal_fs,
                    max_sample_process=max_sample_process
            )
            
            step = 0
            sample_from = 0
            sample_to = 0
            while sample_to < len(ecg_signal):
                try:
                    # region Pre-process
                    step += 1
                    start = time.monotonic()
                    
                    sample_len = min(max_sample_process, (len(ecg_signal) - sample_from))
                    sample_to = sample_from + sample_len
                    
                    fs_sample_from = int(sample_from * fs_ratio)
                    fs_sample_to = int(sample_to * fs_ratio)
                    buf_ecg = pre_ecg_signal[fs_sample_from: fs_sample_to]
                    
                    data_index = (np.arange(self.datastore['feature_len'])[None, :] +
                                  np.arange(0, len(buf_ecg), self.datastore['feature_len'])[:, None])
                    data_index[data_index > len(buf_ecg) - 1] = len(buf_ecg) - 1
                    
                    data_index_frame = np.concatenate(list(map(lambda f: data_index + f, self._beat_offset_frame)))
                    frame = np.concatenate((buf_ecg, np.zeros(max(self._beat_offset_frame))))[data_index_frame]
                    
                    len_of_frame = len(data_index_frame) // len(self._beat_offset_frame)
                    process_data = np.expand_dims(frame, axis=-1)
                    process_time = df.get_update_time_process(process_time, start, key='preProcess')
                    # endregion Pre-process
                    
                    # region GRPC request tf server
                    start = time.monotonic()
                    output = self.make_and_send_grpc_request(data=process_data)
                    process_time = df.get_update_time_process(process_time, start, key='tfServer')
                    # endregion GRPC request tf server
                    
                    # region Post-processing
                    start = time.monotonic()
                    group_beat_prob = np.reshape(output, newshape=(len(frame), -1, len(self.datastore['classes'])))
                    gr_candidate = np.argmax(group_beat_prob, axis=-1)
                    
                    beat_candidate = list()
                    for offset in range(len(self._beat_offset_frame)):
                        tmp = gr_candidate[(len_of_frame * offset): (len_of_frame * (offset + 1)), :]
                        if len(beat_candidate) == 0:
                            beat_candidate = tmp
                        else:
                            beat_candidate = np.concatenate((beat_candidate, tmp))
                    
                    data_index_frame = data_index_frame.reshape(
                            (len(data_index_frame), self.datastore['NUM_BLOCK'], self._label_len)
                    )
                    buf_frame = frame.reshape((len(data_index_frame), self.datastore['NUM_BLOCK'], self._label_len))
                    
                    beat_valid = beat_candidate.copy()
                    beat_valid[beat_valid > self.datastore['classes']['NOTABEAT']] = 1
                    
                    buf_frame_mean = np.mean(buf_frame, axis=2)
                    
                    peaks = np.abs(beat_valid * np.max(buf_frame, axis=2) - buf_frame_mean)
                    peaks = peaks.reshape((len(data_index_frame), self.datastore['NUM_BLOCK'], 1))
                    
                    inv_peaks = np.abs(beat_valid * np.max(buf_frame * -1, axis=2) - buf_frame_mean)
                    inv_peaks = inv_peaks.reshape((len(data_index_frame), self.datastore['NUM_BLOCK'], 1))
                    
                    mb = np.argmax(np.concatenate((peaks, inv_peaks), axis=2), axis=2)
                    mb = (beat_valid * np.argmax(buf_frame, axis=2) * ((mb * (-1)) + 1) +
                          beat_valid * np.argmax(buf_frame * -1, axis=2) * mb)
                    beats = (data_index_frame[:, :, 0] * beat_valid + mb).flatten()
                    
                    ind = np.flatnonzero(beats != 0)
                    beats = beats[ind]
                    symbols = np.array(list(map(
                            lambda x: beat_inv[x],
                            gr_candidate.flatten()[ind]
                    )))
                
                    if len(beats) > 0:
                        beats, unique_indices = np.unique(beats, return_index=True)
                        symbols = symbols[unique_indices]
                        
                        index_sort = np.argsort(beats)
                        symbols = np.asarray(symbols)[index_sort]
                        beats = np.asarray(beats, dtype=int)[index_sort]
                        
                        valid_beats = np.flatnonzero(beats < len(buf_ecg))
                        beats = beats[valid_beats]
                        symbols = symbols[valid_beats]
                        amps = np.abs(buf_ecg[beats])
                        
                        if len(beats) > 0:
                            beats, symbols, amps = self._post_process(beats, symbols, amps)
                    
                    if len(beats) > 0:
                        beats = df.resample_beat_samples(
                                samples=beats,
                                sampling_rate=self.datastore['sampling_rate'],
                                target_sampling_rate=ecg_signal_fs
                        )
                        beats += sample_from
                        
                        if len(total_beat) == 0 and len(beats) > self.datastore['NUM_BEAT_BEFORE']:
                            total_beat = beats
                            total_symbol = symbols
                        
                        elif len(beats) > self.datastore['NUM_BEAT_BEFORE']:
                            num_beat = -1
                            tm_beat = (beats[self.datastore['NUM_BEAT_BEFORE']] -
                                       total_beat[-self.datastore['NUM_BEAT_AFTER']])
                            
                            if tm_beat < self._min_rr_interval:
                                num_beat = self.datastore['NUM_BEAT_BEFORE']
                            
                            elif (
                                    len(beats) > self.datastore['NUM_BEAT_BEFORE']
                                    or (len(buf_ecg) <= self.max_sample
                                        and len(beats) <= self.datastore['NUM_BEAT_BEFORE'])
                            ):
                                num_beat = 0
                            
                            if num_beat >= 0:
                                inx = np.flatnonzero(beats[num_beat] - total_beat > self._min_rr_interval)
                                total_symbol = np.concatenate((total_symbol[inx], symbols[num_beat:]))
                                total_beat = np.concatenate((total_beat[inx], beats[num_beat:]))
                    
                    process_time = df.get_update_time_process(process_time, start, key='postProcess')
                    # endregion Post-processing
                    
                    if sample_from < (sample_to - int(self.datastore['NUM_OVERLAP'] * ecg_signal_fs)):
                        sample_to -= int(self.datastore['NUM_OVERLAP'] * ecg_signal_fs)
                    sample_from = sample_to
                
                except (Exception, ) as error:
                    st.get_error_exception(error, class_name=self.__class__.__name__)
            
            self._log_process_time and self.log_time(process_time)
            self.close_channel()
            
            total_beat = np.asarray(total_beat, dtype=int)
            beat_valid_index = np.flatnonzero(total_beat < len(ecg_signal))
            
            total_beat = total_beat[beat_valid_index]
            total_symbol = np.asarray(total_symbol)[beat_valid_index]
            
            self.update_performance_time(
                step=step,
                start_process_time=start_process_time,
                process_time=process_time
            )

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return total_beat, total_symbol
