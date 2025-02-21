from btcy_holter import *


class RhythmClassification(
    ut.TFServing
):
    
    _TFSERVER_TIMEOUT:      Final[float] = 60.0  # seconds

    def __init__(
            self,
            is_process_event:      bool = False,
            **kwargs
    ) -> None:
        
        try:
            super(RhythmClassification, self).__init__(
                model_spec=cf.TF_RHYTHM_MODEL,
                datastore=cf.RHYTHMS_DATASTORE,
                is_process_event=is_process_event,
                timeout_sec=self._TFSERVER_TIMEOUT
            )
    
            try:
                self._log_process_time = kwargs['log_process_time']
            except (Exception,):
                self._log_process_time = False
    
            try:
                self._rhythm_bandpass_filter = self.datastore['BANDPASS_FILTER']
            except (Exception,):
                st.get_error_exception('Missing BANDPASS_FILTER in the configuration')
    
            try:
                self._rhythm_clip_range = self.datastore['R_PEAK_AMPLITUDE']
            except (Exception,):
                self._rhythm_clip_range = None
                
        except (Exception, ) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    # @df.timeit
    def _pre_process(
            self,
            ecg_signal:         NDArray,
            ecg_signal_fs:      int,
            max_sample_process: int
    ) -> NDArray:

        p_signal = ecg_signal.copy()
        try:
            if len(p_signal) < max_sample_process:
                p_signal = np.concatenate([p_signal, np.zeros(max_sample_process - len(p_signal))])

            if ecg_signal_fs != self.datastore['sampling_rate']:
                p_signal, _ = resample_sig(p_signal, ecg_signal_fs, self.datastore['sampling_rate'])

            redundant = len(p_signal) % self.datastore['feature_len']
            if redundant:
                p_signal = np.concatenate([p_signal, np.zeros(self.datastore['feature_len'] - redundant)])

            p_signal = ut.butter_bandpass_filter(p_signal,
                                                 *self._rhythm_bandpass_filter,
                                                 self.datastore['sampling_rate'])

            if self._rhythm_clip_range is not None:
                p_signal = np.clip(p_signal, *self._rhythm_clip_range)

            if self.datastore['bwr']:
                p_signal = ut.bwr_smooth(p_signal, self.datastore['sampling_rate'])

            if self.datastore['norm']:
                p_signal = ut.norm(
                        p_signal,
                        window_len=int(self.datastore['NUM_NORM'] * self.datastore['sampling_rate'])
                )

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return p_signal
        
    # @df.timeit
    def prediction(
            self,
            ecg_signal:         NDArray,
            ecg_signal_fs:      int
    ) -> NDArray:

        total_rhythm_predict = list()
        process_time = dict()

        try:
            start_process_time = time.monotonic()

            buffer = self.standard_event_length * ecg_signal_fs
            max_sample_process = buffer * int(self.max_sample_process / df.SECOND_IN_MINUTE)
            
            if (
                    len(ecg_signal) is not None
                    and max_sample_process != len(ecg_signal)
                    and self.is_process_event
            ):
                if len(ecg_signal) % buffer == 0:
                    max_sample_process = int(len(ecg_signal) // buffer) * buffer
                else:
                    max_sample_process = (int(len(ecg_signal) // buffer) + 1) * buffer

            sample_to = 0
            sample_from = 0

            fs_ratio = self.datastore['sampling_rate'] / ecg_signal_fs
            pre_ecg_signal = self._pre_process(
                ecg_signal=ecg_signal,
                ecg_signal_fs=ecg_signal_fs,
                max_sample_process=max_sample_process
            )

            step = 0
            while sample_to < len(ecg_signal):
                sample_len = min(max_sample_process, (len(ecg_signal) - sample_from))
                if sample_len <= 0:
                    break

                step += 1
                # region Pre-process
                start = time.monotonic()
                sample_to = sample_from + sample_len

                ecg_segment = pre_ecg_signal[int(sample_from * fs_ratio): int(sample_to * fs_ratio)]

                data_index = (np.arange(self.datastore['feature_len'])[None, :] +
                              np.arange(0, len(ecg_segment), self.datastore['feature_len'])[:, None])
                data_index[data_index > len(ecg_segment) - 1] = len(ecg_segment) - 1

                process_data = np.expand_dims(ecg_segment[data_index], axis=-1)
                process_time = df.get_update_time_process(process_time, start, key='preProcess')
                # endregion Pre-process

                # region GRPC request tf server
                start = time.monotonic()
                output = self.make_and_send_grpc_request(data=process_data)
                process_time = df.get_update_time_process(process_time, start, key='tfServer')
                # endregion GRPC request tf server

                # region Post-processing
                start = time.monotonic()
                rhythm_prod = np.reshape(output, (-1, len(self.datastore['classes'])))
                rhythm_prod = rhythm_prod[:int(len(ecg_segment) / self.datastore['sampling_rate'])]
                rhythm_predict = np.argmax(rhythm_prod, axis=-1).flatten()

                if len(rhythm_predict) > 0:
                    if len(total_rhythm_predict) == 0:
                        total_rhythm_predict = rhythm_predict
                    else:
                        total_rhythm_predict = np.concatenate((
                            total_rhythm_predict[:-int(self.datastore['NUM_OVERLAP'])],
                            rhythm_predict
                        ))
                        
                process_time = df.get_update_time_process(process_time, start, key='postProcess')
                # endregion Post-processing

                if sample_from < (sample_to - int(self.datastore['NUM_OVERLAP'] * ecg_signal_fs)):
                    sample_to -= int(self.datastore['NUM_OVERLAP'] * ecg_signal_fs)
                sample_from = sample_to
            
            self._log_process_time and self.log_time(process_time)
            self.close_channel()

            self.update_performance_time(
                step=step,
                start_process_time=start_process_time,
                process_time=process_time
            )
        except (Exception, ) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return total_rhythm_predict
