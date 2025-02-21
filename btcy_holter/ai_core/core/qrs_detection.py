from btcy_holter import *


class QRSDetection(
    ut.TFServing
):
    _TFSERVER_TIMEOUT:  Final[float] = 60.0  # seconds
    
    def __init__(
            self,
            **kwargs
    ) -> None:
        
        try:
            super(QRSDetection, self).__init__(
                model_spec=cf.TF_QRS_MODEL,
                datastore=cf.QRS_DATASTORE,
                timeout_sec=self._TFSERVER_TIMEOUT
            )
            
            self._beat_offset = (np.array(self.datastore['beat_offset']) * self.datastore['sampling_rate']).astype(int)
            self._md_feature_len = np.sum(self._beat_offset)
    
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
    
            self._beat_offset_frame = self.datastore['OFFSET_FRAME_BEAT']
            
        except (Exception, ) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    # @df.timeit
    def _pre_process(
            self,
            ecg_signal:     NDArray,
            ecg_signal_fs:  int,
    ) -> NDArray:

        p_signal = ecg_signal.copy()
        try:
            if ecg_signal_fs != self.datastore['sampling_rate']:
                p_signal, _ = resample_sig(p_signal, ecg_signal_fs, self.datastore['sampling_rate'])

            redundant = len(p_signal) % self._md_feature_len
            if redundant:
                p_signal = np.concatenate([p_signal, np.zeros(self._md_feature_len - redundant)])

            p_signal = ut.butter_bandpass_filter(p_signal, *self._beat_bandpass_filter, self.datastore['sampling_rate'])
            if self._beat_clip_range is not None:
                p_signal = np.clip(p_signal, *self._beat_clip_range)

            bwr_buf_ecg = ut.bwr(p_signal, self.datastore['sampling_rate'])
            p_signal = ut.norm(
                    bwr_buf_ecg,
                    window_len=int(self.datastore['NUM_NORM'] * self.datastore['sampling_rate'])
            )

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return p_signal

    # @df.timeit
    def prediction(
            self,
            beat:           NDArray,
            ecg_signal:     NDArray,
            ecg_signal_fs:  int
    ) -> NDArray:

        qrs = np.zeros_like(beat)
        process_time = dict()
        try:
            start_process_time = time.monotonic()

            # region Pre-process
            start = time.monotonic()
            pre_ecg_signal = self._pre_process(
                ecg_signal=ecg_signal,
                ecg_signal_fs=ecg_signal_fs
            )

            frames = np.arange(-self._beat_offset[0], self._beat_offset[-1])[None, :]
            data_index_frame = np.concatenate(list(map(lambda f: frames + f, self._beat_offset_frame)))
            buf_frames = np.array(list(chain.from_iterable(map(lambda x: x + data_index_frame, beat))))
            buf_frames = buf_frames.reshape((-1, self._md_feature_len))

            buf_frames[buf_frames < 0] = 0
            buf_frames[buf_frames >= len(pre_ecg_signal)] = len(pre_ecg_signal) - 1

            process_data = np.expand_dims(pre_ecg_signal[buf_frames], axis=-1)
            process_time = df.get_update_time_process(process_time, start, key='preProcess')
            # endregion Pre-process

            # region GRPC request tf server
            start = time.monotonic()
            output = self.make_and_send_grpc_request(data=process_data, output_type='int')
            process_time = df.get_update_time_process(process_time, start, key='tfServer')
            # endregion GRPC request tf server

            # region PostProcess
            output = output.reshape((-1, len(self._beat_offset_frame)))
            qrs = np.array(list(map(lambda x: Counter(x).most_common()[0][0], output)))

            self._log_process_time and self.log_time(process_time)
            self.close_channel()
            self.update_performance_time(
                step=1,
                start_process_time=start_process_time,
                process_time=process_time
            )
            # endregion PostProcess

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return qrs
