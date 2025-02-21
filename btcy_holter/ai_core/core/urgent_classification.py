from btcy_holter import *


class UrgentClassification(
    ut.TFServing
):
    _TFSERVER_TIMEOUT:  Final[float] = 60.0  # seconds

    def __init__(
            self,
            **kwargs
    ) -> None:
        try:
            super(UrgentClassification, self).__init__(
                model_spec=cf.TF_URGENT_MODEL,
                datastore=cf.URGENT_DATASTORE,
                timeout_sec=self._TFSERVER_TIMEOUT
            )
            
            self._start_sample:          Final[int] = 10  # seconds
    
            try:
                self._log_process_time = kwargs['log_process_time']
            except (Exception,):
                self._log_process_time = False
    
        except (Exception, ) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    # @df.timeit
    def _pre_process(
            self,
            ecg_signal:     NDArray,
            ecg_signal_fs:  int
    ) -> NDArray:

        p_signal = ecg_signal.copy()
        try:
            if ecg_signal_fs != self.datastore['sampling_rate']:
                p_signal, _ = resample_sig(
                        p_signal,
                        fs=ecg_signal_fs,
                        fs_target=self.datastore['sampling_rate']
                )

            sig_len = len(p_signal)
            if len(p_signal) < self.standard_event_length * self.datastore['sampling_rate']:
                begin = max(0, int((sig_len // 2) - self.datastore['feature_len']))
                end = min(int((sig_len // 2) + self.datastore['feature_len']), sig_len)
                if end - begin < self.datastore['feature_len'] * 2:
                    seg = np.concatenate((
                        p_signal[begin: end],
                        np.zeros(self.datastore['feature_len'] * 2 - (end - begin))
                    ))
                else:
                    seg = p_signal[begin: end]

                p_signal = np.asarray([seg for _ in range(3)])

            else:
                frames = np.arange(1, 4) * self._start_sample
                frames = (np.arange(self.datastore['feature_len'] * 2)[None, :] +
                          (frames * self.datastore['sampling_rate'])[:, None])
                p_signal = p_signal[frames]

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return p_signal

    # @df.timeit
    def prediction(
            self,
            ecg_signal:     NDArray,
            ecg_signal_fs:  int = cf.SAMPLING_RATE,
    ) -> int:
        predict = 0
        try:
            start_process_time = time.monotonic()
            process_time = dict()

            # region Pre-process
            start = time.monotonic()
            process_data = self._pre_process(
                ecg_signal=ecg_signal,
                ecg_signal_fs=ecg_signal_fs
            )
            process_data = np.expand_dims(process_data, axis=-1)
            process_time = df.get_update_time_process(process_time, start, key='preProcess')
            # endregion Pre-process

            # region GRPC request tf server
            start = time.monotonic()
            output = self.make_and_send_grpc_request(data=process_data)
            process_time = df.get_update_time_process(process_time, start, key='tfServer')
            # endregion GRPC request tf server

            # region Post-processing
            start = time.monotonic()
            predict = np.reshape(output, (-1, len(self.datastore['classes'].keys())))
            predict = np.max(np.argmax(predict, axis=-1).flatten())
            process_time = df.get_update_time_process(process_time, start, key='postProcess')
            # endregion Post-processing

            self._log_process_time and self.log_time(process_time)
            self.close_channel()
            self.update_performance_time(
                step=1,
                start_process_time=start_process_time,
                process_time=process_time
            )

        except (Exception, ) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return predict
