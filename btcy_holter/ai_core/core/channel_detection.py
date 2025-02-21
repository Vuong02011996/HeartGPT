from btcy_holter import *


class ChannelDetection(
    ut.TFServing
):
    _TFSERVER_TIMEOUT:  Final[float] = 60.0  # seconds
    
    def __init__(
            self,
            is_process_event:      bool = False,
            **kwargs
    ) -> None:
        try:
            super(ChannelDetection, self).__init__(
                model_spec=cf.TF_CHANNEL_MODEL,
                datastore=cf.CHANNELS_DATASTORE,
                is_process_event=is_process_event,
                timeout_sec=self._TFSERVER_TIMEOUT
            )
    
            self._channel_class_invert:      Final[Dict] = {i: k for k, i in self.datastore['classes'].items()}
            self._default_channel_valid:    Final[int] = cf.CHANNEL
    
            try:
                self._log_process_time = kwargs['log_process_time']
            except (Exception,):
                self._log_process_time = False
    
            try:
                self._channel_bandpass_filter = self.datastore['BANDPASS_FILTER']
            except (Exception,):
                st.get_error_exception('Missing BANDPASS_FILTER in the configuration')
    
            try:
                self._channel_clip_range = self.datastore['R_PEAK_AMPLITUDE']
            except (Exception,):
                self._channel_clip_range = None
                
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    def _get_ecg_signals(
            self,
            ecg_signal:     NDArray,
            ecg_signal_fs:  int,
            **kwargs
    ) -> [NDArray, int]:
        try:
            if isinstance(ecg_signal, np.ndarray) and isinstance(ecg_signal_fs, int):
                return ecg_signal, ecg_signal_fs
            
            if 'record_config' in kwargs.keys():
                record_config = kwargs['record_config']
                record_path = record_config.record_path
                if record_path is None:
                    st.get_error_exception('Error: Record path is None.')
                
                if kwargs.get('included_header', False) or df.check_hea_file(record_config.record_path):
                    record = wf.rdrecord(record_config.record_path)
                    ecg_signal = np.nan_to_num(record.p_signal)
                    record_config.sampling_rate = record.fs
                
                else:
                    ecg_signal = ut.get_data_from_dat(
                            file=record_config.record_path,
                            record_config=record_config
                    )
                
                ecg_signal_fs = record_config.sampling_rate
            else:
                st.get_error_exception('Invalid input.')
                
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return ecg_signal, ecg_signal_fs
        
    # @df.timeit
    def _pre_process(
            self,
            ecg_signal:         NDArray,
            ecg_signal_fs:      int,
    ) -> NDArray:

        p_signal = deepcopy(ecg_signal)
        try:

            if ecg_signal_fs != self.datastore['sampling_rate']:
                p_signal = np.asarray(list(map(
                    lambda x: resample_sig(
                            x=x,
                            fs=ecg_signal_fs,
                            fs_target=self.datastore['sampling_rate']
                    )[0],
                    p_signal.T
                )))
    
            p_signal = ut.butter_bandpass_filter(
                    p_signal,
                    *self._channel_bandpass_filter,
                    self.datastore['sampling_rate']
            )
            
            if self._channel_clip_range:
                p_signal = np.clip(p_signal, *self._channel_clip_range)
    
            if self.datastore['bwr']:
                p_signal = ut.bwr_smooth(p_signal, self.datastore['sampling_rate'])
    
            if self.datastore['norm']:
                p_signal = ut.norm(
                        p_signal,
                        window_len=int(self.datastore['NUM_NORM'] * self.datastore['sampling_rate'])
                )
    
            if self.datastore['norm'] and self._channel_clip_range:
                p_signal = (p_signal + 1.0) / 2.0

            elif self._channel_clip_range:
                for chn in range(len(p_signal)):
                    if (max(p_signal[chn]) - min(p_signal[chn])) > 0:
                        p_signal[chn] = (p_signal[chn] - min(p_signal[chn])) / (max(p_signal[chn]) - min(p_signal[chn]))
    
            p_signal = p_signal.T
            redundant = len(p_signal) % self.datastore['feature_len']
            if redundant:
                p_signal = np.concatenate([
                    p_signal,
                    np.zeros((self.datastore['feature_len'] - redundant, p_signal.shape[-1]))
                ])

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return p_signal

    def _post_processing(
            self,
            channel_predicts:   NDArray,
            sig_len:            int,
            fs:                 int,
            **kwargs
    ) -> int | None:
        channel_selected = None
        try:

            try:
                valid_channel = kwargs['valid_channel']
            except (Exception,):
                valid_channel = False

            try:
                ignore_channel_low_amp = kwargs['ignore_channel_low_amp']
            except (Exception,):
                ignore_channel_low_amp = None

            try:
                config_segment = kwargs['config_segment']
            except (Exception,):
                config_segment = None

            channel_len = Counter(channel_predicts)
            if config_segment is None:
                channel_selected = max(channel_len.items(), key=lambda x: x[1])[0]

                if ignore_channel_low_amp is not None and len(np.unique(channel_predicts)) > 1:
                    list_selection_len = sorted(channel_len.items(), key=operator.itemgetter(1))[::-1]
                    for channel_id, i in list_selection_len:
                        c = self._channel_class_invert[channel_id]
                        if c == 'LEADOFF':
                            channel_selected = self.datastore['classes']['LEADOFF']
                            break

                        ch = int(c[-1]) - 1
                        if ch not in ignore_channel_low_amp:
                            continue
                        else:
                            channel_selected = ch
                            break

                if channel_selected == self.datastore['classes']['LEADOFF'] and valid_channel:
                    channel_selected = channel_len.most_common()[1][0] if len(channel_len.keys()) > 1 else cf.CHANNEL

            else:
                unique_channels = np.unique(channel_predicts)
                if len(unique_channels) == 1:
                    channel = channel_predicts[0]
                    if channel == self.datastore['classes']['LEADOFF']:
                        channel = cf.CHANNEL

                    channel_selected = np.ones(sig_len, dtype=int) * channel

                elif len(unique_channels) == 2 and self.datastore['classes']['LEADOFF'] in unique_channels:
                    channel = unique_channels[unique_channels != self.datastore['classes']['LEADOFF']]
                    channel_selected = np.ones(sig_len, dtype=int) * channel

                else:
                    if len(channel_predicts) % config_segment != 0:
                        redundant = np.ones(config_segment - len(channel_predicts) % config_segment, dtype=int)
                        channel_predicts = np.concatenate((channel_predicts, redundant * channel_predicts[-1]))

                    frame = np.arange(0, len(channel_predicts), config_segment)[:, None] + np.arange(config_segment)
                    channel_segments = np.apply_along_axis(
                        lambda x: np.argmax(np.bincount(x)), axis=1,
                        arr=channel_predicts[frame]
                    )
                    lead_off_index = np.flatnonzero(channel_segments == self.datastore['classes']['LEADOFF'])
                    if len(lead_off_index) > 0 and len(channel_segments) > 2:
                        tmp = lead_off_index[np.flatnonzero(np.logical_and(
                            lead_off_index > 0,
                            lead_off_index < len(channel_segments) - 1
                        ))]
                        if len(tmp) > 0:
                            index = np.flatnonzero(np.logical_and(
                                channel_segments[tmp - 1] != self.datastore['classes']['LEADOFF'],
                                channel_segments[tmp + 1] != self.datastore['classes']['LEADOFF']
                            ))
                            channel_segments[tmp[index]] = channel_segments[tmp[index] - 1]

                        tmp = lead_off_index[np.flatnonzero(lead_off_index == 0)]
                        if len(tmp) > 0:
                            if channel_segments[tmp + 1] != self.datastore['classes']['LEADOFF']:
                                channel_segments[0] = channel_segments[1]

                        tmp = lead_off_index[np.flatnonzero(lead_off_index == len(channel_segments) - 1)]
                        if len(tmp) > 0:
                            if channel_segments[tmp - 1] != self.datastore['classes']['LEADOFF']:
                                channel_segments[len(channel_segments)-1] = channel_segments[len(channel_segments) - 2]

                        lead_off_index = np.flatnonzero(channel_segments == self.datastore['classes']['LEADOFF'])

                    if len(lead_off_index) > 0:
                        ind = np.delete(np.arange(len(channel_segments)), lead_off_index)
                        if len(ind) > 0:
                            channel_segments[lead_off_index] = np.argmax(np.bincount(channel_segments[ind]))

                    unique_channels = np.unique(channel_segments)
                    if len(unique_channels) == 1:
                        channel = channel_segments[0]
                        if channel == self.datastore['classes']['LEADOFF']:
                            channel = cf.CHANNEL

                        channel_selected = np.ones(sig_len, dtype=int) * channel

                    elif len(unique_channels) == 2 and self.datastore['classes']['LEADOFF'] in unique_channels:
                        channel = unique_channels[unique_channels != self.datastore['classes']['LEADOFF']]
                        channel_selected = np.ones(sig_len, dtype=int) * channel

                    else:
                        config_sample = config_segment * fs
                        frames = (np.arange(len(channel_segments)) * config_sample)[:, None] + np.arange(config_sample)
                        channel_segments = np.tile(channel_segments.reshape((-1, 1)), (1, frames.shape[1])).flatten()

                        index_frames = frames.flatten()
                        index = np.flatnonzero(index_frames < sig_len)

                        channel_selected = np.ones(sig_len, dtype=int) * -1
                        channel_selected[index_frames[index]] = channel_segments[index]

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return channel_selected

    # @df.timeit
    def prediction(
            self,
            ecg_signal:     NDArray = None,
            ecg_signal_fs:  int = None,
            **kwargs
    ) -> int:

        process_time = dict()
        channel_cleanest = cf.CHANNEL
        try:
            start_process_time = time.monotonic()

            # region Process when input is file path
            ecg_signal, ecg_signal_fs = self._get_ecg_signals(
                    ecg_signal=ecg_signal,
                    ecg_signal_fs=ecg_signal_fs,
                    **kwargs
            )
            # endregion Process when input is file path

            # region Predict
            fs_ratio = self.datastore['sampling_rate'] / ecg_signal_fs
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

            pre_ecg_signal = self._pre_process(
                ecg_signal=ecg_signal,
                ecg_signal_fs=ecg_signal_fs,
            )
            total_channel = pre_ecg_signal.shape[1]
            signal_in_second = int(len(ecg_signal) / ecg_signal_fs)

            step = 0
            sample_to = 0
            sample_from = 0

            total_channel_predicts = list()
            while sample_to < len(ecg_signal):
                sample_len = min(max_sample_process, (len(ecg_signal) - sample_from))
                if sample_len <= 0:
                    break

                step += 1
                # region Pre-process
                start = time.monotonic()
                sample_to = sample_from + sample_len
                
                buf_ecg = pre_ecg_signal[int(sample_from * fs_ratio): int(sample_to * fs_ratio)]

                data_index = (np.arange(self.datastore['feature_len'])[None, :] +
                              np.arange(0, len(buf_ecg), self.datastore['feature_len'])[:, None])
                data_index[data_index > len(buf_ecg) - 1] = len(buf_ecg) - 1
                len_data = len(data_index)

                data_index = np.concatenate(list(map(lambda f: data_index, np.arange(total_channel))))
                channel_frames = np.hstack(np.arange(total_channel)[:, None] + np.zeros(len_data, dtype=int))

                data_frames = buf_ecg[data_index, channel_frames[:, None]]
                process_data = np.split(data_frames, np.arange(len_data, len(data_index), len_data))
                process_data = np.expand_dims(np.concatenate(process_data, axis=1), axis=-1)
                process_time = df.get_update_time_process(process_time, start, key='preProcess')
                # endregion Pre-process

                # region GRPC request tf server
                start = time.monotonic()
                output = self.make_and_send_grpc_request(data=process_data)
                process_time = df.get_update_time_process(process_time, start, key='tfServer')
                # endregion GRPC request tf server

                # region Post-processing
                start = time.monotonic()
                channel_prob = np.reshape(output, (-1, len(self.datastore['classes'])))[:signal_in_second]
                channel_predict = np.argmax(channel_prob, axis=-1).flatten()
                if len(total_channel_predicts) == 0:
                    total_channel_predicts = channel_predict
                else:
                    total_channel_predicts = np.concatenate((total_channel_predicts, channel_predict))

                process_time = df.get_update_time_process(process_time, start, key='postProcess')
                sample_from = sample_to
                # endregion Post-processing

            self.close_channel()

            start = time.monotonic()
            channel_cleanest = self._post_processing(
                channel_predicts=total_channel_predicts,
                sig_len=len(ecg_signal),
                fs=ecg_signal_fs,
                **kwargs
            )

            process_time = df.get_update_time_process(process_time, start, key='validChannel')
            # endregion Predict

            self._log_process_time and self.log_time(process_time)
            self.update_performance_time(
                step=step, 
                start_process_time=start_process_time, 
                process_time=process_time
            )

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return channel_cleanest
