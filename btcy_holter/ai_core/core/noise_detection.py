from btcy_holter import *


class NoiseDetection(
    ut.TFServing
):
    _TFSERVER_TIMEOUT:  Final[float] = 60.0  # seconds
    
    def __init__(
            self,
            is_process_event:      bool = False,
            **kwargs
    ) -> None:
        try:
            super(NoiseDetection, self).__init__(
                model_spec=cf.TF_NOISE_MODEL,
                datastore=cf.NOISE_DATASTORE,
                is_process_event=is_process_event,
                timeout_sec=self._TFSERVER_TIMEOUT
            )
    
            try:
                self._log_process_time = kwargs['log_process_time']
            except (Exception,):
                self._log_process_time = False
    
            try:
                self._bandpass_filter = self.datastore['BANDPASS_FILTER']
            except (Exception,):
                st.get_error_exception('Missing BANDPASS_FILTER in the configuration')
    
            try:
                self._clip_range = self.datastore['R_PEAK_AMPLITUDE']
            except (Exception,):
                self._clip_range = None
    
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
            
    @staticmethod
    def std_error(
            y_true:             NDArray,
            y_predict:          NDArray
    ) -> NDArray:

        std = np.std(y_true - y_predict, axis=1)
        std = np.round(std, 10) * df.MILLISECOND

        return std

    def mse_error(
            self,
            y_true:             NDArray,
            y_predict:          NDArray
    ) -> NDArray:

        mse = np.linalg.norm(y_true - y_predict, axis=1) ** 2 / self.datastore['feature_len']
        mse = np.round(mse, 10) * self.datastore['feature_len']

        return mse

    # @df.timeit
    def _pre_process(
            self,
            ecg_signal:         NDArray,
            ecg_signal_fs:      int,
            max_sample_process: int
    ) -> [NDArray, NDArray]:
        
        p_signal = ecg_signal.copy()
        input_signal = ecg_signal.copy()
        output_signal = ecg_signal.copy()
        try:
            if ecg_signal_fs != self.datastore['sampling_rate']:
                p_signal, _ = resample_sig(
                        p_signal,
                        ecg_signal_fs,
                        self.datastore['sampling_rate']
                )

            if self._clip_range is not None:
                p_signal = np.clip(p_signal, *self._clip_range)

            if self.datastore['bwr']:
                p_signal = ut.bwr_smooth(
                        p_signal,
                        fs=self.datastore['sampling_rate']
                )

            hp_signal = ut.butter_bandpass_filter(
                    p_signal,
                    *self._bandpass_filter,
                    self.datastore['sampling_rate']
            )
            
            norm_signal = ut.norm(
                    hp_signal,
                    window_len=int(self.datastore['NUM_NORM'] * self.datastore['sampling_rate'])
            )

            if self.datastore['highpass'][0]:
                input_signal = hp_signal.copy()

            if self.datastore['highpass'][1]:
                output_signal = hp_signal.copy()

            if self.datastore['norm'][0]:
                input_signal = norm_signal.copy()

            if self.datastore['norm'][1]:
                output_signal = norm_signal.copy()

            redundant = len(norm_signal) % self.datastore['feature_len']
            if redundant:
                input_signal = np.concatenate([input_signal, np.zeros(self.datastore['feature_len'] - redundant)])
                output_signal = np.concatenate([output_signal, np.zeros(self.datastore['feature_len'] - redundant)])

            if len(input_signal) < max_sample_process:
                input_signal = np.concatenate([input_signal, np.zeros(max_sample_process - len(input_signal))])
                output_signal = np.concatenate([output_signal, np.zeros(max_sample_process - len(output_signal))])

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return input_signal, output_signal

    @staticmethod
    def _check_condition(
            values:             NDArray,
            i:                  int,
            threshold:          float
    ) -> bool:

        check = values[i] > threshold
        check = (
                values[i] > threshold * 1.2
                or (check and any(values[i - x] > threshold * 0.8 for x in [-1, 1]))
                or any(np.count_nonzero(values[i - 1: i + 2] > x) > 2 for x in [threshold, threshold * 0.8])
        )

        return check

    def _post_process(
            self,
            frames:             NDArray,
            predict:            NDArray,
    ) -> [NDArray, NDArray, NDArray]:

        noise = np.ones(len(frames), dtype=int) * self.datastore['classes']['SINUS']
        std = np.zeros(len(frames), dtype=int)
        mse = np.zeros(len(frames), dtype=int)

        try:
            std_threshold = self.datastore['ABNORMAL_THR_STD'][1]
            mse_threshold = self.datastore['ABNORMAL_THR_MSE'][1]

            frames = np.squeeze(frames)
            predict = np.squeeze(predict)

            std = self.std_error(frames, predict)
            mse = self.mse_error(frames, predict)

            ind = list(filter(
                lambda i: self._check_condition(std, i, std_threshold) | self._check_condition(mse, i, mse_threshold),
                range(1, len(frames) - 1, 1)
            ))

            if std[0] > std_threshold or mse[0] > std_threshold:
                ind.append(0)

            if std[-1] > std_threshold or mse[-1] > std_threshold:
                ind.append(len(frames) - 1)

            if len(ind) > 0:
                noise[np.array(ind)] = self.datastore['classes']['NOISE']

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return noise, std, mse

    def _review(
            self,
            raw_ecg:            NDArray,
            raw_ecg_fs:         int,
            ref_process_ecg:    NDArray,
            process_ecg:        NDArray,
            frame_index:        NDArray,
            noise:              NDArray,
            std:                NDArray,
            mse:                NDArray
    ) -> None:
        try:
            color = ['r', 'b', 'y', 'g', 'm']
    
            fig, (axis) = plt.subplots(2, 1, sharey='all', sharex='all')
    
            t = np.arange(len(raw_ecg)) / raw_ecg_fs
            axis[0].plot(t, raw_ecg)
    
            mark_noise = noise.copy()
            mark_noise[np.flatnonzero(mark_noise == 1)] = max(raw_ecg)
            axis[0].plot(t, mark_noise, 'r--')
    
            major_ticks = np.arange(0, len(raw_ecg), raw_ecg_fs) / raw_ecg_fs
            axis[0].set_xticks(major_ticks)
            axis[0].set_yticks(np.arange(round(np.min(raw_ecg), 0), round(min([np.max(raw_ecg), 5]), 0) + 1, 1))
    
            axis[0].set_xlim(t[0], t[-1])
    
            t2 = np.arange(len(process_ecg)) / self.datastore['sampling_rate']
            axis[1].plot(t2, process_ecg, label='ecg_process')
            axis[1].plot(t2, ref_process_ecg, label='ecg_ref')
    
            major_ticks = np.arange(0, len(process_ecg), self.datastore['sampling_rate'])
            major_ticks = major_ticks / self.datastore['sampling_rate']
            axis[1].set_xticks(major_ticks)
            axis[1].set_yticks(np.arange(round(np.min(process_ecg), 0), round(min([np.max(process_ecg), 5]), 0) + 1, 1))
    
            axis[1].grid(which='major', color='#CCCCCC', linestyle='--')
            axis[1].grid(which='minor', color='#CCCCCC', linestyle=':')
    
            check_group = np.array(list(map(lambda x: x[[0, -1]] / self.datastore['sampling_rate'], frame_index)))
    
            y_min = np.min(process_ecg)
            y_max = np.max(process_ecg)
            for index, gr in enumerate(check_group):
                axis[1].axvspan(gr[0], gr[-1], y_min, y_max, facecolor=color[int(index % len(color))], alpha=0.05)
                axis[1].annotate(str(np.round(mse[index], 4)), (gr[0], y_max - 0.2))
                axis[1].annotate(str(np.round(std[index], 4)), (gr[0], y_min + 0.2))
    
            axis[1].set_ylim(y_min, y_max)
            axis[0].set_title('NOISE DETECTION')
            axis[1].set_title('STD + MSE')
            axis[1].legend()
            plt.show()
            plt.close()
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def prediction(
            self,
            ecg_signal:         NDArray,
            ecg_signal_fs:      int,
            review:             bool = False
    ) -> NDArray:

        noise = list()
        try:
            if len(ecg_signal) / ecg_signal_fs <= self.datastore['feature_len'] / self.datastore['sampling_rate']:
                return np.array(noise)
            
            start_process_time = time.monotonic()
            process_time = dict()
            total_noise_predict = list()
            
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
            pre_ecg_signal, ref_ecg_signal = self._pre_process(
                ecg_signal=ecg_signal,
                ecg_signal_fs=ecg_signal_fs,
                max_sample_process=max_sample_process
            )

            total_buf_frames = list()
            step = 0
            
            overlap_sample = int(self.datastore['NUM_OVERLAP'] * self.datastore['sampling_rate'])
            overlap = self.datastore['feature_len'] - overlap_sample
            
            while sample_to < len(pre_ecg_signal):
                sample_len = min(max_sample_process, (len(ecg_signal) - sample_from))
                if sample_len <= 0:
                    break

                step += 1
                # region Pre-process
                start = time.monotonic()
                sample_to = sample_from + sample_len

                fs_sample_from = int(sample_from * fs_ratio)
                fs_sample_to = int(sample_to * fs_ratio)

                buf_ecg = pre_ecg_signal[fs_sample_from: fs_sample_to]
                data_index = (np.arange(0, self.datastore['feature_len'], 1)[None, :] +
                              np.arange(0, len(buf_ecg) - overlap_sample, overlap)[:, None])
            
                data_index[data_index > len(buf_ecg) - 1] = len(buf_ecg) - 1

                buf_frames = ref_ecg_signal[fs_sample_from: fs_sample_to][data_index]
                process_data = np.expand_dims(buf_ecg[data_index], axis=-1)
                df.get_update_time_process(process_time, start, key='preProcess')
                # endregion Pre-process

                # region GRPC request tf server
                start = time.monotonic()
                output = self.make_and_send_grpc_request(data=process_data)
                process_time = df.get_update_time_process(process_time, start, key='tfServer')
                # endregion GRPC request tf server

                # region Post-processing
                start = time.monotonic()
                output = np.reshape(output, newshape=data_index.shape)
                if len(output) > 0:
                    if len(total_noise_predict) == 0:
                        total_buf_frames = buf_frames
                        total_noise_predict = output
                    else:
                        total_buf_frames = np.row_stack((total_buf_frames, buf_frames))
                        total_noise_predict = np.row_stack((total_noise_predict, output))
                        
                process_time = df.get_update_time_process(process_time, start, key='postProcess')
                # endregion Post-processing

                if sample_from < (sample_to - overlap_sample):
                    sample_to -= overlap_sample

                sample_from = sample_to

            self.close_channel()

            # region Post-processing
            start = time.monotonic()
            noise_predict, std, mse = self._post_process(
                    frames=total_buf_frames,
                    predict=total_noise_predict
            )
            frame_index = np.arange(0, len(pre_ecg_signal) - overlap_sample, overlap)[:, None]
            frame_index = np.arange(0, self.datastore['feature_len'], 1)[None, :] + frame_index
            frame_index[frame_index > len(pre_ecg_signal) - 1] = len(pre_ecg_signal) - 1

            noise = np.ones(len(pre_ecg_signal), dtype=int)
            if np.all(noise_predict == self.datastore['classes']['SINUS']):
                noise *= self.datastore['classes']['SINUS']
            elif np.all(noise_predict == self.datastore['classes']['NOISE']):
                noise *= self.datastore['classes']['NOISE']
            else:
                for i, ind_frame in enumerate(frame_index):
                    if i >= len(noise_predict):
                        break
                        
                    noise[ind_frame] *= noise_predict[i]
                
            if self.datastore['sampling_rate'] != ecg_signal_fs:
                noise, _ = resample_sig(
                        noise,
                        self.datastore['sampling_rate'],
                        ecg_signal_fs
                )
                noise = np.round(noise).astype(int)
                
            noise = noise[np.flatnonzero(np.arange(len(noise)) < len(ecg_signal))]
            review and self._review(
                raw_ecg=ecg_signal,
                raw_ecg_fs=ecg_signal_fs,
                ref_process_ecg=ref_ecg_signal,
                process_ecg=pre_ecg_signal,
                frame_index=frame_index,
                noise=noise,
                std=std,
                mse=mse
            )

            process_time = df.get_update_time_process(process_time, start, key='postProcess')
            # endregion Post-processing

            self._log_process_time and self.log_time(process_time)

            self.update_performance_time(
                step=step,
                start_process_time=start_process_time,
                process_time=process_time
            )

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return noise
