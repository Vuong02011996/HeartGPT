from btcy_holter import *


class LeadOffDetection:
    LIMIT_SAMPLES: Final[int] = 30      # Samples
    SAMPLING_RATE: Final[int] = 250     # Hz
    
    def __init__(
            self
    ):
        pass
    
    def pre_process(
            self,
            ecg_signal:     NDArray,
            ecg_signal_fs:  int
    ) -> Any:
        padding_size = 0
        try:
            if ecg_signal_fs != self.SAMPLING_RATE:
                ecg_signal, _ = resample_sig(
                        x=ecg_signal,
                        fs=ecg_signal_fs,
                        fs_target=self.SAMPLING_RATE
                )
            
            buffer = int(self.SAMPLING_RATE * self.LIMIT_SAMPLES)
            if len(ecg_signal) % buffer != 0:
                padding_size = buffer - len(ecg_signal) % buffer
                ecg_signal = np.concatenate((ecg_signal, np.zeros(padding_size)))
                
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            
        return ecg_signal, padding_size
    
    @staticmethod
    def rec_plot(
            s:      NDArray,
            eps:    float = 0.15,
            steps:  int = None,
            deri:   bool = True
    ):
        if np.max(s) < 2.5 and deri:
            s *= 2.5 / np.max(s)

        if eps is None:
            eps = 0.01

        if steps is None:
            steps = 10

        arr = np.repeat(s[None, :], s.size, axis=0)
        z = np.floor(np.abs(arr - arr.T) / eps)
        z[z > steps] = steps
        z = np.abs((1 * (z < steps / 2)) - 1)

        return z

    @staticmethod
    def clustering(
            line,
            w
    ):
        positions = np.flatnonzero(line == 1)
        border = []
        width = []
        if len(positions) > 0:
            groups_temp = [positions[0]]
            for index in range(1, len(positions)):
                if abs(positions[index] - groups_temp[-1]) > w:
                    if len(groups_temp) >= (w + 1) // 2:
                        border.append(groups_temp[0])
                        width.append(len(groups_temp))
                    groups_temp.clear()

                groups_temp.append(positions[index])
            if len(groups_temp) >= (w + 1) // 2:
                border.append(groups_temp[0])
                width.append(len(groups_temp))

            border = np.asarray(border)
            width = np.asarray(width)

        return border, width

    def removes_horizontal_lines(
            self,
            start_check,
            rec_plot,
            dev_rp,
            w=3
    ):
        res = rec_plot.copy()
        horizontal_dev_rp = []
        try:
            horizontal_dev_rp = dev_rp.copy()
            next_col = start_check
            line = horizontal_dev_rp[:, next_col]
            border, width = self.clustering(line, w)

            shape_hor = horizontal_dev_rp.shape[0]
            while (len(border) == 0 or np.sum(width) > 0.5 * len(line)) and \
                    next_col + horizontal_dev_rp.shape[0] // 10 < shape_hor:
                next_col += horizontal_dev_rp.shape[0] // 10
                line = horizontal_dev_rp[:, next_col]

                border, width = self.clustering(line, w)

            horizontal_line_index = np.zeros_like(line)

            for i in range(len(border)):
                if width[i] < 0.1 * len(line):
                    horizontal_line_index[border[i]: border[i] + width[i]] |= 1

            if np.sum(horizontal_line_index) < 0.5 * len(horizontal_line_index):
                horizontal_line_index = np.asarray(horizontal_line_index, dtype=bool)
                res[horizontal_line_index, :] = 0

            horizontal_dev_rp[horizontal_line_index, :] = 0

        except Exception as err:
            st.get_error_exception(err, class_name=self.__class__.__name__)

        return res, horizontal_dev_rp

    def removes_vertical_lines(
            self,
            start_check,
            rec_plot,
            dev_rp,
            w=3
    ):
        res = rec_plot.copy()
        vertical_dev_rp = []
        try:
            vertical_dev_rp = dev_rp.copy()
            next_row = start_check
            line = vertical_dev_rp[next_row, :]
            border, width = self.clustering(line, w)

            ver_shape = vertical_dev_rp.shape[0]
            while (len(border) == 0 or np.sum(width) > 0.5 * len(line)) and \
                    next_row + vertical_dev_rp.shape[0] // 10 < ver_shape:
                next_row += vertical_dev_rp.shape[0] // 10
                line = vertical_dev_rp[next_row, :]

                border, width = self.clustering(line, w)

            vertical_line_index = np.zeros_like(line)

            for i in range(len(border)):
                if width[i] < 0.1 * len(line):
                    vertical_line_index[border[i]: border[i] + width[i]] |= 1

            if np.sum(vertical_line_index) < 0.5 * len(vertical_line_index):
                vertical_line_index = np.asarray(vertical_line_index, dtype=bool)
                res[:, vertical_line_index] = 0

            vertical_dev_rp[:, vertical_line_index] = 0

        except Exception as err:
            st.get_error_exception(err, class_name=self.__class__.__name__)

        return res, vertical_dev_rp

    def process(
            self,
            ecg_signal:             NDArray,
            ecg_signal_fs:          int,
            len_segment:            int = 10,
            eps_der:                float = 0.04,
            eps_sig:                float = 0.075,
            w:                      int = 3,
            only_last_segment:      bool = True,
    ):

        return_list = []
        try:
            buf_ecg, padding_size = self.pre_process(
                    ecg_signal=ecg_signal,
                    ecg_signal_fs=ecg_signal_fs
            )
            
            downsampling_rate = self.SAMPLING_RATE // 4
            buf_ecg_downsampling_rate, _ = resample_sig(buf_ecg, self.SAMPLING_RATE, downsampling_rate)
            buf_ecg_downsampling_rate = ut.bwr_smooth(buf_ecg_downsampling_rate, downsampling_rate)

            sig_len_downsampling_rate = len(buf_ecg_downsampling_rate)
            num_feature = int(len_segment * downsampling_rate)

            if only_last_segment:
                if padding_size != 0:
                    padding_size_down_rate = int(padding_size * downsampling_rate / self.SAMPLING_RATE)
                    len_buf = len(buf_ecg_downsampling_rate)
                    if padding_size_down_rate + num_feature > len_buf:
                        padding_size_down_rate = len_buf - num_feature

                    f_n = buf_ecg_downsampling_rate[:-padding_size_down_rate]
                    frame_noise_detection = [f_n[-num_feature:]]
                else:
                    frame_noise_detection = [buf_ecg_downsampling_rate[-num_feature:]]
                frame_index = [frame_noise_detection]
            else:
                range_ind = np.arange(0, sig_len_downsampling_rate, num_feature)
                frame_index = np.arange(num_feature)[None, :] + range_ind[:, None]
                frame_noise_detection = buf_ecg_downsampling_rate[frame_index]

            for i, org in enumerate(frame_noise_detection):
                dev = np.gradient(org)
                rp_dev = self.rec_plot(dev, eps=eps_der, deri=False)
                rp_org = self.rec_plot(org, eps=eps_sig, deri=False)

                start_check = []
                if len(start_check) > 1:
                    start_check = start_check[0] + (start_check[1] - start_check[0]) // 2
                    start_check -= frame_index[i][0]
                else:
                    start_check = rp_org.shape[0] // 4

                rp_remain, v = self.removes_vertical_lines(start_check, rp_org, rp_dev, w)
                rp_remain, h = self.removes_horizontal_lines(start_check, rp_remain, rp_dev, w)

                mna_cr = 100 * (np.count_nonzero(rp_remain) / (rp_remain.shape[0] * rp_remain.shape[1]))
                mna_cr_rp_dev = 100 * (np.count_nonzero(rp_dev) / (rp_dev.shape[0] * rp_dev.shape[1]))
                mna_cr_rp_org = 100 * (np.count_nonzero(rp_org) / (rp_org.shape[0] * rp_org.shape[1]))

                if mna_cr < 1 and mna_cr_rp_dev < 1 and mna_cr_rp_org < 1:
                    return_list.append(0)
                else:
                    return_list.append(1)

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return return_list
