from btcy_holter import *
from scipy.signal import find_peaks


class PTStructures:
    def __init__(
            self,
            beat_count: int = 1
    ) -> None:
        self.qt:            NDArray = np.zeros(beat_count, dtype=float)
        self.qtc:           NDArray = np.zeros(beat_count, dtype=float)
                            
        self.st_slope:      NDArray = np.zeros(beat_count, dtype=float)
        self.st_level:      NDArray = np.zeros(beat_count, dtype=float)
        
        self.t_peak:        NDArray = np.zeros(beat_count, dtype=int)       # sample
        self.t_amplitude:   NDArray = np.zeros(beat_count, dtype=float)     # mV
        self.t_onset:       NDArray = np.zeros(beat_count, dtype=int)       # sample
        self.t_offset:      NDArray = np.zeros(beat_count, dtype=int)       # sample
        
        self.p_peak:        NDArray = np.zeros(beat_count, dtype=int)       # sample
        self.p_amplitude:   NDArray = np.zeros(beat_count, dtype=float)     # mV
        self.p_onset:       NDArray = np.zeros(beat_count, dtype=int)       # sample
        self.p_offset:      NDArray = np.zeros(beat_count, dtype=int)       # sample
        self.p_peak_av2:    NDArray = np.zeros(beat_count, dtype=int)       # sample
        
        self.qrs_onset:     NDArray = np.zeros(beat_count, dtype=int)
        self.qrs_offset:    NDArray = np.zeros(beat_count, dtype=int)
        
        self.hr:            NDArray = np.zeros(beat_count, dtype=int)


class PQSTDetection:
    LIMIT_OFFSET_SIGNAL:                    Final[float] = 2.5  # second
    
    Rv_P:                                   Final[float] = 0.05
    Rv_T:                                   Final[float] = 0.1
    
    # _check_phase_length, _t_offset, _t_onset, _adjust_t_peak, _p_onset, _p_offset
    THR_PHASE_LEN:                          Final[int] = 40
    FIVE_SAMPLES:                           Final[int] = 5  # samples
    
    # _cal_theta
    DEGREE_180:                             Final[float] = 180
    
    # _get_condition_based_on_rr_ratio
    THR_RR_RATIO_1:                         Final[float] = 1.6
    THR_RR_RATIO_2:                         Final[float] = 0.85
    CONDITION_RR_RATIO:                     Final[dict] = {'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4}
    
    # _find_p_wave_position
    THR_P_PEAK_AMP:                         Final[float] = 0.1
    AFTER_T_PEAK:                           Final[float] = 0.2  # s
    RATIO_BEAT_P_AMP:                       Final[float] = 0.3
    THR_DURATION_P2P_MIN:                   Final[float] = 0.2  # s
    THR_DURATION_P2P_TACHY_MIN:             Final[float] = 0.05  # s
    COUNT_P_PEAKS:                          Final[dict] = {'ONE': 1, 'TWO': 2}
    
    # _detect_qrs_onset_offset
    QRS_ONSET_DURATION:                     Final[float] = 0.06     # s
    QRS_OFFSET_DURATION:                    Final[float] = 0.08     # s
    
    # if symbol is not V, Wide of QRS is less than 160ms
    QRS_ONSET_DURATION_FOR_NARROW_QRS:      Final[float] = 0.08     # s
    
    # _t_peak
    START_T_RATIO_RR:                       Final[float] = 0.10
    STOP_T_RATIO_RR:                        Final[float] = 0.57
    T_START_DURATION_FROM_QRS_OFFSET:       Final[int] = 5          # samples
    T_PEAK_DURATION_FROM_QRS_OFFSET:        Final[float] = 0.1          # s
    
    # _p_peak
    START_P_RATIO_RR:                       Final[float] = 0.3
    STOP_P_RATIO_RR:                        Final[float] = 0.97
    P_STOP_DURATION_FROM_QRS_ONSET:         Final[int] = 5  # samples
    
    # _adjust_qrs_onset_offset
    NUM_DIV:                                Final[int] = 3
    
    # _cal_qt_qtc_hr
    THR_500_SAMPLES:                        Final[int] = 500
    
    # _cal_st
    ST_SEG_POST_DURATION_FROM_QRS_OFFSET:   Final[float] = 0.06         # s
    THR_HR_120:                             Final[int] = 120            # bpm
    ST_SEG_POST_DURATION_FOR_HR_120:        Final[float] = 0.08
    
    # process
    THR_RR_INVALID:                         Final[int] = 40             # samples
    
    SAMPLING_RATE:                          Final[int] = 250
    
    def __init__(
            self,
            ecg_signal:         NDArray,
            smooth_ecg_signal:  NDArray,
            samples:            NDArray | List,
            symbols:            NDArray | List,
            events:             NDArray | List,
            ecg_signal_fs:      int = cf.SAMPLING_RATE,
            is_hes_event:       bool = False
    ):
        try:
            self.is_hes_event = is_hes_event
            self.ecg_signal_fs = ecg_signal_fs
            
            self.events = df.convert_to_array(events)
            self.samples = df.convert_to_array(samples)
            self.symbols = df.convert_to_array(symbols)
            
            self.ecg_signal = deepcopy(ecg_signal)
            self.smooth_ecg_signal = deepcopy(smooth_ecg_signal)
            self._preprocess()
            
            self.pt = PTStructures(beat_count=len(samples))
            
            self.phase_P = np.arctan(self.smooth_ecg_signal / self.Rv_P)
            self.phase_T = np.arctan(self.smooth_ecg_signal / self.Rv_T)
            
            self.rr = np.diff(self.samples)
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
            
    def _preprocess(
            self
    ):
        try:
            if self.ecg_signal_fs != self.SAMPLING_RATE:
                self.smooth_ecg_signal, _ = resample_sig(
                        x=self.smooth_ecg_signal,
                        fs=self.ecg_signal_fs,
                        fs_target=self.SAMPLING_RATE
                )
                
                self.ecg_signal, _ = resample_sig(
                        x=self.ecg_signal,
                        fs=self.ecg_signal_fs,
                        fs_target=self.SAMPLING_RATE
                )
                
                self.samples = df.resample_beat_samples(
                        samples=self.samples,
                        sampling_rate=self.ecg_signal_fs,
                        target_sampling_rate=self.SAMPLING_RATE
                )
            
            to = min(int(self.samples[-1] + int(self.LIMIT_OFFSET_SIGNAL * self.SAMPLING_RATE)), len(self.ecg_signal))
            self.ecg_signal = self.ecg_signal[:to]
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
     
    def _check_phase_length(
            self,
            phase: NDArray
    ) -> NDArray:
        
        try:
            ind = np.flatnonzero(len(phase) < self.THR_PHASE_LEN)
            if len(ind) > 0:
                phase[ind] = np.asarray(list(map(
                    lambda i: np.concatenate((phase[i], np.zeros(self.THR_PHASE_LEN - len(phase[i])))),
                    ind
                )))
                
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return phase
    
    def _cal_theta(
            self,
            p: List,
            i: float,
            k: int
    ) -> NDArray:
        
        val = None
        try:
            
            k_s = k / self.SAMPLING_RATE
            val = np.abs(np.pi - np.arctan((p[:, i] - p[:, i + k]) / k_s))
            val -= np.abs(np.arctan((p[:, i + k] - p[:, i + 2 * k]) / k_s))
            
            val = val * self.DEGREE_180 / np.pi
            val[val > self.DEGREE_180] = 0
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return val
    
    def _get_condition_based_on_rr_ratio(
            self,
            i: int
    ) -> int:
        
        cond = self.CONDITION_RR_RATIO['ONE']
        try:
            if self.symbols[i] == df.HolterSymbols.N.value and i - 2 >= 0:
                rr_val = np.diff(self.samples[i - 2: i + 1])
                if (
                        rr_val[1] > self.THR_RR_RATIO_1 * rr_val[0]
                        or ((rr_val[1] > self.THR_RR_RATIO_2 * rr_val[0])
                            and (i > 1 and any(self.pt.p_peak_av2[i-2:i])))
                ):
                    cond = self.CONDITION_RR_RATIO['THREE']
                    if i - 3 > -1:
                        if (
                                not any(self.pt.p_peak_av2[i - 2: i])
                                and rr_val[0] < self.THR_RR_RATIO_2 * (self.samples[i - 2] - self.samples[i - 3])
                        ):
                            cond = self.CONDITION_RR_RATIO['FOUR']
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return cond
    
    # @df.timeit
    def _find_t_wave_position(
            self,
            start:  int,
            stop:   int
    ) -> int:

        wave_pos = 0
        try:
            phase_t_wave = self.phase_T[start: stop]
            phase_t_wave[phase_t_wave < 0] = 0
            phase_t_wave[phase_t_wave > np.pi / 2] = np.pi / 2
            if len(phase_t_wave) > 0:
                wave_pos = int(np.argmax(phase_t_wave) + start)

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return wave_pos
    
    # @df.timeit
    def _find_p_wave_position(
            self,
            start:      int,
            stop:       int,
            index:      int,
            condition:  int,
    ) -> [int, int]:

        p_wave_1 = 0
        p_wave_2 = 0
        try:
            t_after = self.THR_DURATION_P2P_MIN
            if self.is_hes_event:
                if df.check_hes_event(self.events[index], df.HOLTER_TACHY):
                    t_after = self.THR_DURATION_P2P_TACHY_MIN
            elif index > 0 and self.rr[index - 1] in df.VALID_BEAT_TYPE:
                rr = (self.samples[index] - self.samples[index - 1]) / self.SAMPLING_RATE
                hr = df.SECOND_IN_MINUTE / rr
                if cf.TACHY_THR <= hr <= df.HR_MAX_THR:
                    t_after = self.THR_DURATION_P2P_TACHY_MIN
            
            phase_p_wave = self.phase_P[start: stop]
            peak: NDArray = find_peaks(phase_p_wave)[0]
            peak = peak[np.flatnonzero(phase_p_wave[peak] > self.THR_P_PEAK_AMP)]  # 0.1 * qrs_amp
            beat_phase_p_amp = self.phase_P[self.samples[index]]
            
            # p_peak_1 == None: First P-wave
            # ==> delete all SMALL peak ==> find Second P-wave
            # The lot of peak ==> NOISE
            if len(peak) > 0 and self.symbols[index] == df.HolterSymbols.N.value:
                if (
                        condition == self.CONDITION_RR_RATIO['THREE']
                        and len(peak) > 1
                        and self.symbols[index - 1] == df.HolterSymbols.N.value
                ):
                    peak = peak[peak + start > self.pt.t_peak[index - 1] + t_after * self.SAMPLING_RATE]
                    if len(peak) == 0:
                        return p_wave_1, p_wave_2
                    
                    peak_val = phase_p_wave[peak]
                    peak = peak[peak_val >= self.RATIO_BEAT_P_AMP * beat_phase_p_amp]
                    for i in range(1, len(peak)):
                        if peak[i] - peak[i - 1] < t_after * self.SAMPLING_RATE:
                            peak[i - 1 + np.argmin(phase_p_wave[peak[i - 1: i]])] = 0
                    
                    peak = peak[peak > 0]
                    if len(peak) >= self.COUNT_P_PEAKS['TWO']:
                        phase_magnitude_p_sorted_index = np.argsort(phase_p_wave[peak])
                        
                        # return P-wave 1, P-wave 2
                        _p_wave_1 = peak[phase_magnitude_p_sorted_index[-1]] + start
                        _p_wave_2 = peak[phase_magnitude_p_sorted_index[-2]] + start
                        
                        if _p_wave_1 == _p_wave_2:
                            p_wave_1 = _p_wave_1
                        
                        elif _p_wave_1 > _p_wave_2:
                            p_wave_1 = _p_wave_1
                            p_wave_2 = _p_wave_2
                        
                        else:
                            p_wave_1 = _p_wave_2
                            p_wave_2 = _p_wave_1
                    
                    elif len(peak) == self.COUNT_P_PEAKS['ONE']:
                        p_wave_1 = peak[0] + start
                
                else:
                    # P-wave
                    peak = np.sort(peak)
                    peak = peak[peak + start > self.pt.t_peak[index - 1] + t_after * self.SAMPLING_RATE]
                    if len(peak) >= self.COUNT_P_PEAKS['ONE']:
                        p_wave_1 = peak[np.argmax(phase_p_wave[peak])] + start
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return p_wave_1, p_wave_2
    
    # @df.timeit
    def _detect_qrs_onset_offset(
            self
    ) -> None:
        try:
            # region Cal
            num_onset_area_candidate = int(self.QRS_ONSET_DURATION * self.SAMPLING_RATE)
            num_offset_area_candidate = int(self.QRS_OFFSET_DURATION * self.SAMPLING_RATE)
            
            x_r = np.concatenate((self.samples, [len(self.ecg_signal) - 1]))
            x_l = np.concatenate((np.array([0]), self.samples))
            rr_range = list(starmap(np.arange, zip(x_l + 1, x_r)))
            
            y_r = self.ecg_signal[x_r]
            y_l = self.ecg_signal[x_l]
            y = y_r - y_l
            x = x_r - x_l
            xy = np.multiply(x_r, y_l) - np.multiply(y_r, x_l)
            
            y_rrs = list(map(lambda x1: x1[0] * x1[-1], zip(y, rr_range)))
            rr_len = np.array(list(map(len, rr_range)))
            divs = list(map(lambda r: 2 * np.pi * (rr_range[r] - x_l[r]) / x[r], range(len(rr_range))))
            sqrts = np.sqrt(np.power(x, 2) + np.power(y, 2))
            
            wd = [
                np.multiply(
                    np.divide(
                        np.abs(y_rrs[i] - x[i] * self.ecg_signal[rr_range[i]] + np.repeat(xy[i], rr_len[i])),
                        np.repeat(sqrts[i], rr_len[i])
                    ),
                    np.cos(divs[i])
                )
                for i in range(len(rr_range))
            ]
            
            qs_point = np.array(list(starmap(
                lambda a, b: [
                    a + (len(b) // 2) + np.argmax(b[len(b) // 2:]),
                    a + np.argmax(b[:len(b) // 2])
                ],
                zip(x_l, wd)
            )))
            # endregion Cal

            # region Junctions of Q wave and S wave respectively
            qi = qs_point[:-1, 0]
            si = qs_point[1:-1, 1]
            si = np.concatenate((si, [self.samples[-1] + si[-1] - self.samples[-2]]), axis=0)
            # endregion Junctions of Q wave and S wave respectively

            # region QRS Onset And Offset Detection
            # qrs_onset Detection of XR beats after qrs_q detection
            k = 4
            onset_area = (qi - k).reshape(-1, 1) + np.arange(-num_onset_area_candidate, 0)[None, :]
            onset_y = self.ecg_signal[onset_area]

            offset_area = (si + k).reshape(-1, 1) + np.arange(0, num_offset_area_candidate)[None, :]
            offset_area[offset_area >= len(self.ecg_signal)] = len(self.ecg_signal) - 1
            offset_y = self.ecg_signal[offset_area]
            
            step = num_onset_area_candidate - (k * 2)
            theta_onset = np.array(list(map(
                self._cal_theta,
                list([onset_y]) * step,
                np.arange(step),
                [k] * step
            )))
            
            step = num_offset_area_candidate - (k * 2)
            theta_offset = np.array(list(map(
                self._cal_theta,
                list([offset_y]) * step,
                np.arange(step),
                [k] * step
            )))

            self.pt.qrs_onset = (qi - num_onset_area_candidate) + np.argmax(theta_onset, axis=0)
            self.pt.qrs_offset = si + np.argmax(theta_offset, axis=0)

            self.pt.qrs_onset[self.pt.qrs_onset == 0] = self.samples[self.pt.qrs_onset == 0]
            self.pt.qrs_onset[self.pt.qrs_offset == 0] = self.samples[self.pt.qrs_offset == 0]
            # endregion QRS Onset And Offset Detection

            # region CHECK wide of QRS is 160ms
            # if symbol is not V, Wide of QRS is less than 160ms???
            num_onset_area = int(self.QRS_ONSET_DURATION_FOR_NARROW_QRS * self.SAMPLING_RATE)
            norm_index = np.logical_and(
                    self.symbols == df.HolterSymbols.N.value,
                    (self.samples - self.pt.qrs_onset) > num_onset_area
            )
            self.pt.qrs_onset[norm_index] = self.samples[norm_index] - num_onset_area
            # endregion CHECK wide of QRS is 160ms

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _t_peak(
            self,
            index: int
    ) -> bool:
        check_find_t_peak = True
        try:
            if self.rr[index - 1] > 1.4 * self.rr[index - 2]:
                start_t = self.samples[index - 1] + int(self.START_T_RATIO_RR * self.rr[index - 2])
                stop_t = self.samples[index - 1] + int(self.STOP_T_RATIO_RR * self.rr[index - 2])
            else:
                start_t = self.samples[index - 1] + int(self.START_T_RATIO_RR * self.rr[index - 1])
                stop_t = self.samples[index - 1] + int(self.STOP_T_RATIO_RR * self.rr[index - 1])
            
            if start_t < self.pt.qrs_offset[index - 1]:
                start_t = self.pt.qrs_offset[index - 1] + self.T_START_DURATION_FROM_QRS_OFFSET
            
            t_wave = self._find_t_wave_position(int(start_t), int(stop_t))
            check_find_t_peak = t_wave != 0
            if t_wave == 0:
                self.pt.t_peak[index - 1] = self.pt.qrs_offset[index - 1] + int(
                    self.T_PEAK_DURATION_FROM_QRS_OFFSET * self.SAMPLING_RATE
                    )
            else:
                self.pt.t_peak[index - 1] = t_wave
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return check_find_t_peak
    
    # @df.timeit
    def _p_peak(
            self,
            index: int
    ) -> None:
        try:
            if self.pt.t_peak[index - 1] != 0:
                start_p: int = int(self.samples[index - 1] + int(self.START_P_RATIO_RR * self.rr[index - 1]) + 1)
                stop_p: int = int(self.samples[index - 1] + int(self.STOP_P_RATIO_RR * self.rr[index - 1]) + 1)
                if stop_p > self.pt.qrs_onset[index] > 0:
                    stop_p = int(self.pt.qrs_onset[index] + self.P_STOP_DURATION_FROM_QRS_ONSET)

                condition = self._get_condition_based_on_rr_ratio(index)
                if self.symbols[index] == df.HolterSymbols.N.value:
                    p_peak, p_peak_av2 = self._find_p_wave_position(
                        start=start_p,
                        stop=stop_p,
                        index=index,
                        condition=condition,
                    )
                    self.pt.p_peak[index] = p_peak
                    self.pt.p_peak_av2[index] = p_peak_av2

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _t_offset(
            self
    ) -> None:
        try:
            self.phase_T[self.phase_T < 0] = 0

            phase_t_matrix = np.zeros((len(self.pt.t_peak), self.THR_PHASE_LEN))
            ind_phase = np.flatnonzero(np.array(self.pt.t_peak) != 0)
            if len(ind_phase) > 0:
                for idx, i_t in enumerate(ind_phase):
                    t_p = self.pt.t_peak[i_t]
                    arr = self.phase_T[t_p] - self.phase_T[t_p: t_p + self.THR_PHASE_LEN]
                    if len(arr) <= self.THR_PHASE_LEN:
                        arr = np.concatenate((arr, np.zeros(self.THR_PHASE_LEN - len(arr))))
                    phase_t_matrix[idx] = arr

                phase_t_matrix = self._check_phase_length(phase_t_matrix)

            self.pt.t_offset = np.array(self.pt.t_peak) + np.argmax(phase_t_matrix, axis=1) - 1
            self.pt.t_offset[self.pt.t_offset < 0] = 0

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _t_onset(
            self
    ) -> None:
        try:
            phase_t_matrix = np.zeros((len(self.pt.t_peak), self.THR_PHASE_LEN))
            ind_phase = np.flatnonzero(np.array(self.pt.t_peak) != 0)
            if len(ind_phase) > 0:
                for idx, i_t in enumerate(ind_phase):
                    t_p = self.pt.t_peak[i_t]
                    arr = self.phase_T[t_p - self.THR_PHASE_LEN: t_p][::-1] - t_p
                    if len(arr) <= self.THR_PHASE_LEN:
                        arr = np.concatenate((arr, np.zeros(self.THR_PHASE_LEN - len(arr))))
                    phase_t_matrix[idx] = arr

                phase_t_matrix = self._check_phase_length(phase_t_matrix)

            self.pt.t_onset = np.asarray(self.pt.t_peak) - np.argmin(phase_t_matrix, axis=1) - self.FIVE_SAMPLES
            self.pt.t_onset[self.pt.t_onset < 0] = 0

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _adjust_t_peak(
            self
    ) -> None:
        try:
            self.pt.t_peak[-1]      = (self.pt.qrs_offset[-1] + self.pt.t_peak[-2] - self.pt.qrs_offset[-2])
            self.pt.t_onset[-1]     = self.pt.t_peak[-1] - (self.pt.t_peak[-2] - self.pt.t_onset[-2])
            self.pt.t_offset[-1]    = self.pt.t_peak[-1] + self.pt.t_offset[-2] - self.pt.t_peak[-2]
            self.pt.t_offset += self.FIVE_SAMPLES

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _p_onset(
            self
    ) -> None:
        try:
            self.phase_P[self.phase_P < 0] = 0

            phase_p_matrix = np.zeros((len(self.pt.p_peak), self.THR_PHASE_LEN))
            ind_phase = np.flatnonzero(np.array(self.pt.p_peak) != 0)
            if len(ind_phase) > 0:
                for idx, i_p in enumerate(ind_phase):
                    arr = self.phase_P[self.pt.p_peak[i_p] - self.THR_PHASE_LEN:
                                       self.pt.p_peak[i_p]][::-1] - self.pt.p_peak[i_p]
                    if len(arr) <= self.THR_PHASE_LEN:
                        arr = np.concatenate((arr, np.zeros(self.THR_PHASE_LEN - len(arr))))
                    phase_p_matrix[idx] = arr

                phase_p_matrix = self._check_phase_length(phase_p_matrix)

            self.pt.p_onset = np.asarray(self.pt.p_peak) - np.argmin(phase_p_matrix, axis=1) - 1
            self.pt.p_onset[self.pt.p_onset > 0] -= self.FIVE_SAMPLES
            self.pt.p_onset[self.pt.p_onset < 0] = 0

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _p_offset(
            self
    ) -> None:
        try:
            phase_p_matrix = np.zeros((len(self.pt.p_peak), self.THR_PHASE_LEN))
            ind_phase = np.flatnonzero(np.array(self.pt.p_peak) != 0)
            if len(ind_phase) > 0:
                for idx, i_p in enumerate(ind_phase):
                    arr = (self.phase_P[self.pt.p_peak[i_p]] -
                           self.phase_P[self.pt.p_peak[i_p]: self.pt.p_peak[i_p] + self.THR_PHASE_LEN])
                    if len(arr) <= self.THR_PHASE_LEN:
                        arr = np.concatenate((arr, np.zeros(self.THR_PHASE_LEN - len(arr))))
                    phase_p_matrix[idx] = arr

                phase_p_matrix = self._check_phase_length(phase_p_matrix)

            self.pt.p_offset = self.pt.p_peak + np.argmax(phase_p_matrix, axis=1) - 1
            self.pt.p_offset[self.pt.p_offset > 0] += self.FIVE_SAMPLES
            self.pt.p_offset[self.pt.p_offset < 0] = 0

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _detect_p_t_wave(
            self
    ) -> None:
        try:
            for i in range(1, len(self.samples)):
                self._t_peak(i) and self._p_peak(i)

            self._t_offset()
            self._t_onset()

            self._adjust_t_peak()

            self._p_onset()
            self._p_offset()

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _adjust_qrs_onset_offset(
            self
    ) -> None:
        try:
            for i in range(len(self.samples)):
                if 0 < self.pt.qrs_onset[i] < self.pt.p_offset[i] and self.pt.p_offset[i] > 0:
                    tmp = int((self.samples[i] - self.pt.p_offset[i]) / self.NUM_DIV)
                    self.pt.qrs_onset[i] = self.pt.p_offset[i] + tmp

                if self.pt.qrs_offset[i] > 0 and 0 < self.pt.t_onset[i] < self.pt.qrs_offset[i]:
                    tmp = int((self.pt.t_onset[i] - self.samples[i]) / self.NUM_DIV)
                    self.pt.qrs_offset[i] = self.pt.t_onset[i] - tmp

            self.pt.t_peak[self.pt.t_peak < 0] = 0
            self.pt.t_peak[self.pt.t_peak >= len(self.ecg_signal)] = len(self.ecg_signal) - 1

            self.pt.qrs_onset[self.pt.qrs_onset < 0] = 0
            self.pt.qrs_onset[self.pt.qrs_onset >= len(self.ecg_signal)] = len(self.ecg_signal) - 1

            self.pt.qrs_offset[self.pt.qrs_offset < 0] = 0
            self.pt.qrs_offset[self.pt.qrs_offset >= len(self.ecg_signal)] = len(self.ecg_signal) - 1

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _cal_qt_qtc_hr(
            self
    ) -> None:
        try:
            self.pt.qt = (self.pt.t_offset - self.pt.qrs_onset)
            self.pt.qt[(self.pt.qt < 0) + (self.pt.qt > self.THR_500_SAMPLES)] = 0
            self.pt.qt = np.divide(self.pt.qt, self.SAMPLING_RATE)
            
            self.pt.qt[self.pt.qt < 0] = 0
            rr_sample = np.diff(np.concatenate((self.samples, [len(self.ecg_signal) - 1])))
            rr_sample[rr_sample > self.THR_500_SAMPLES] = self.THR_500_SAMPLES  # 30 pbm = 2s = 500 sample
            
            rr_interval = np.divide(rr_sample, self.SAMPLING_RATE)
            self.pt.qtc = np.around(np.divide(self.pt.qt, np.sqrt(rr_interval)), decimals=4)
            
            self.pt.hr = (60 / rr_interval).astype(int)
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _cal_st(
            self
    ) -> None:
        try:
            st_seg_post = self.pt.qrs_offset + int(self.ST_SEG_POST_DURATION_FROM_QRS_OFFSET * self.SAMPLING_RATE)
            st_seg_post[self.pt.hr < self.THR_HR_120] = self.pt.qrs_offset[self.pt.hr < self.THR_HR_120] + int(
                self.ST_SEG_POST_DURATION_FOR_HR_120 * self.SAMPLING_RATE
            )
            st_segment = st_seg_post - self.pt.qrs_offset
            st_seg_post[st_seg_post >= len(self.ecg_signal)] = len(self.ecg_signal) - 1

            self.pt.st_slope = np.around(
                    np.divide((self.ecg_signal[st_seg_post] - self.ecg_signal[self.pt.qrs_offset]), st_segment),
                    decimals=4
            )
            
            self.pt.st_level = np.zeros(len(self.pt.p_onset), dtype=np.float32)
            ind = np.flatnonzero(self.pt.p_onset > 0)
            for i in ind:
                begin_idx = self.pt.p_onset[i]
                end_idx = min(self.pt.t_offset[i], len(self.ecg_signal))
                self.pt.st_level[i] = self.ecg_signal[begin_idx] - np.average(self.ecg_signal[begin_idx: end_idx])
            self.pt.st_level = np.nan_to_num(np.around(self.pt.st_level, decimals=4))
            pass
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _revert_beat_invalid(
            self,
            data_invalid:   Dict,
            check_index:    NDArray
    ) -> None:
        try:
            if len(check_index) > 0:
                samples = np.concatenate((self.samples, data_invalid['peak']))
                index_sort = np.argsort(samples)
                
                offset = np.zeros(len(check_index), dtype=int)
                self.pt.qrs_onset = np.concatenate((self.pt.qrs_onset, offset))[index_sort]
                self.pt.qrs_offset = np.concatenate((self.pt.qrs_offset, offset))[index_sort]
                
                self.pt.p_peak = np.concatenate((self.pt.p_peak, offset))[index_sort]
                self.pt.p_onset = np.concatenate((self.pt.p_onset, offset))[index_sort]
                self.pt.p_offset = np.concatenate((self.pt.p_offset, offset))[index_sort]
                self.pt.p_peak_av2 = np.concatenate((self.pt.p_peak_av2, offset))[index_sort]
                
                self.pt.t_peak = np.concatenate((self.pt.t_peak, offset))[index_sort]
                self.pt.t_onset = np.concatenate((self.pt.t_onset, offset))[index_sort]
                self.pt.t_offset = np.concatenate((self.pt.t_offset, offset))[index_sort]
                
                self.pt.hr = np.concatenate((self.pt.hr, offset))[index_sort]
                
                offset = np.zeros(len(check_index), dtype=float)
                self.pt.qt = np.concatenate((self.pt.qt, offset))[index_sort]
                self.pt.qtc = np.concatenate((self.pt.qtc, offset))[index_sort]
                
                self.pt.st_slope = np.concatenate((self.pt.st_slope, offset))[index_sort]
                self.pt.st_level = np.concatenate((self.pt.st_level, offset))[index_sort]
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _post_process(
            self
    ) -> None:
        try:
            
            def resample(x):
                y = df.resample_beat_samples(
                        x,
                        sampling_rate=self.SAMPLING_RATE,
                        target_sampling_rate=self.ecg_signal_fs
                )
                y[y > len(self.ecg_signal)] = len(self.ecg_signal) - 1
                
                return y
            
            self.pt.qrs_onset   = resample(self.pt.qrs_onset)
            self.pt.qrs_offset  = resample(self.pt.qrs_offset)
            
            self.pt.p_peak      = resample(self.pt.p_peak)
            self.pt.p_onset     = resample(self.pt.p_onset)
            self.pt.p_offset    = resample(self.pt.p_offset)
            self.pt.p_peak_av2  = resample(self.pt.p_peak_av2)
            
            self.pt.t_peak      = resample(self.pt.t_peak)
            self.pt.t_onset     = resample(self.pt.t_onset)
            self.pt.t_offset    = resample(self.pt.t_offset)
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def process(
            self,
    ) -> PTStructures:

        try:
            if len(self.samples) == 0:
                return self.pt

            data_invalid = dict()
            check_index = np.flatnonzero(np.diff(self.samples) <= self.THR_RR_INVALID)
            if len(check_index) > 0:
                data_invalid['peak'] = self.samples[check_index + 1]
                self.samples = np.delete(self.samples, check_index + 1)
                self.events = np.delete(self.events, check_index + 1)
                self.symbols = np.delete(self.symbols, check_index + 1)
                self.pt = PTStructures(beat_count=len(self.samples))
            
            self._detect_qrs_onset_offset()
            self._detect_p_t_wave()
            self._adjust_qrs_onset_offset()
            self._cal_qt_qtc_hr()
            self._cal_st()
            self._revert_beat_invalid(data_invalid, check_index)
            self._post_process()
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return self.pt


class PQST(
        pt.Algorithm
):
    MAX_BEAT_COUNT_IN_VALID_REGION: Final[int] = 100
    MIN_BEAT_COUNT_IN_VALID_REGION: Final[int] = 60
    
    LIMIT_PT_WAVE_SAMPLE:       Final[int] = 40  # sample
    
    DEFINE_ARRAY_COLUMNS:       Final[List] = [
        'qt', 'qtc', 'st_level', 'st_slope',
        'p_onset', 'p_peak', 'p_offset', 'p_amplitude',
        't_onset', 't_peak', 't_offset', 't_amplitude',
        'qrs_onset', 'qrs_offset'
    ]
    
    def __init__(
            self,
            data_structure:         sr.AIPredictionResult,
            hes_process:            bool = False,
            return_data_format:     str = 'array',
    ) -> None:
        try:
            super(PQST, self).__init__(
                    data_structure=data_structure,
                    is_hes_process=hes_process,
            )
            self.data_structure.symbol = self._convert_ivcd_to_normal(self.data_structure.symbol)
            if not self.validate_input_is_symbol(self.data_structure.symbol):
                self.data_structure.symbol = df.convert_hes_beat_to_symbol(self.data_structure.symbol)
                
            self._return_data_format:   Final[str] = return_data_format

            self._ecg_fil = None
            self._ecg_bwr = None
            
            self._preprocess()
            self.mark = self._mark_invalid_region()
            
            self._pt_array = None
            self._pt_attr = None
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
            
    def _review(
            self
    ) -> None:
        try:
            fig, (axis) = plt.subplots(3, 1, sharex='all', sharey='all', figsize=(19.2, 10.08))
            fig.subplots_adjust(hspace=0.25, wspace=0.2, left=0.02, right=0.99, bottom=0.03, top=0.95)
            
            axis_count = 0
            axis[axis_count].set_title(f'QRS Onset/Offset')
            axis[axis_count].plot(self.data_structure.ecg_signal)
            
            qrs_onset = self._pt_array.qrs_onset[np.flatnonzero(self._pt_array.qrs_onset > 0)]
            qrs_offset = self._pt_array.qrs_offset[np.flatnonzero(self._pt_array.qrs_offset > 0)]
            
            axis[axis_count].plot(qrs_onset, self.data_structure.ecg_signal[qrs_onset], 'b^', label='qrs onset')
            axis[axis_count].plot(qrs_offset, self.data_structure.ecg_signal[qrs_offset], 'k^', label='qrs offset')
            axis[axis_count].legend(loc='upper right')
            axis_count += 1
            
            max_t_peak = np.max(self._pt_array.t_peak)
            min_t_peak = np.min(self._pt_array.t_peak)
            axis[axis_count].set_title(f'T PEAK - maxAmp: {max_t_peak} | minAmp: {min_t_peak}')
            axis[axis_count].plot(self.data_structure.ecg_signal)
            axis[axis_count].plot(
                    self._pt_array.t_peak,
                    self.data_structure.ecg_signal[self._pt_array.t_peak],
                    'r*',
                    label='t peak'
            )
            axis[axis_count].plot(
                    self._pt_array.t_onset,
                    self.data_structure.ecg_signal[self._pt_array.t_onset],
                    'b^',
                    label='t onset'
            )
            axis[axis_count].plot(
                    self._pt_array.t_offset,
                    self.data_structure.ecg_signal[self._pt_array.t_offset],
                    'k^',
                    label='t offset'
            )
            axis[axis_count].legend(loc='upper right')
            axis_count += 1
            
            axis[axis_count].set_title(f'P ONSET / OFFSET')
            axis[axis_count].plot(self.data_structure.ecg_signal)
            axis[axis_count].plot(
                    self._pt_array.p_peak,
                    self.data_structure.ecg_signal[self._pt_array.p_peak],
                    'r*',
                    label='p peak'
            )
            axis[axis_count].plot(
                    self._pt_array.p_onset,
                    self.data_structure.ecg_signal[self._pt_array.p_onset],
                    'b^',
                    label='p onset'
            )
            axis[axis_count].plot(
                    self._pt_array.p_offset,
                    self.data_structure.ecg_signal[self._pt_array.p_offset],
                    'k^',
                    label='p offset'
            )
            axis[axis_count].legend(loc='upper right')
            
            axis[axis_count].set_xlim(0, len(self.data_structure.ecg_signal))
            
            plt.show()
            plt.close()
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            
    def _preprocess(
            self
    ) -> None:
        try:
            self._ecg_fil = ut.butter_highpass_filter(
                    signal=self.data_structure.ecg_signal,
                    cutoff=0.5,
                    fs=self.data_structure.sampling_rate
            )
            
            self._ecg_bwr = ut.smooth(
                    self._ecg_fil,
                    window_len=int(0.04 * self.data_structure.sampling_rate)
            )
            
            self._ecg_bwr = ut.bwr_smooth(
                    self._ecg_bwr,
                    fs=self.data_structure.sampling_rate
            )
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    def _mark_invalid_region(
            self,
    ) -> NDArray:
        
        region = np.ones_like(self.data_structure.beat) * self.VALID
        try:
            if len(region) == 0:
                return region
            
            if self.is_hes_process:
                invalid_region = np.logical_or(
                        df.check_hes_event(self.data_structure.rhythm, df.HOLTER_ARTIFACT),
                        df.check_hes_event(self.data_structure.rhythm, df.HOLTER_AFIB),
                )
            else:
                invalid_region = np.in1d(
                        self.data_structure.rhythm,
                        [cf.RHYTHMS_DATASTORE['classes']['OTHER'], cf.RHYTHMS_DATASTORE['classes']['AFIB']]
                )
            
            invalid_beat = self.data_structure.symbol != df.HolterSymbols.N.value
            
            invalid_sample = np.logical_or(
                    self.data_structure.beat < self.LIMIT_PT_WAVE_SAMPLE,
                    self.data_structure.beat > len(self.data_structure.ecg_signal) - self.LIMIT_PT_WAVE_SAMPLE
            )
            
            region[invalid_region | invalid_beat | invalid_sample] = self.INVALID
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return region
    
    # @df.timeit
    def _process_valid_region(
            self,
            index: NDArray
    ) -> None:
        try:
            pt_structure = PQSTDetection(
                    ecg_signal=self._ecg_fil,
                    smooth_ecg_signal=self._ecg_bwr,
                    samples=self.data_structure.beat[index],
                    events=self.data_structure.rhythm[index],
                    symbols=self.data_structure.symbol[index],
                    ecg_signal_fs=self.data_structure.sampling_rate,
                    is_hes_event=self.is_hes_process
            ).process()
            for attr in self._pt_attr:
                if 'amplitude' in attr:
                    continue
                    
                tmp = getattr(pt_structure, attr)
                if len(tmp) == len(index):
                    getattr(self._pt_array, attr)[index] = tmp
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _process_region(
            self
    ) -> None:
        try:
            self._pt_array = PTStructures(beat_count=len(self.data_structure.beat))
            
            self._pt_attr = list(filter(
                lambda x: not callable(getattr(self._pt_array, x)) and not x.startswith('__'),
                dir(self._pt_array)
            ))
            
            group_index = df.get_group_from_index_event(np.flatnonzero(self.mark != self.INVALID))
            if len(group_index) == 0:
                return
            
            threshold = self.MAX_BEAT_COUNT_IN_VALID_REGION
            max_len = max(list(map(len, group_index)))
            if max_len < threshold:
                if max_len > self.MIN_BEAT_COUNT_IN_VALID_REGION:
                    threshold = self.MIN_BEAT_COUNT_IN_VALID_REGION
                else:
                    return
                
            group_index = list(filter(lambda x: len(x) > threshold, group_index))
            if len(group_index) == 0:
                return
            
            for i, index in enumerate(group_index):
                self._process_valid_region(index)
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _packaging_results(
            self
    ) -> None:
        
        try:
            for attr in self._pt_attr:
                if attr in ['hr']:
                    continue
                    
                if 'amplitude' in attr:
                    continue
                    
                tmp = getattr(self._pt_array, attr)
                match attr:
                    case 'qt' | 'qtc' | 'st_slope' | 'st_level':
                        tmp = np.multiply(tmp, df.MILLISECOND)
                        
                    case _:
                        pass
                
                getattr(self._pt_array, attr)[:] = np.array(tmp, dtype=int)
            
            ind = np.flatnonzero(self._pt_array.p_peak > len(self.data_structure.ecg_signal) - 1)
            self._pt_array.p_peak[ind] = len(self.data_structure.ecg_signal) - 1
            
            ind = np.flatnonzero(self._pt_array.p_offset > len(self.data_structure.ecg_signal) - 1)
            self._pt_array.p_offset[ind] = len(self.data_structure.ecg_signal) - 1
            
            ind = np.flatnonzero(self._pt_array.p_onset > len(self.data_structure.ecg_signal) - 1)
            self._pt_array.p_onset[ind] = len(self.data_structure.ecg_signal) - 1
            
            self._pt_array.p_amplitude = np.multiply(
                    self.data_structure.ecg_signal[self._pt_array.p_peak],
                    df.VOLT_TO_MV
            )
            
            ind = np.flatnonzero(self._pt_array.t_peak > len(self.data_structure.ecg_signal) - 1)
            self._pt_array.t_peak[ind] = len(self.data_structure.ecg_signal) - 1
            
            ind = np.flatnonzero(self._pt_array.t_offset > len(self.data_structure.ecg_signal) - 1)
            self._pt_array.t_offset[ind] = len(self.data_structure.ecg_signal) - 1
            
            ind = np.flatnonzero(self._pt_array.t_onset > len(self.data_structure.ecg_signal) - 1)
            self._pt_array.t_onset[ind] = len(self.data_structure.ecg_signal) - 1
            
            self._pt_array.t_amplitude = np.multiply(
                    self.data_structure.ecg_signal[self._pt_array.t_peak],
                    df.VOLT_TO_MV
            )
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
    def _format_output(
            self
    ) -> pl.DataFrame | List | PTStructures:
        
        result = None
        try:
            match self._return_data_format:
                case 'pt_structure':
                    result = self._pt_array
                
                case 'dataframe':
                    result = pl.DataFrame(
                            {
                                'qt'         : self._pt_array.qt.astype(int),
                                'qtc'        : self._pt_array.qtc.astype(int),
                                'st_level'   : self._pt_array.st_level.astype(int),
                                'st_slope'   : self._pt_array.st_slope.astype(int),
                                'p_onset'    : self._pt_array.p_onset.astype(int),
                                'p_peak'     : self._pt_array.p_peak.astype(int),
                                'p_offset'   : self._pt_array.p_offset.astype(int),
                                'p_amplitude': self._pt_array.p_amplitude.astype(int),
                                't_onset'    : self._pt_array.t_onset.astype(int),
                                't_peak'     : self._pt_array.t_peak.astype(int),
                                't_offset'   : self._pt_array.t_offset.astype(int),
                                't_amplitude': self._pt_array.t_amplitude.astype(int),
                                'qrs_onset'  : self._pt_array.qrs_onset.astype(int),
                                'qrs_offset' : self._pt_array.qrs_offset.astype(int)
                            }
                    ).select(self.DEFINE_ARRAY_COLUMNS)
                
                case 'dict':
                    result = {
                        'qt'         : self._pt_array.qt.astype(int),
                        'qtc'        : self._pt_array.qtc.astype(int),
                        'st_level'   : self._pt_array.st_level.astype(int),
                        'st_slope'   : self._pt_array.st_slope.astype(int),
                        'p_onset'    : self._pt_array.p_onset.astype(int),
                        'p_peak'     : self._pt_array.p_peak.astype(int),
                        'p_offset'   : self._pt_array.p_offset.astype(int),
                        'p_amplitude': self._pt_array.p_amplitude.astype(int),
                        't_onset'    : self._pt_array.t_onset.astype(int),
                        't_peak'     : self._pt_array.t_peak.astype(int),
                        't_offset'   : self._pt_array.t_offset.astype(int),
                        't_amplitude': self._pt_array.t_amplitude.astype(int),
                        'qrs_onset'  : self._pt_array.qrs_onset.astype(int),
                        'qrs_offset' : self._pt_array.qrs_offset.astype(int)
                    }
                
                case 'array':
                    result = np.zeros([len(self.data_structure.beat), len(self.DEFINE_ARRAY_COLUMNS)], dtype=int)
                    for i, attr in enumerate(self.DEFINE_ARRAY_COLUMNS):
                        try:
                            result[:, i] = getattr(self._pt_array, attr).astype(int)
                            
                        except (Exception,) as err:
                            st.write_error_log(f'{attr} - {err}', class_name=self.__class__.__name__)
                    pass
            
        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return result
        
    # @df.timeit
    def process(
            self,
            review: bool = False
    ) -> pl.DataFrame | List | PTStructures | NDArray:
        
        result = list()
        try:
            if (
                    self.data_structure.ecg_signal is None
                    or len(self.data_structure.beat) == 0
            ):
                return result
            
            self._process_region()
            self._packaging_results()
            result = self._format_output()
            review and self._review()
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return result
