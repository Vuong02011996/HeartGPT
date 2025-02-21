from btcy_holter import *
from btcy_holter.ai_core.core.beat_classification_by_IES import BeatClassificationByIES


class BeatDetectionClassificationPantompkins:
    """
       This is the Code for the Pan-Tompkins++ algorithm that is an improved Pan-Tompkins algorithm
       (https://github.com/Fabrizio1994/ECGClassification/blob/master/rpeakdetection/pan_tompkins/pan.py)
    """
    
    QRS_OFFSET_SAMPLES:         Final[float] = 0.360                # second
    ST_SLOPE_OFFSET_SAMPLES:    Final[float] = 0.070                # second
    
    QRS_REGION:                 Final[float] = 1.4                  # second
    NRR:                        Final[int] = 9
    BEAT_OFFSET:                Final[int] = 3
    
    MOVING_WINDOW:              Final[float] = 0.150                # second
    SMOOTHING_SIZE:             Final[int] = 0.06                 # second
    
    FREQUENCY_FILTER:           Final[List] = np.array([5, 18])     # Hz
    FREQUENCY_FILTER_200Hz:     Final[List] = np.array([5, 12])     # Hz
    
    VECTOR:                     Final[NDArray] = np.array([1, 2, 0, -2, -1])
    
    def __init__(
            self,
            ecg_signal:     NDArray,
            ecg_signal_fs:  int,
    ):
        try:
            self.ecg_signal = ecg_signal
            self.ecg_signal_fs = ecg_signal_fs
        
            self.wins = round(self.MOVING_WINDOW * self.ecg_signal_fs)
            self.qrs_offset_samples = round(self.QRS_OFFSET_SAMPLES * self.ecg_signal_fs)
            self.st_slope_samples = round(self.ST_SLOPE_OFFSET_SAMPLES * self.ecg_signal_fs)
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    def detect(
            self,
            ecg_signal:     NDArray | List,
            lookahead:      int = 200,
            delta:          int = 0
    ) -> NDArray:
        
        peaks = list()
        try:
            mx = -np.inf
            mn = np.inf
            for i, amp in enumerate(ecg_signal[:-lookahead]):
                if amp > mx:
                    mx = amp
                
                if amp < mn:
                    mn = amp
                    
                if amp < mx - delta and mx != np.inf and np.max(ecg_signal[i: i + lookahead]) < mx:
                    peaks.append(i)
                    mx = np.inf
                    mn = np.inf
                    if i + lookahead >= len(ecg_signal):
                        break
                        
                if amp > mn + delta and mn != -np.inf and np.min(ecg_signal[i:i + lookahead]) > mn:
                    mn = -np.inf
                    mx = -np.inf
                    if i + lookahead >= len(ecg_signal):
                        break
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return np.array(peaks)
    
    def smoother(
            self,
            ecg_signal: NDArray,
            size:       int = 10
    ) -> NDArray:
        
        ecg_smoothed = np.array([])
        try:
            size = max(1, min(size, len(ecg_signal) - 1))
            win: NDArray = scipy.signal.windows.flattop(size)
            ecg: NDArray = np.concatenate(
                    (
                        ecg_signal[0] * np.ones(size),
                        ecg_signal,
                        ecg_signal[-1] * np.ones(size)
                    )
            )
            ecg_smoothed = np.convolve(win / win.sum(), ecg, mode='same')[size:-size]
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return ecg_smoothed
    
    def pre_process(
            self
    ) -> [NDArray, NDArray]:
        
        ecg = self.ecg_signal.copy()
        ecg_fil = self.ecg_signal.copy()
        try:
            
            if self.ecg_signal_fs == cf.HES_SAMPLING_RATE:
                ecg_fil = ut.butter_lowpass_filter(
                        signal=ecg_fil - np.mean(ecg_fil),
                        cutoff=self.FREQUENCY_FILTER_200Hz[-1],
                        fs=self.ecg_signal_fs
                )
                ecg_fil = ecg_fil / np.max(np.abs(ecg_fil))
                
                ecg_fil = ut.butter_highpass_filter(
                        signal=ecg_fil,
                        cutoff=self.FREQUENCY_FILTER_200Hz[0],
                        fs=self.ecg_signal_fs
                )
                
            else:
                ecg_fil = ut.butter_bandpass_filter(
                        ecg_fil,
                        *self.FREQUENCY_FILTER,
                        fs=self.ecg_signal_fs
                )
                
            ecg_fil = ecg_fil / np.max(np.abs(ecg_fil))
            b2 = self.VECTOR * self.ecg_signal_fs / 8
            if self.ecg_signal_fs != cf.HES_SAMPLING_RATE:
                b2 = scipy.interpolate.interp1d(range(1, 6), b2)(np.arange(1, 5.1, 160 / self.ecg_signal_fs))
            
            ecg_filtered: NDArray = scipy.signal.filtfilt(b2, 1, ecg_fil)
            ecg_filtered = ecg_filtered / np.max(ecg_filtered)
            
            ecg_filtered = self.smoother(
                    ecg_signal=ecg_filtered ** 2,
                    size=int(self.SMOOTHING_SIZE * self.ecg_signal_fs)
            )
            
            ecg = np.convolve(ecg_filtered, (np.ones((1, self.wins)) / self.wins).flatten())
    
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return ecg, ecg_fil
    
    def _get_ecg_segment_specification(
            self,
            ecg_signal: NDArray,
            _from:      int | float,
            _to:        int | float
    ) -> [int, float]:
        
        max_ind = None
        max_amp = None
        try:
            _from = int(max(0, _from))
            _to = int(min(_to, len(ecg_signal)))
            segment = ecg_signal[_from: _to]
            
            max_ind = np.argmax(segment)
            max_amp = np.max(segment)
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return max_ind, max_amp
    
    def _check_t_wave_detected(
            self,
            ecg_smooth: NDArray,
            qrs: NDArray,
            beat_index: int,
            peak: int,
            
    ) -> bool:
        
        is_t_wave_identified = False
        try:
            mean_rr = 0.5 * np.mean(np.diff(qrs[beat_index - self.NRR: beat_index]))
            
            if (
                    (beat_index >= 3 and (peak - qrs[beat_index - 1] <= self.qrs_offset_samples))
                    or (beat_index >= self.NRR and (peak - qrs[beat_index - 1]) <= mean_rr)
            ):
                s1 = np.mean(np.diff(ecg_smooth[peak - self.st_slope_samples: peak + 1]))
                s2 = np.mean(
                    np.diff(
                            ecg_smooth[int(qrs[beat_index - 1] - self.st_slope_samples) - 1:
                                       int(qrs[beat_index - 1]) + 1])
                    )
                is_t_wave_identified = np.abs(s1) <= np.abs(0.6 * s2)
                
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return is_t_wave_identified
    
    def post_process(
            self,
            ecg_smooth: NDArray,
            ecg_filter: NDArray,
            peaks:      NDArray,
    ) -> NDArray:
        
        beat_samples = np.zeros(len(peaks))
        try:
            qrs = np.zeros(len(peaks))
            
            sig_lev = thr_sig = np.max(ecg_smooth[: 2 * self.ecg_signal_fs + 1]) * 1 / 3
            noise_lev = thr_noise = np.mean(ecg_smooth[: 2 * self.ecg_signal_fs + 1]) * 1 / 2
            
            sig_lev1 = thr_sig1 = np.max(ecg_filter[: 2 * self.ecg_signal_fs + 1]) * 1 / 3
            noise_lev1 = thr_noise1 = np.mean(ecg_filter[: 2 * self.ecg_signal_fs + 1]) * 1 / 2
            
            beat_index = 0
            beat_index1 = 0
            for i, (p) in enumerate(peaks):
                peak_ind, peak_amp = self._get_ecg_segment_specification(
                        ecg_signal=ecg_filter,
                        _from=p - self.wins,
                        _to=p + 1
                )
                if peak_ind is None:
                    break
                
                # region Post Process
                if (p - qrs[beat_index - 1]) >= round(self.QRS_REGION * self.ecg_signal_fs):
                    p_ind, p_amp = self._get_ecg_segment_specification(
                            ecg_signal=ecg_smooth,
                            _from=int(qrs[beat_index - 1]) + self.qrs_offset_samples,
                            _to=p + 1
                    )
                    if p_amp is not None and p_amp > thr_noise * 0.2:
                        if beat_index >= len(peaks):
                            break
                        
                        sig_lev = 0.75 * p_amp + 0.25 * sig_lev
                        qrs[beat_index] = qrs[beat_index] + self.qrs_offset_samples + p_ind
                        
                        p_ind, p_amp = self._get_ecg_segment_specification(
                                ecg_signal=ecg_smooth,
                                _from=int(qrs[beat_index] - self.wins) + 1,
                                _to=int(qrs[beat_index]) + 2
                        )
                        beat_index += 1
                        if p_amp is not None and p_amp > thr_noise1 * 0.2:
                            if beat_index1 >= len(peaks):
                                break
                                
                            beat_samples[beat_index1] = qrs[beat_index] - self.wins + p_ind
                            if 0.25 < p_amp / sig_lev1 < 5:
                                sig_lev1 = 0.75 * p_amp + 0.25 * sig_lev1
                            beat_index1 += 1
                
                elif beat_index >= self.NRR:
                    mean_rr = float(1.66 * np.mean(np.diff(qrs[beat_index - self.NRR: beat_index])))
                    if p - qrs[beat_index - 1] > min([1 * self.ecg_signal_fs, mean_rr]):
                        _from = int(qrs[beat_index - 1]) + self.qrs_offset_samples
                        _to = int(p) + 2
                        
                        if _from > _to:
                            continue
                            
                        p_ind, p_amp = self._get_ecg_segment_specification(
                                ecg_signal=ecg_smooth,
                                _from=_from,
                                _to=_to
                        )
                        
                        if p_amp is not None:
                            sample = qrs[beat_index - 1] + self.qrs_offset_samples + p_ind
                            
                            thr_noise = thr_noise
                            if i < (len(peaks) - self.BEAT_OFFSET):
                                vec = ecg_smooth[int(qrs[beat_index - self.BEAT_OFFSET] + self.qrs_offset_samples):
                                                 int(peaks[i + self.BEAT_OFFSET]) + 1]
                                thr_noise = 0.5 * thr_noise + 0.5 * (np.mean(vec) * 1 / 2)
                            
                            if p_amp > thr_noise:
                                beat_index = beat_index + 1
                                if (beat_index - 1) >= len(peaks):
                                    break
                                
                                qrs[beat_index - 1] = sample
                                p_ind, p_amp = self._get_ecg_segment_specification(
                                        ecg_signal=ecg_filter,
                                        _from=int(sample - self.wins) + 1,
                                        _to=int(sample) + 2
                                )
                                
                                thr_noise = thr_noise1
                                if i < (len(peaks) - self.BEAT_OFFSET):
                                    seg = ecg_filter[int(
                                        qrs[beat_index - self.BEAT_OFFSET] + self.qrs_offset_samples - self.wins + 1):
                                                     int(peaks[i + self.BEAT_OFFSET]) + 1]
                                    thr_noise = 0.5 * thr_noise1 + 0.5 * (np.mean(seg) / 2)
                                
                                if p_amp > thr_noise:
                                    beat_index1 = beat_index1 + 1
                                    if (beat_index1 - 1) >= len(peaks):
                                        break
                                    
                                    beat_samples[beat_index1 - 1] = sample - self.wins + p_ind
                                    sig_lev1 = 0.75 * p_amp + 0.25 * sig_lev1
                                sig_lev = 0.75 * p_amp + 0.25 * sig_lev
                
                if p >= thr_sig:
                    is_t_wave_identified = self._check_t_wave_detected(
                            ecg_smooth=ecg_smooth,
                            qrs=qrs,
                            beat_index=beat_index,
                            peak=p
                    )
                    if not is_t_wave_identified:
                        if beat_index >= len(peaks):
                            break
                        
                        qrs[beat_index] = p
                        beat_index += 1
                        if peak_amp >= thr_sig1:
                            if beat_index1 >= len(peaks):
                                break
                            
                            beat_samples[beat_index1] = p - self.wins + peak_ind
                            beat_index1 += 1
                            
                            sig_lev1 = 0.125 * peak_amp + 0.875 * sig_lev1
                        sig_lev = 0.125 * p + 0.875 * sig_lev
                
                elif p < thr_sig:
                    noise_lev1 = 0.125 * peak_amp + 0.875 * noise_lev1
                    noise_lev = 0.125 * p + 0.875 * noise_lev
                # endregion Post Process
                
                # region Update Threshold
                if noise_lev != 0 or sig_lev != 0:
                    thr_sig = noise_lev + 0.25 * (np.abs(sig_lev - noise_lev))
                    thr_noise = 0.4 * thr_sig
                
                if noise_lev1 != 0 or sig_lev1 != 0:
                    thr_sig1 = noise_lev1 + 0.25 * (np.abs(sig_lev1 - noise_lev1))
                    thr_noise1 = 0.4 * thr_sig1
                # endregion Update Threshold
            
            beat_samples = beat_samples[:beat_index1].astype(int)
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return beat_samples
    
    def predict(
            self,
    ) -> [NDArray, NDArray]:
        
        samples = None
        symbols = None
        try:
            ecg_smooth, ecg_filter = self.pre_process()
            peaks = self.detect(
                    ecg_signal=ecg_smooth,
                    lookahead=int(self.MOVING_WINDOW * self.ecg_signal_fs)
            )
            
            if len(peaks) <= df.LIMIT_BEAT_SAMPLE_IN_SIGNAL:
                samples = df.initialize_two_beats_at_ecg_data(
                        len_ecg=len(self.ecg_signal),
                        sampling_rate=self.ecg_signal_fs
                )
                symbols = np.array([df.HolterSymbols.MARKED.value] * len(samples))
                
            else:
                samples = self.post_process(
                        ecg_smooth=ecg_smooth,
                        ecg_filter=ecg_filter,
                        peaks=peaks
                )
                
                func = BeatClassificationByIES()
                symbols = func.predict(
                        ecg_signal=self.ecg_signal,
                        ecg_signal_fs=self.ecg_signal_fs,
                        samples=samples,
                )
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return samples, symbols
