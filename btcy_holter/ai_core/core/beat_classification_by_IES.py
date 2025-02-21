from btcy_holter import *


class QRSAmplitudeSpecs:
    index = 0
    sample = 0
    rr = 0.0
    w20 = 0
    w50 = 0
    w80 = 0
    max_amp = 0.0
    min_amp = 0.0
    peak_peak = 0.0

    def __eq__(
            self,
            other:  Any
    ) -> bool:
        if isinstance(other, QRSAmplitudeSpecs):
            for key in self.__dict__:
                if key != 'methods':
                    if self.__dict__[key] != other.__dict__[key]:
                        return False
            return True
        return False


class QRSRRSpecs:
    RR1 = 0.0
    RR2 = 0.0
    RR3 = 0.0

    R12 = 0.0
    R23 = 0.0

    RR1_CAL = 0.0
    RR2_CAL = 0.0
    RR3_CAL = 0.0

    RR1_UDT = 0.0
    RR2_UDT = 0.0
    RR3_UDT = 0.0


class DecisionTree:
    _THR_RR_MIN:         Final[float] = 0.7         # second
    _THR_RR_MAX:         Final[float] = 1.25        # second
    _THR_W_LOCAL:        Final[float] = 1.6         # second
    _THR_AMP_MAX:        Final[float] = 1.8         # second
    _THR_AMP_MIN:        Final[float] = 0.55        # second

    # S_RUN
    _THR_RUN_RR_MIN:     Final[float] = 0.85        # second
    _THR_RUN_RR_MAX:     Final[float] = 1.15        # second
    _THR_RUN_RR1_MIN:    Final[float] = 0.6         # second
    _THR_RUN_RR1_MAX:    Final[float] = 1.28        # second

    _THR_QRS_MOR:        Final[float] = 0.2        # second

    def _check_function(
            self,
            args:                   List,
            multiplier:             int = 1
    ) -> NDArray:
        check = np.array(list(map(
            lambda x: self._THR_AMP_MIN * multiplier < x < self._THR_AMP_MAX * multiplier,
            args
        )))

        return check

    def _check_ratio_condition(
            self,
            qrs_ratio_to_calibrate: QRSAmplitudeSpecs,
            qrs_ratio_to_updated:   QRSAmplitudeSpecs,
    ) -> [bool, bool, bool]:

        check_width = False
        check_peaks = False
        check_amp_max = False
        check_amp_min = False

        ind_amp_max = list()
        ind_amp_min = list()
        try:
            width_data = np.array([qrs_ratio_to_calibrate.w20, qrs_ratio_to_calibrate.w50, qrs_ratio_to_calibrate.w80,
                                   qrs_ratio_to_updated.w20, qrs_ratio_to_updated.w50, qrs_ratio_to_updated.w80])

            check_width = np.count_nonzero(width_data < self._THR_W_LOCAL) > 2
            check_peaks = all(self._check_function([qrs_ratio_to_calibrate.peak_peak, qrs_ratio_to_updated.peak_peak]))

            ind_amp_max = self._check_function([qrs_ratio_to_calibrate.max_amp, qrs_ratio_to_updated.max_amp])
            check_amp_max = all(ind_amp_max)

            ind_amp_min = self._check_function([qrs_ratio_to_calibrate.min_amp, qrs_ratio_to_updated.min_amp])
            check_amp_min = all(ind_amp_min)

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return check_width, check_peaks, check_amp_max, check_amp_min, ind_amp_max, ind_amp_min

    def _check_amp(
            self,
            amp:       float,
            ind_amp:   NDArray,
            args:      List
    ) -> bool:
        condition = any([
            np.abs(amp) > self._THR_QRS_MOR and any(~ind_amp),
            np.abs(amp) <= self._THR_QRS_MOR and any(~self._check_function(args, multiplier=2)),
        ])

        return condition

    # @df.timeit
    def width_amp_predict(
            self,
            qrs_label:              Any,
            qrs_beat:               QRSAmplitudeSpecs,
            qrs_ratio_to_calibrate: QRSAmplitudeSpecs,
            qrs_ratio_to_updated:   QRSAmplitudeSpecs
    ) -> str:

        """
            Abnormal ==> using WIDTH + AMP to classify SVE, VES
            Normal ==> using WIDTH + AMP to classify Normal, VES
        """

        symbol = None
        try:
            (check_width, check_peak, check_amp_max, check_amp_min,
             ind_amp_max, ind_amp_min) = self._check_ratio_condition(
                qrs_ratio_to_calibrate=qrs_ratio_to_calibrate,
                qrs_ratio_to_updated=qrs_ratio_to_updated
            )
            if all([check_width, check_amp_max, check_amp_min, check_peak]):
                symbol = df.HolterSymbols.SVE.value \
                    if qrs_label != df.HolterSymbols.N.value \
                    else df.HolterSymbols.N.value

            else:
                if qrs_label != df.HolterSymbols.N.value:
                    if not check_amp_min:
                        check_amp = self._check_amp(
                            amp=qrs_beat.min_amp,
                            ind_amp=ind_amp_min,
                            args=[qrs_ratio_to_calibrate.min_amp, qrs_ratio_to_updated.min_amp]
                        )
                        symbol = df.HolterSymbols.VE.value \
                            if check_amp \
                            else df.HolterSymbols.SVE.value

                    elif not check_amp_max:
                        check_amp = self._check_amp(
                            amp=qrs_beat.max_amp,
                            ind_amp=ind_amp_max,
                            args=[qrs_ratio_to_calibrate.max_amp, qrs_ratio_to_updated.max_amp]
                        )
                        symbol = df.HolterSymbols.VE.value \
                            if check_amp \
                            else df.HolterSymbols.SVE.value

                    else:
                        symbol = df.HolterSymbols.VE.value
                else:
                    symbol = df.HolterSymbols.N.value \
                        if not all([check_amp_max, check_amp_min]) \
                        else df.HolterSymbols.VE.value

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return symbol

    # @df.timeit
    def normal_rr_predict(
            self,
            symbol_bef: str,
            qrs_rr:     QRSRRSpecs
    ) -> str:

        """
            Normal ==> using RR to classify Normal or SVes
        """

        symbol = None
        try:
            condition = any([
                qrs_rr.R23 > self._THR_RR_MAX,
                all(x < self._THR_RR_MIN for x in [qrs_rr.RR3_CAL, qrs_rr.RR3_UDT]),
                all(x > self._THR_RR_MAX for x in [qrs_rr.RR2_CAL, qrs_rr.RR2_UDT]),
                all(x <= self._THR_RR_MAX / 2 for x in [qrs_rr.RR3_CAL, qrs_rr.RR2_UDT])
            ])

            condition_2 = all([
                all(x > self._THR_RR_MAX for x in [qrs_rr.RR1_CAL, qrs_rr.RR1_UDT]),
                self._THR_RR_MIN < qrs_rr.R23 < self._THR_RR_MAX,
                symbol_bef == df.HolterSymbols.SVE.value
            ])

            if condition or condition_2:
                symbol = df.HolterSymbols.SVE.value
            else:
                symbol = df.HolterSymbols.N.value
                
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return symbol

    @staticmethod
    def logic_function(
            value:  float,
            ranges: List = None
    ) -> bool:
        cond = ranges is None
        if not cond:
            thr_from, thr_to = ranges
            if thr_from == 0:
                cond = value <= thr_to

            elif thr_to == 0:
                cond = value > thr_from

            else:
                cond = thr_from < value <= thr_to

        return cond

    # @df.timeit
    def abnormal_rr_predict(
            self,
            qrs_rr: QRSRRSpecs
    ) -> str | None:

        """
            Abnormal (Normal-SVes) ==> using RR to classify Normal, SVes
        """

        symbol = None
        try:
            if any(x < self._THR_RUN_RR1_MIN for x in [qrs_rr.RR3_CAL, qrs_rr.RR3_UDT]) and 0.9 < qrs_rr.R23 < 1.1:
                if (
                        any(x < self._THR_RR_MIN for x in [qrs_rr.RR3_CAL, qrs_rr.RR3_UDT])
                        and any(x < self._THR_RR_MIN for x in [qrs_rr.RR2_CAL, qrs_rr.RR2_UDT])
                ):
                    symbol = df.HolterSymbols.N.value
                else:
                    symbol = df.HolterSymbols.SVE.value

            elif any(x <= self._THR_RUN_RR_MIN for x in [qrs_rr.RR3_CAL, qrs_rr.RR3_UDT]):
                if qrs_rr.R23 >= self._THR_RUN_RR_MAX:
                    symbol = df.HolterSymbols.SVE.value

                elif (
                        all(x < self._THR_RUN_RR_MIN for x in [qrs_rr.RR2_CAL, qrs_rr.RR2_UDT])
                        and 0.9 <= qrs_rr.R23 <= 1.1
                ):
                    symbol = df.HolterSymbols.SVE.value

                else:
                    symbol = df.HolterSymbols.N.value

            else:
                symbol = df.HolterSymbols.N.value

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return symbol

    # @df.timeit
    def rr_width_predict(
            self,
            qrs_rr:     QRSRRSpecs,
            qrs_amp:    QRSAmplitudeSpecs,
    ) -> str | None:
        sym = None
        try:
            rr_check = any([
                all(x < self._THR_RR_MIN for x in [qrs_rr.RR3_CAL, qrs_rr.RR3_UDT]),
                all(x <= self._THR_RR_MAX / 2 for x in [qrs_rr.RR3_CAL, qrs_rr.RR2_UDT]),
                all(x > self._THR_RR_MAX for x in [qrs_rr.RR2_CAL, qrs_rr.RR2_UDT]),
                qrs_rr.R23 > self._THR_RR_MAX,
                qrs_amp.w20 > self._THR_W_LOCAL,
                qrs_amp.w50 > self._THR_W_LOCAL,
                qrs_amp.w80 > self._THR_W_LOCAL * 1.5,
            ])

            sym = df.HolterSymbols.SVE.value \
                if rr_check \
                else df.HolterSymbols.N.value

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return sym

    # @df.timeit
    def process(
            self,
            sym_bef:                str,
            qrs_rr:                 QRSRRSpecs,
            qrs_beat:               QRSAmplitudeSpecs,
            ratio_to_updated:       QRSAmplitudeSpecs,
            ratio_to_calibrate:     QRSAmplitudeSpecs,
    ) -> str | None:
        sym = None
        try:
            # Normal-N vs Abnormal-A
            sym = self.rr_width_predict(qrs_rr=qrs_rr, qrs_amp=ratio_to_calibrate)

            # N: N (N-S) vs V
            if sym == df.HolterSymbols.N.value:
                sym = self.width_amp_predict(qrs_label=sym,
                                             qrs_beat=qrs_beat,
                                             qrs_ratio_to_calibrate=ratio_to_calibrate,
                                             qrs_ratio_to_updated=ratio_to_updated)
                # N: N vs S
                if sym == df.HolterSymbols.N.value:
                    sym = self.normal_rr_predict(symbol_bef=sym_bef, qrs_rr=qrs_rr)

            else:
                # A: S (N-S) vs V
                sym = self.width_amp_predict(qrs_label=sym,
                                             qrs_beat=qrs_beat,
                                             qrs_ratio_to_calibrate=ratio_to_calibrate,
                                             qrs_ratio_to_updated=ratio_to_updated)
                # S: N vs S
                if sym == df.HolterSymbols.SVE.value:
                    sym = self.abnormal_rr_predict(qrs_rr=qrs_rr)

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return sym


class BeatClassificationByIES:
    SAMPLING_RATE:                  Final[int] = 250    # Hz
    
    _CONFIG_OFFSET_BEAT:             Final[int] = 3

    _NUM_BEAT_TO_CALI:               Final[int] = 30
    _NUM_ENOUGH_BEAT:                Final[int] = 10

    _THR_RUN_RR_MIN:                 Final[float] = 0.85
    _THR_RUN_RR1_MAX:                Final[float] = 1.28

    _SVM_THR_RR_RATIO_MAX:           Final[float] = 1.483
    _SVM_THR_RR_RATIO_MIN:           Final[float] = 0.789

    _THR_AMP_MIN_RATIO_MIN_LOCAL:    Final[float] = 0.8
    _THR_AMP_MIN_RATIO_MAX_LOCAL:    Final[float] = 1.2

    _THR_AMP_MAX_RATIO_MIN_LOCAL:    Final[float] = 0.8

    _THR_CALIBRATE_BEAT_POSITION:    Final[float] = 0.65

    _OFFSET_WIDTH_QRS:              Final[float] = 0.16     # second
    _OFFSET_WIDTH_QRS_BEFORE:       Final[float] = 0.06     # second
    
    def __init__(
            self,
            **kwargs
    ) -> None:
        try:
            self.debug_mode = cf.DEBUG_MODE
            
            self.ecg_signal = None
            self.decision_tree_func = DecisionTree()
    
            self.bef_ind = int(self._OFFSET_WIDTH_QRS_BEFORE * self.SAMPLING_RATE)
            self.aft_ind = int((self._OFFSET_WIDTH_QRS - self._OFFSET_WIDTH_QRS_BEFORE) * self.SAMPLING_RATE)
    
            self.bandpass_filter: Final[List] = [1.0, 30.0]  # Hz
    
            try:
                self.record_name = kwargs['record_name']
            except (Exception,):
                self.record_name = None
                
        except (Exception, ) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    # @df.timeit
    def _pre_processing(
            self,
            ecg_signal:     NDArray,
            ecg_signal_fs:  int,
            beat_samples:   NDArray
    ) -> Tuple[NDArray, NDArray]:

        try:
            if ecg_signal_fs != self.SAMPLING_RATE:
                ecg_signal, _ = resample_sig(
                        x=ecg_signal,
                        fs=ecg_signal_fs,
                        fs_target=self.SAMPLING_RATE
                )
                
                beat_samples = df.resample_beat_samples(
                        samples=beat_samples,
                        sampling_rate=ecg_signal_fs,
                        target_sampling_rate=self.SAMPLING_RATE
                )

            ecg_signal = ut.butter_bandpass_filter(
                    ecg_signal,
                    *self.bandpass_filter,
                    fs=self.SAMPLING_RATE,
                    order=3
            )

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return ecg_signal, beat_samples

    def _calibrate_position_beat(
            self,
            sample:         int | float,
            start_sample:   int,
            end_sample:     int
    ) -> int | float:
        position = sample
        try:
            ecg_seg = self.ecg_signal[start_sample: end_sample]
            amp_max = np.max(ecg_seg)
            amp_min = np.min(ecg_seg)

            ecg_seg = ecg_seg \
                if amp_min < 0 < amp_max and np.abs(amp_max / amp_min) > self._THR_CALIBRATE_BEAT_POSITION \
                else np.abs(ecg_seg)
            position = start_sample + np.argmax(ecg_seg)

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return position

    def argmin_position(
            self,
            start:          int,
            stop:           int,
            amplitude:      float,
            ratio:          float,
            bef_flag:       bool = False
    ) -> int:
        position = start
        try:
            ecg_segment = np.abs(self.ecg_signal[start: stop]) - np.abs(ratio * amplitude)
            index = np.flatnonzero(np.abs(np.diff(np.sign(ecg_segment))) == 2) + 1
            if len(index) == 0:
                if bef_flag:
                    position = start - np.argmin(ecg_segment)
                else:
                    position = start + np.argmin(ecg_segment)

            else:
                group_segments = np.split(ecg_segment, index)
                if bef_flag:
                    position = start - np.argmin(group_segments[-1]) + index[-1]
                else:
                    position = start + np.argmin(group_segments[0])

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return position

    # @df.timeit
    def beat_width(
            self,
            index:          int,
            sample:         int,
    ) -> QRSAmplitudeSpecs:

        width = QRSAmplitudeSpecs()
        try:
            width.index = index
            before_sample = max([0, sample - self.bef_ind])
            after_sample = min([sample + self.aft_ind, len(self.ecg_signal)])
            sample = self._calibrate_position_beat(sample, before_sample, after_sample)

            width.sample = sample
            before_sample = max([0, sample - self.bef_ind])
            after_sample = min([sample + self.aft_ind, len(self.ecg_signal)])

            ind_bef_20 = self.argmin_position(before_sample, sample, self.ecg_signal[sample], ratio=0.8, bef_flag=True)
            ind_aft_20 = self.argmin_position(sample, after_sample, self.ecg_signal[sample], ratio=0.8, bef_flag=False)
            width.w20 = ind_aft_20 - ind_bef_20

            ind_bef_50 = self.argmin_position(before_sample, sample, self.ecg_signal[sample], ratio=0.5, bef_flag=True)
            ind_aft_50 = self.argmin_position(sample, after_sample, self.ecg_signal[sample], ratio=0.5, bef_flag=False)
            width.w50 = ind_aft_50 - ind_bef_50

            ind_bef_80 = self.argmin_position(before_sample, sample, self.ecg_signal[sample], ratio=0.2, bef_flag=True)
            ind_aft_80 = self.argmin_position(sample, after_sample, self.ecg_signal[sample], ratio=0.2, bef_flag=False)
            width.w80 = ind_aft_80 - ind_bef_80

            sample_start = max([0, sample - 3 * self.bef_ind // 2])
            sample_stop = min([sample + 3 * self.aft_ind // 2, len(self.ecg_signal) - 1])

            ecg_segment = self.ecg_signal[sample_start:sample_stop]
            width.max_amp = np.max(ecg_segment)
            width.min_amp = np.min(ecg_segment)
            width.peak_peak = width.max_amp - width.min_amp

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return width

    def get_valid_symbol(
            self,
            samples:        NDArray,
            symbols:        NDArray,
            ecg_signal_fs:  int
    ) -> [NDArray, NDArray, bool, NDArray]:

        ind_r = np.array([])
        check_marked_beat = False
        try:
            check_marked_beat = (
                    (symbols[0] == df.HolterSymbols.MARKED.value)
                    or (symbols[0] in df.INVALID_BEAT_TYPE
                        and samples[0] <= int(self._OFFSET_WIDTH_QRS_BEFORE * ecg_signal_fs))
            )
            if check_marked_beat:
                symbols = symbols[1:]
                samples = samples[1:]

            ind_r = np.flatnonzero(symbols == df.HolterSymbols.IVCD.value)
            symbols[ind_r] = df.HolterSymbols.N.value

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return samples, symbols, check_marked_beat, ind_r

    @staticmethod
    def get_index_normal_value(
            values:         NDArray | List,
    ) -> NDArray:
        
        values = df.convert_to_array(values)
        index = np.flatnonzero(np.abs(values - np.mean(values)) <= np.std(values))

        return index

    def get_normal_values(
            self,
            values:         NDArray | List,
    ):
        values = df.convert_to_array(values)
        values = values[self.get_index_normal_value(values)]

        return values

    def get_valid_cali(
            self,
            values:         NDArray | List,
            index_delete:   NDArray
    ) -> NDArray:
        
        values = df.convert_to_array(values)
        values = np.delete(values, index_delete)
        values = self.get_normal_values(values)

        return values

    # @df.timeit
    def get_calibrate_threshold(
            self,
            beats_width:    NDArray[QRSAmplitudeSpecs]
    ) -> Tuple[QRSAmplitudeSpecs, QRSAmplitudeSpecs]:

        qrs_values = QRSAmplitudeSpecs()
        qrs_specs = QRSAmplitudeSpecs()
        try:
            sample_cali = np.array(list(map(lambda x: x.sample, beats_width))).astype(int)

            w20 = np.array(list(map(lambda x: x.w20, beats_width))).astype(int)
            w50 = np.array(list(map(lambda x: x.w50, beats_width))).astype(int)
            w80 = np.array(list(map(lambda x: x.w80, beats_width))).astype(int)

            max_amp = np.array(list(map(lambda x: x.max_amp, beats_width))).astype(float)
            min_amp = np.array(list(map(lambda x: x.min_amp, beats_width))).astype(float)
            peak_peak = np.array(list(map(lambda x: x.peak_peak, beats_width))).astype(float)

            rr_intervals = np.diff(sample_cali[:-1])
            rr_intervals = np.concatenate(([rr_intervals[0]], rr_intervals))
            rr_ratio = rr_intervals[:-1] / rr_intervals[1:]

            abnormal_index = np.flatnonzero(np.logical_or(
                rr_ratio < self._SVM_THR_RR_RATIO_MIN,
                rr_ratio > self._SVM_THR_RR_RATIO_MAX
            ))
            if len(abnormal_index) == len(rr_ratio):
                return qrs_values, qrs_specs

            elif len(rr_intervals) - len(abnormal_index) > len(rr_intervals) / 3:
                # [Normal > Abnormal] 1/ Remove abnormal RR and calculate mean of RR = Cali_RR
                cali_rr_intervals = np.delete(rr_intervals, abnormal_index + 1)
                cali_w20 = self.get_valid_cali(w20, abnormal_index)
                cali_w50 = self.get_valid_cali(w50, abnormal_index)
                cali_w80 = self.get_valid_cali(w80, abnormal_index)
                cali_max_amp = self.get_valid_cali(max_amp, abnormal_index)
                cali_min_amp = self.get_valid_cali(min_amp, abnormal_index)
                cali_peak_peak = self.get_valid_cali(peak_peak, abnormal_index)

            else:
                # [Normal < Abnormal] 1/ Find 2 RR at abnormal RR,
                # if abnormal 2RR ~ normal 2RR ==> abnormal 2 RR / 2 = normal RR
                abnormal_index = np.flatnonzero(rr_ratio > self._SVM_THR_RR_RATIO_MAX)
                intervals = rr_intervals[abnormal_index] + rr_intervals[abnormal_index - 1]
                ind = self.get_index_normal_value(intervals)
                normal_2rr_index = abnormal_index[ind]

                cali_rr_intervals = intervals[ind] // 2
                cali_w20 = self.get_normal_values(np.array(w20)[normal_2rr_index])
                cali_w50 = self.get_normal_values(np.array(w50)[normal_2rr_index])
                cali_w80 = self.get_normal_values(np.array(w80)[normal_2rr_index])
                cali_max_amp = self.get_normal_values(np.array(max_amp)[normal_2rr_index])
                cali_min_amp = self.get_normal_values(np.array(min_amp)[normal_2rr_index])
                cali_peak_peak = self.get_normal_values(np.array(peak_peak)[normal_2rr_index])

            qrs_values.rr = cali_rr_intervals
            qrs_specs.rr = np.mean(cali_rr_intervals)

            qrs_values.w20 = cali_w20
            qrs_specs.w20 = np.mean(cali_w20)

            qrs_values.w50 = cali_w50
            qrs_specs.w50 = np.mean(cali_w50)

            qrs_values.w80 = cali_w80
            qrs_specs.w80 = np.mean(cali_w80)

            qrs_values.max_amp = cali_max_amp
            qrs_specs.max_amp = np.mean(cali_max_amp)

            qrs_values.min_amp = cali_min_amp
            qrs_specs.min_amp = np.mean(cali_min_amp)

            qrs_values.peak_peak = cali_peak_peak
            qrs_specs.peak_peak = np.mean(cali_peak_peak)

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return qrs_values, qrs_specs

    def cal_rr_specs(
            self,
            samples:        NDArray,
            qrs_calibrate:  QRSAmplitudeSpecs,
            qrs_updated:    QRSAmplitudeSpecs,
    ) -> QRSRRSpecs:

        qrs_rr = QRSRRSpecs()
        try:
            [qrs_rr.RR1, qrs_rr.RR2, qrs_rr.RR3] = np.diff(samples)

            qrs_rr.R12 = qrs_rr.RR1 / qrs_rr.RR2
            qrs_rr.R23 = qrs_rr.RR2 / qrs_rr.RR3

            qrs_rr.RR1_CAL = qrs_rr.RR1 / qrs_calibrate.rr
            qrs_rr.RR2_CAL = qrs_rr.RR2 / qrs_calibrate.rr
            qrs_rr.RR3_CAL = qrs_rr.RR3 / qrs_calibrate.rr

            qrs_rr.RR1_UDT = qrs_rr.RR1 / qrs_updated.rr
            qrs_rr.RR2_UDT = qrs_rr.RR2 / qrs_updated.rr
            qrs_rr.RR3_UDT = qrs_rr.RR3 / qrs_updated.rr

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return qrs_rr

    def cal_amp_specs(
            self,
            qrs:            QRSAmplitudeSpecs,
            qrs_calibrate:  QRSAmplitudeSpecs,
            qrs_updated:    QRSAmplitudeSpecs,
    ) -> [QRSAmplitudeSpecs, QRSAmplitudeSpecs]:

        ratio_to_calibrate = QRSAmplitudeSpecs()
        ratio_to_updated = QRSAmplitudeSpecs()
        try:
            ratio_to_calibrate.w20 = qrs.w20 / qrs_calibrate.w20
            ratio_to_calibrate.w50 = qrs.w50 / qrs_calibrate.w50
            ratio_to_calibrate.w80 = qrs.w80 / qrs_calibrate.w80
            ratio_to_calibrate.min_amp = qrs.min_amp / qrs_calibrate.min_amp
            ratio_to_calibrate.max_amp = qrs.max_amp / qrs_calibrate.max_amp
            ratio_to_calibrate.peak_peak = qrs.peak_peak / qrs_calibrate.peak_peak

            ratio_to_updated = QRSAmplitudeSpecs()
            ratio_to_updated.w20 = qrs.w20 / qrs_updated.w20
            ratio_to_updated.w50 = qrs.w50 / qrs_updated.w50
            ratio_to_updated.w80 = qrs.w80 / qrs_updated.w80
            ratio_to_updated.min_amp = qrs.min_amp / qrs_updated.min_amp
            ratio_to_updated.max_amp = qrs.max_amp / qrs_updated.max_amp
            ratio_to_updated.peak_peak = qrs.peak_peak / qrs_updated.peak_peak

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return ratio_to_calibrate, ratio_to_updated

    def _get_beat_cali(
            self,
            symbols:        NDArray
    ) -> NDArray:

        ind = np.arange(self._NUM_BEAT_TO_CALI)
        try:
            for beat_valid in df.VALID_BEAT_TYPE:
                beat_index = np.flatnonzero(symbols == beat_valid)
                if len(beat_index) == 0:
                    group_index = np.split(beat_index, np.flatnonzero(np.abs(np.diff(beat_index)) != 1) + 1)
                    group_index = sorted(group_index, key=lambda x: -len(x))[0]
                    if len(group_index) >= self._NUM_BEAT_TO_CALI:
                        ind = group_index[:self._NUM_BEAT_TO_CALI]
                        break

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return ind

    def get_beat_width_of_valid_beats(
            self,
            samples:        NDArray,
            symbols:        NDArray,
    ) -> [NDArray, NDArray]:

        svm_symbols = np.array([df.HolterSymbols.N.value] * len(samples))
        widths = list()
        try:
            if len(samples) < self._NUM_BEAT_TO_CALI:
                ind = np.arange(len(samples))
            else:
                index_calibrate = self._get_beat_cali(symbols)

                ind_tmp = np.setdiff1d(np.arange(len(samples)), index_calibrate)
                index_beat_valid = ind_tmp[np.flatnonzero(np.isin(symbols[ind_tmp], df.VALID_BEAT_TYPE))]
                ind = np.sort(np.unique(np.concatenate((index_calibrate, index_beat_valid))))

                index_invalid = np.setdiff1d(np.arange(len(samples)), ind)
                svm_symbols[index_invalid] = df.HolterSymbols.SVE.value

            ind = ind[ind <= len(samples) - 1]
            ind = ind[np.logical_and(samples[ind] >= self.bef_ind, samples[ind] <= samples[-1] + self.aft_ind)]
            widths = np.array(list(map(
                lambda i: self.beat_width(i, samples[i]),
                ind
            )))

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return widths, svm_symbols

    def update_threshold(
            self,
            symbols:        NDArray,
            index:          int
    ) -> int:
        return np.count_nonzero(np.isin(
                symbols[index - self._CONFIG_OFFSET_BEAT: index],
                [df.HolterSymbols.SVE.value, df.HolterSymbols.VE.value]
        )) == 0

    # @df.timeit
    def svm_classification(
            self,
            samples:        NDArray,
            symbols:        NDArray
    ) -> NDArray:

        try:
            if len(samples) < self._NUM_ENOUGH_BEAT:
                return symbols
            
            # region Calibrate
            qrs_widths, svm_symbols = self.get_beat_width_of_valid_beats(samples, symbols)
            specs = self.get_calibrate_threshold(qrs_widths[:self._NUM_BEAT_TO_CALI])
            if any(x is None for x in specs):
                st.get_error_exception(error='Could not calibrate width.')

            qrs_val, qrs_calibrate = specs
            qrs_updated = deepcopy(qrs_calibrate)
            # endregion Calibrate

            # region Classification
            normal_beat = dict()
            for qrs in qrs_widths:
                if qrs.index <= self._CONFIG_OFFSET_BEAT or qrs.index == len(samples) - 1:
                    svm_symbols[qrs.index] = symbols[qrs.index]
                    continue

                # Using RR to classify N vs S/V
                qrs_rr = self.cal_rr_specs(
                    samples=samples[qrs.index - self._CONFIG_OFFSET_BEAT: qrs.index + 1],
                    qrs_calibrate=qrs_calibrate,
                    qrs_updated=qrs_updated,
                )

                # Using AMP to classify N vs S vs V
                ratio_to_calibrate, ratio_to_updated = self.cal_amp_specs(
                    qrs=qrs,
                    qrs_calibrate=qrs_calibrate,
                    qrs_updated=qrs_updated,
                )

                symbol_before = None if qrs.index == 0 else svm_symbols[qrs.index - 1]
                sym = self.decision_tree_func.process(
                    sym_bef=symbol_before,
                    qrs_rr=qrs_rr,
                    qrs_beat=qrs,
                    ratio_to_updated=ratio_to_updated,
                    ratio_to_calibrate=ratio_to_calibrate
                )

                # region confirm actual S/V or not
                if sym != df.HolterSymbols.N.value:
                    if (
                            sym == df.HolterSymbols.VE.value
                            and svm_symbols[qrs.index - 1] == df.HolterSymbols.N.value
                            and bool(normal_beat)
                    ):
                        amplitude_check = any([
                            qrs.max_amp / normal_beat['max_amp'] > self._THR_AMP_MIN_RATIO_MAX_LOCAL,
                            qrs.max_amp / normal_beat['max_amp'] < self._THR_AMP_MAX_RATIO_MIN_LOCAL,
                            qrs.min_amp / normal_beat['min_amp'] > self._THR_AMP_MIN_RATIO_MAX_LOCAL,
                            qrs.min_amp / normal_beat['min_amp'] < self._THR_AMP_MIN_RATIO_MIN_LOCAL
                        ])
                        if amplitude_check:
                            sym = df.HolterSymbols.VE.value
                        else:
                            sym = df.HolterSymbols.N.value

                    svm_symbols[qrs.index] = sym

                else:  # S_Run

                    if (
                            svm_symbols[qrs.index - 1] == df.HolterSymbols.SVE.value
                            and qrs_rr.R12 < self._THR_RUN_RR1_MAX
                            and all(x < self._THR_RUN_RR_MIN for x in [qrs_rr.RR2_UDT, qrs_rr.RR2_CAL])
                    ):
                        svm_symbols[qrs.index] = df.HolterSymbols.SVE.value \
                            if qrs_rr.R23 > self._THR_RUN_RR_MIN \
                            else df.HolterSymbols.N.value

                    elif self.update_threshold(svm_symbols, index=qrs.index):
                        qrs_val.rr = np.concatenate((qrs_val.rr[1:], [samples[qrs.index] - samples[qrs.index - 1]]))
                        qrs_val.w20 = np.concatenate((qrs_val.w20[1:], [qrs.w20]))
                        qrs_val.w50 = np.concatenate((qrs_val.w50[1:], [qrs.w50]))
                        qrs_val.w80 = np.concatenate((qrs_val.w80[1:], [qrs.w80]))
                        qrs_val.min_amp = np.concatenate((qrs_val.min_amp[1:], [qrs.min_amp]))
                        qrs_val.max_amp = np.concatenate((qrs_val.max_amp[1:], [qrs.max_amp]))
                        qrs_val.peak_peak = np.concatenate((qrs_val.peak_peak[1:], [qrs.peak_peak]))

                        qrs_updated.rr = np.mean(qrs_val.rr)
                        qrs_updated.w20 = np.mean(qrs_val.w20)
                        qrs_updated.w50 = np.mean(qrs_val.w50)
                        qrs_updated.w80 = np.mean(qrs_val.w80)
                        qrs_updated.min_amp = np.mean(qrs_val.min_amp)
                        qrs_updated.max_amp = np.mean(qrs_val.max_amp)
                        qrs_updated.peak_peak = np.mean(qrs_val.peak_peak)

                    normal_beat['min_amp'] = qrs.min_amp
                    normal_beat['max_amp'] = qrs.max_amp
                    normal_beat['peak_peak'] = qrs.peak_peak
                # endregion confirm actual S/V or not
            # endregion Classification

            svm_symbols = np.array(svm_symbols)

        except (Exception,) as error:
            svm_symbols = np.array(symbols)
            st.write_error_log(error if self.record_name is None else f'{self.record_name} - {error}',
                               class_name=self.__class__.__name__)

        return svm_symbols

    def merge_symbols(
            self,
            symbols:        NDArray,
            hes_symbols:    NDArray,
    ) -> NDArray:
        concat_symbols = deepcopy(symbols)
        try:
            index = np.flatnonzero(symbols == df.HolterSymbols.SVE.value)
            index = (index[:, None] + np.arange(-1, 2, 1)).flatten()
            index = index[np.flatnonzero(np.logical_and(index >= 0, index <= len(symbols) - 1))]
            concat_symbols[index] = hes_symbols[index]

        except (Exception,) as error:
            st.write_error_log(
                error if self.record_name is None else f'{self.record_name} - {error}',
                class_name=self.__class__.__name__
            )

        return concat_symbols

    def _review(
            self,
            samples:        NDArray,
            symbols:        NDArray,
            hes_symbols:    NDArray,
            merge_symbols:  NDArray
    ) -> None:
        try:
            fig, (axis) = plt.subplots(4, 1, sharey='all', sharex='all')
            fig.suptitle(f'{str(self.record_name).upper()}')
            fig.subplots_adjust(hspace=0.12, wspace=0.2, left=0.03, right=0.98, bottom=0.03, top=0.93)

            def bbox_color(x):
                return dict(boxstyle='round', fc=df.BEAT_COLORS.get(x, 'white'), ec=df.BEAT_COLORS_EC.get(x, 'black'))

            from btcy_holter.utils import beat_annotations
            ref_ann = wf.rdann(record_name=self.record_name, extension='atr')
            ref_sample, ref_symbol = beat_annotations(ref_ann)
            ref_sample = ref_sample * self.SAMPLING_RATE // ref_ann.fs

            y_max = min(2, np.max(self.ecg_signal[samples[np.flatnonzero(np.isin(symbols, df.VALID_BEAT_TYPE))]])) + 0.3
            for i, (title, sym, beats) in enumerate([
                ('ACTUAL', ref_symbol, ref_sample),
                ('AI PREDICTED', symbols, samples),
                ('HES ALGORITHM', hes_symbols, samples),
                ('FINAL', merge_symbols, samples)
            ]):
                axis[i].plot(self.ecg_signal)
                axis[i].vlines(beats, ymin=-y_max - 0.3, ymax=y_max, lw=0.5, color='r', linestyles='dotted')
                [axis[i].annotate(x, xy=(beats[j], y_max), xycoords='data', textcoords='data', bbox=bbox_color(x))
                 for j, x in enumerate(sym)]

                ind = np.flatnonzero(np.logical_or(
                        sym == df.HolterSymbols.VE.value,
                        sym == df.HolterSymbols.SVE.value
                ))
                for j in ind:
                    axis[i].axvspan(beats[j] - 50, beats[j] + 50, color='yellow', alpha=0.5)

                axis[i].set_title(f'{title}')
                axis[i].set_xlim(0, len(self.ecg_signal))

            plt.show()
            plt.close()

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    def predict(
            self,
            ecg_signal:     NDArray,
            ecg_signal_fs:  int,
            samples:        NDArray,
    ) -> NDArray:
        
        symbol = None
        try:
            self.ecg_signal, resample_samples = self._pre_processing(
                    ecg_signal=ecg_signal,
                    ecg_signal_fs=ecg_signal_fs,
                    beat_samples=deepcopy(samples)
            )
            
            symbol = self.svm_classification(
                    resample_samples,
                    symbols=np.array([df.HolterSymbols.N.value] * len(resample_samples))
            )
            
        except (Exception,) as error:
            st.write_error_log(
                error if self.record_name is None else f'{self.record_name} - {error}',
                class_name=self.__class__.__name__
            )
            
        return symbol
    
    # @df.timeit
    def process(
            self,
            ecg_signal:     NDArray,
            ecg_signal_fs:  int,
            samples:        NDArray,
            symbols:        NDArray,
            run_evaluate:   bool = False,
            review:         bool = False
    ) -> NDArray:
        
        try:
            if len(samples) == 0:
                return symbols
            
            if np.count_nonzero(np.isin(symbols, df.VALID_BEAT_TYPE)) < self._NUM_BEAT_TO_CALI:
                return symbols

            # region Process
            samples, symbols, check_marked_beat, ind_r = self.get_valid_symbol(samples, symbols, ecg_signal_fs)
            self.ecg_signal, resample_samples = self._pre_processing(
                    ecg_signal=ecg_signal,
                    ecg_signal_fs=ecg_signal_fs,
                    beat_samples=deepcopy(samples)
            )
            hes_symbols = self.svm_classification(resample_samples, symbols)
            if not run_evaluate:
                merge_symbols = self.merge_symbols(symbols, hes_symbols)
            else:
                merge_symbols = hes_symbols
            # endregion Process

            # region Pos
            review and self.debug_mode and self._review(resample_samples, symbols, hes_symbols, merge_symbols)
            
            symbols = deepcopy(merge_symbols)
            if len(ind_r) > 0:
                symbols[ind_r] = df.HolterSymbols.IVCD.value

            if check_marked_beat:
                symbols = np.concatenate((
                    [df.HolterSymbols.MARKED.value],
                    symbols
                ))

        except (Exception,) as error:
            st.write_error_log(
                error if self.record_name is None else f'{self.record_name} - {error}',
                class_name=self.__class__.__name__
            )

        return symbols
