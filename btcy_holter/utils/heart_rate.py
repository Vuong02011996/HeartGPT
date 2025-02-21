from btcy_holter import *


def format_hr_value(
        hr_value: Union[int, float]
) -> int:
    return int(round(hr_value))


def calculate_hr_by_geometric_mean(
        values: List | NDArray
) -> int:
    value = df.DEFAULT_HR_VALUE
    try:
        if isinstance(values, list):
            values = np.array(values)
            
        ind_valid = np.flatnonzero(np.logical_and(
            values >= df.HR_MIN_THR,
            values <= df.HR_MAX_THR
        ))
        values = values[ind_valid]
        
        if len(values) == 1:
            value = format_hr_value(float(values[0]))
        elif len(values) > 1:
            value = format_hr_value(geometric_mean(values))
            
    except (Exception,) as error:
        st.write_error_log(error)
        
    return value


class HeartRate:
    WINDOW_SIZE: Final[int] = 10        # seconds
    
    def __init__(
            self,
            beats:          NDArray | List | pl.Series,
            symbols:        NDArray | List | pl.Series,
            sampling_rate:  int,
    ) -> None:
        try:
            self.sampling_rate: Final[int] = sampling_rate
            
            self.beats:         NDArray = self.init(beats)
            self.symbols:       NDArray = self.init(symbols)
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    def init(
            self,
            value: NDArray | List | pl.Series
    ) -> NDArray:
        try:
            if isinstance(value, pl.Series):
                return value.to_numpy().copy()
            
            elif isinstance(value, list):
                return np.array(value)
            
            else:
                return value.copy()
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    def _preprocess(
            self
    ) -> None:
        
        try:
            self.beats = df.convert_to_array(self.beats)
            self.symbols = df.convert_to_array(self.symbols)
            if np.any(np.in1d(self.symbols, np.array(df.SYMBOL_TO_HES.keys()))):
                self.symbols = np.asarray(df.convert_symbol_to_hes_beat(self.symbols))
            self.symbols[self.symbols == df.HolterBeatTypes.MARKED.value] = df.HolterBeatTypes.OTHER.value
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    def _get_group_valid_index(
            self
    ) -> List:
        
        group_valid = list()
        try:
            index = np.flatnonzero(np.isin(self.symbols, df.VALID_HES_BEAT_TYPE))
            group_valid = np.split(
                    index,
                    np.flatnonzero(np.logical_or(np.abs(np.diff(index)) != 1, np.diff(self.beats[index]) < 0)) + 1
            )
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return group_valid

    def __cal(
            self,
            index: NDArray
    ) -> Any:

        heart_rate = df.DEFAULT_HR_VALUE
        try:
            if len(index) <= df.LIMIT_BEAT_SAMPLE_IN_SIGNAL:
                return heart_rate

            rrs = np.diff(self.beats[index])
            hr_values = df.SECOND_IN_MINUTE / (rrs[rrs > 0] / self.sampling_rate)

            index_valid_hr = np.flatnonzero(np.logical_and(
                hr_values >= df.HR_MIN_THR,
                hr_values <= df.HR_MAX_THR
            ))
            if len(index_valid_hr) > 0:
                heart_rate = calculate_hr_by_geometric_mean(hr_values[index_valid_hr])

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return heart_rate

    def __cal_strip_hr_greater_than_window_size(
            self,
            index:          NDArray,
            return_index:   bool        = False
    ) -> NDArray:

        heart_rate = np.array([])
        try:

            lows = self.beats[index]
            high = self.beats[index] + (self.sampling_rate * self.WINDOW_SIZE)

            if index[-1] < len(self.beats) - 1:
                i = index[-1] + 1
            else:
                i = index[-1]
                
            ind = np.flatnonzero(high <= self.beats[i])
            
            lows = lows[ind]
            high = high[ind]

            index_ranges = df.get_index_within_multiple_ranges(
                nums=self.beats[index],
                low=lows,
                high=high
            )

            for (begin, end) in index_ranges:
                if end - begin <= df.LIMIT_BEAT_SAMPLE_IN_SIGNAL:
                    continue

                value = self.__cal(index[begin: end + 1])
                if value == df.DEFAULT_HR_VALUE:
                    continue
                
                if return_index:
                    value = [index[begin], value]
                    if len(heart_rate) == 0:
                        heart_rate = np.array([value])
                    else:
                        heart_rate = np.row_stack([heart_rate, value])
                    
                else:
                    heart_rate = np.append(heart_rate, value)

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return heart_rate

    def __cal_strip_hr_less_than_window_size(
            self,
            index: NDArray
    ) -> NDArray:

        heart_rate = np.array([])
        try:
            heart_rate = [self.__cal(index)]

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return heart_rate

    def _calculate_heart_rate_strips(
            self,
            index: NDArray
    ) -> NDArray:

        heart_rate = np.array([])
        try:
            if len(index) < df.LIMIT_BEAT_SAMPLE_IN_SIGNAL:
                return heart_rate

            rrs = np.diff(self.beats[index])
            duration = (rrs[rrs > 0]).sum() / self.sampling_rate
            if duration >= self.WINDOW_SIZE:
                heart_rate_function = self.__cal_strip_hr_greater_than_window_size
            else:
                heart_rate_function = self.__cal_strip_hr_less_than_window_size
            heart_rate = heart_rate_function(index)

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return heart_rate

    def process(
            self,
    ) -> Dict:
        
        result = dict()
        result['minHr'] = df.DEFAULT_HR_VALUE
        result['avgHr'] = df.DEFAULT_HR_VALUE
        result['maxHr'] = df.DEFAULT_HR_VALUE
        
        try:
            if len(self.beats) != len(self.symbols):
                st.get_error_exception('Length of beats and symbols must be equal',
                                       class_name=self.__class__.__name__)
                
            if len(self.beats) == len(self.symbols) == 0:
                return result
                
            self._preprocess()
            group_index_valid = self._get_group_valid_index()
            if len(group_index_valid) == 0:
                return result

            # region Calculate min/max HR
            heart_rate = np.array(list(chain.from_iterable(map(
                lambda x: self._calculate_heart_rate_strips(x),
                group_index_valid
            ))))
            if len(heart_rate) == 0:
                return result
            
            heart_rate = np.round(heart_rate).astype(int)
            ind_valid = np.flatnonzero(np.logical_and(
                heart_rate >= df.HR_MIN_THR,
                heart_rate <= df.HR_MAX_THR
            ))
            if len(ind_valid) == 0:
                return result
            
            heart_rate = heart_rate[ind_valid]
            if len(heart_rate) > 0:
                duration = df.calculate_duration(
                        beats=self.beats,
                        sampling_rate=self.sampling_rate
                )
                if duration < self.WINDOW_SIZE:
                    heart_rate = calculate_hr_by_geometric_mean(heart_rate)
                    result['minHr'] = heart_rate
                    result['avgHr'] = heart_rate
                    result['maxHr'] = heart_rate
                else:
                    result['minHr'] = np.min(heart_rate)
                    result['maxHr'] = np.max(heart_rate)
                    result['avgHr'] = calculate_hr_by_geometric_mean(heart_rate)
            # endregion Calculate min/max HR

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return result
    
    def process_strips(
            self,
            index: NDArray
    ) -> NDArray:

        hr_values = np.array([])
        try:
            valid = np.isin(self.symbols[index], df.VALID_HES_BEAT_TYPE)
            group_index = np.split(
                    index,
                    np.flatnonzero(np.logical_or(np.abs(np.diff(index)) != 1, np.diff(valid) != 0)) + 1
            )
            for ind in group_index:
                if len(ind) <= df.LIMIT_BEAT_SAMPLE_IN_SIGNAL:
                    continue
                
                duration = df.calculate_duration(
                        beats=self.beats,
                        index=ind,
                        sampling_rate=self.sampling_rate
                )
                if duration < self.WINDOW_SIZE:
                    value = self.__cal_strip_hr_less_than_window_size(ind)
                    if len(value) == 0:
                        continue
                    
                    if len(hr_values) == 0:
                        hr_values = np.array([ind[0], value[0]])
                    else:
                        hr_values = np.row_stack([hr_values, [ind[0], value[0]]])
                    continue
                
                value = self.__cal_strip_hr_greater_than_window_size(ind, return_index=True)
                if len(value) > 0:
                    if len(hr_values) == 0:
                        hr_values = value
                    else:
                        hr_values = np.row_stack([hr_values, value])
                    pass
            
            if len(hr_values.shape) == 1:
                hr_values = np.array([hr_values])
                
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return hr_values
