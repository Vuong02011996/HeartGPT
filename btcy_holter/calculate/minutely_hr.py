from btcy_holter import *


class MinutelyHr(
        pt.Summary
):
    SECOND_MS:              Final[int] = df.MILLISECOND_IN_MINUTE
    
    MIN_STRIP_LENGTH:       Final[int]      = 7                          # seconds
    MAX_STRIP_LENGTH:       Final[int]      = 10                         # seconds
    
    def __init__(
            self,
            beat_df:            pl.DataFrame,
            event_df:           pl.DataFrame,
            record_config:      sr.RecordConfigurations,
    ) -> None:
        """
        Initializes the MinutelyHr class with the given parameters.
        """
        
        try:
            super(MinutelyHr, self).__init__(
                beat_df=beat_df,
                event_df=event_df,
                record_config=record_config,
            )
            
            self.beat_df: pl.DataFrame = (
                self.beat_df
                .select(
                        [
                            'EPOCH',
                            'CHANNEL',
                            'BEAT',
                            'BEAT_TYPE',
                            'FILE_INDEX'
                        ]
                )
            )
            
            self._study_epoch:  Final[NDArray] = self.beat_df['EPOCH'].to_numpy()
            
            self._valid_index:  Final[NDArray] = self.get_valid_region(
                    event_invalid=['ARTIFACT', 'PAUSE'],
                    beat_valid=[df.HolterBeatTypes.N.value, df.HolterBeatTypes.IVCD.value]
            )
            
            if isinstance(self.record_config.record_start_time, str):
                self.record_config.record_start_time = (
                    self.convert_timestamp_to_epoch_time(self.record_config.record_start_time))
                
            if isinstance(self.record_config.record_stop_time, str):
                self.record_config.record_stop_time = (
                    self.convert_timestamp_to_epoch_time(self.record_config.record_stop_time))

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    def __cal_hr(
            self,
            epoch: NDArray,
    ) -> int:
        heart_rate = df.DEFAULT_HR_VALUE
        try:
            if len(epoch) <= df.LIMIT_BEAT_CALCULATE_HR:
                return heart_rate
            
            heart_rate = df.MILLISECOND_IN_MINUTE / (np.diff(epoch))
            heart_rate = ut.calculate_hr_by_geometric_mean(heart_rate)
            pass

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return heart_rate
    
    # @df.timeit
    def __get_strip_hr(
            self,
            epoch:              NDArray,
            avg_hrs:            List,
    ) -> List:
        
        try:
            if len(epoch) <= df.LIMIT_BEAT_CALCULATE_HR:
                return avg_hrs
            
            stop_epoch = epoch + self.MAX_STRIP_LENGTH * df.MILLISECOND
            ind = np.flatnonzero(stop_epoch < np.max(epoch))
            if len(ind) == 0:
                return avg_hrs
            
            start_epoch = epoch[ind]
            stop_epoch = stop_epoch[ind]

            grs = df.get_index_within_multiple_ranges(
                    nums=epoch,
                    low=start_epoch,
                    high=stop_epoch,
                    is_filter_index=False
            )
            
            for (begin, end), start_strip, stop_strip in zip(grs, start_epoch, stop_epoch):
                if begin == -1 or end == -1:
                    continue
                
                if end < begin:
                    continue
                
                if end - begin <= df.LIMIT_BEAT_CALCULATE_HR:
                    continue
                
                hr = self.__cal_hr(epoch[begin: end + 1])
                if hr == df.DEFAULT_HR_VALUE:
                    continue
                    
                avg_hrs.append(hr)

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return avg_hrs
    
    def __calculate_minutely_hr(
            self,
            start_index: int,
            stop_index: int,
    ) -> Dict:
        """
        Calculates the minutely heart rate for the given epoch range.
        """
        
        result = dict()
        result['avgHr'] = df.DEFAULT_HR_VALUE
        result['minHr'] = df.DEFAULT_HR_VALUE
        result['maxHr'] = df.DEFAULT_HR_VALUE

        try:
            study_epoch = self._study_epoch[start_index: stop_index + 1]
            # region Get Valid Index
            if len(study_epoch) == 0:
                return result
            
            valid = self._valid_index[start_index: stop_index + 1]
            if len(valid) == 0:
                return result
            
            index = np.flatnonzero(valid == self.VALID)
            if len(index) == 0:
                return result
            
            group_epoch_valid = np.split(
                    study_epoch[index],
                    np.flatnonzero(np.diff(index) != 1) + 1
            )
            
            group_epoch_valid = list(filter(
                    lambda y: y[0] >= self.MIN_STRIP_LENGTH,
                    map(lambda x: [(x[-1] - x[0]) / df.MILLISECOND, x], group_epoch_valid)
            ))
            # endregion Get Valid Index
            
            # region Process through Valid Region
            avg_hrs = list()
            for (duration, epoch) in group_epoch_valid:
                if self.MIN_STRIP_LENGTH <= duration <= self.MAX_STRIP_LENGTH:
                    hr = self.__cal_hr(epoch)
                    if hr == df.DEFAULT_HR_VALUE:
                        continue
                     
                    avg_hrs.append(hr)
                    
                else:
                    avg_hrs = self.__get_strip_hr(
                            epoch=epoch,
                            avg_hrs=avg_hrs,
                    )
                    pass
            # endregion Process through Valid Region
            
            # region Calculate HR
            if len(avg_hrs) == 0:
                return result
            
            result['avgHr'] = int(round(geometric_mean(avg_hrs)))
            result['maxHr'] = int(round(np.max(avg_hrs)))
            result['minHr'] = int(round(np.min(avg_hrs)))
            # endregion Calculate HR
            pass
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return result
    
    # @df.timeit
    def process(
            self,
    ) -> Dict:
        """
        Processes the minutely heart rate data.
        """
        
        summary = dict()
        try:
            start_hourly = deepcopy(self.record_config.record_start_time)
            stop_hourly = deepcopy(self.record_config.record_stop_time)

            if start_hourly % self.SECOND_MS != 0:
                start_hourly = start_hourly - start_hourly % self.SECOND_MS

            if stop_hourly % self.SECOND_MS != 0:
                stop_hourly = stop_hourly + (self.SECOND_MS - stop_hourly % self.SECOND_MS)

            if start_hourly >= stop_hourly:
                st.get_error_exception('Invalid Start and Stop Time', class_name=self.__class__.__name__)

            minute_range_intervals = np.arange(start_hourly, stop_hourly + self.SECOND_MS, self.SECOND_MS)
            
            ranges_index = df.get_index_within_multiple_ranges(
                    nums=self._study_epoch,
                    low=minute_range_intervals[:-1],
                    high=minute_range_intervals[1:],
                    is_filter_index=False
            )
            
            minutely_hr_df: pl.DataFrame = pl.DataFrame(list(map(
                lambda x: self.__calculate_minutely_hr(*x),
                ranges_index
            )))
            pass
            
            summary['minHrs'] = (
                minutely_hr_df
                .select(
                        [
                            'minHr'
                        ]
                )
                .to_numpy()
                .flatten()
            )

            summary['maxHrs'] = (
                minutely_hr_df
                .select(
                        [
                            'maxHr'
                        ]
                )
                .to_numpy()
                .flatten()
            )

            summary['avgHrs'] = (
                minutely_hr_df
                .select(
                        [
                            'avgHr'
                        ]
                )
                .to_numpy()
                .flatten()
            )

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return summary
