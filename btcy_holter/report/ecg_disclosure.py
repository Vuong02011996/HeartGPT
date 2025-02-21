from btcy_holter import *


class EcgDisclosure(
        pt.Summary
):
    SHIFT_DURATION: Final[float] = 5  # Minutes
    
    def __init__(
            self,
            beat_df:            pl.DataFrame,
            event_df:           pl.DataFrame,
            all_files:          List[Dict],
            record_config:      sr.RecordConfigurations,
    ) -> None:
        try:
            super(EcgDisclosure, self).__init__(
                beat_df=beat_df,
                event_df=event_df,
                record_config=record_config,
                all_files=all_files
            )
            self.offset_time:       Final[int] = int(df.MILLISECOND / self.record_config.sampling_rate)
            
            self.hrs = df.MILLISECOND_IN_MINUTE / np.diff(self.beat_df['EPOCH'].to_numpy())
            self.hrs = np.insert(self.hrs, 0, df.DEFAULT_HR_VALUE)

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
            
    def _get_continuous_time(
            self
    ) -> List:
        """
        Gets the continuous time intervals from the file data.

        Returns:
            List: List of tuples containing start and stop epochs.
        """
        
        continuous_times = list()
        try:
            all_file_df = (
                self.convert_all_file_to_dataframe()
                .select(
                        [
                            'startEpoch',
                            'stopEpoch'
                        ]
                )
            )
            
            continuous_times = list()
            current_start, current_end = all_file_df[0, "startEpoch"], all_file_df[0, "stopEpoch"]
            for (row) in all_file_df.iter_rows(named=True):
                if row["startEpoch"] > current_end:
                    continuous_times.append((current_start, current_end))
                    current_start, current_end = row["startEpoch"], row["stopEpoch"]
                else:
                    current_end = max(current_end, row["stopEpoch"])
            
            continuous_times.append((current_start, current_end))
            pass
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return continuous_times
    
    def _get_hourly_intervals(
            self,
            continuous_times: List
    ) -> List:
        
        hourly_intervals = list()
        try:
            for begin, end in continuous_times:
                ranges = np.arange(begin, end - df.SECOND_IN_HOUR * df.MILLISECOND, self.SHIFT_DURATION * df.MILLISECOND_IN_MINUTE)
                if len(ranges) > 0:
                    hourly_intervals.extend(list(zip(ranges, ranges + df.SECOND_IN_HOUR * df.MILLISECOND)))
                pass
        
            if len(hourly_intervals) == 0:
                hourly_intervals = [max(continuous_times, key=lambda x: x[1] - x[0])]
                pass
            pass
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return hourly_intervals
    
    def _calculate_quality(
            self,
            times: Tuple[float, float],
            index: NDArray
    ) -> Dict:
        
        results = dict()
        results['start']:            int     = times[0]
        results['stop']:             int     = times[-1]
        results['countBeats']:       int     = 0
        results['noisePercentage']:  float   = 0
        results['burdenHrValid']:    float   = 0
        try:
            if len(index) > 0:
                results['countBeats'] = len(index)
                
            # region burdenHrInvalid
            if len(index) >= df.LIMIT_BEAT_CALCULATE_HR:
                count = np.count_nonzero(np.logical_and(
                        self.hrs[index] >= df.HR_MIN_THR,
                        self.hrs[index] <= df.HR_MAX_THR
                ))
                results['burdenHrValid'] = (count / len(index)) * 100
            # endregion burdenHrInvalid
            
            # region noisePercentage
            if self.event_df.height > 0:
                noise_times = (
                    self.event_df
                    .filter(
                        [
                            pl.col('type').is_in(['ARTIFACT'])
                            & (
                                    pl.col('start').is_between(results['start'], results['stop'])
                                    | pl.col('stop').is_between(results['start'], results['stop'])
                                    | ((pl.col('start') <= results['start']) & (pl.col('stop') >= results['stop']))
                                    | ((pl.col('start') > results['start']) & (pl.col('stop') < results['stop']))
                            )
                        ]
                    )
                    .unique(
                            [
                                'id'
                            ]
                    )
                    .select(
                            [
                                'start',
                                'stop'
                            ]
                    )
                    .to_numpy()
                    .copy()
                )
                if len(noise_times) > 0:
                    noise_times[noise_times < results['start']]    = results['start']
                    noise_times[noise_times > results['stop']]     = results['stop']
                    results['noisePercentage'] = np.sum(np.diff(noise_times, axis=1)) / (results['stop'] - results['start'])
                    pass
            # endregion noisePercentage

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return results
    
    def _log(
            self,
            result: Dict
    ) -> None:
        try:
            bool(result) and self.log_func(f' - ecgDisclosure: {result}')

        except (Exception,) as error:
            cf.DEBUG_MODE and st.write_error_log(error, class_name=self.__class__.__name__)

    @df.timeit
    def process(
            self,
            show_log: bool = True
    ) -> Dict:
        
        result = dict()
        result['start']     = None
        result['stop']      = None
        result['channel']   = cf.CHANNEL
        try:
            continuous_times = self._get_continuous_time()
            hourly_intervals = self._get_hourly_intervals(continuous_times)
            
            if len(hourly_intervals) == 0:
                return result
            
            if len(hourly_intervals) == 1:
                result['start'] = hourly_intervals[0][0]
                result['stop']  = hourly_intervals[0][-1]
                
            else:
                hourly_intervals = np.array(hourly_intervals)
                range_index = df.get_index_within_multiple_ranges(
                        nums=self.beat_df['EPOCH'],
                        low=hourly_intervals[:, 0],
                        high=hourly_intervals[:, 1],
                        is_filter_index=False
                )
                self.hrs = df.MILLISECOND_IN_MINUTE / np.diff(self.beat_df['EPOCH'].to_numpy())
                self.hrs = np.insert(self.hrs, 0, df.DEFAULT_HR_VALUE)
                
                tmp = list()
                for (begin, end), times in zip(range_index, hourly_intervals):
                    _ = self._calculate_quality(
                            times=times,
                            index=np.arange(begin, end + 1)
                    )
                    tmp.append(_)
                
                hourly_df = (
                    pl.DataFrame(tmp)
                    .sort(
                            [
                                'noisePercentage',
                                'burdenHrValid',
                                'countBeats'
                            ],
                            descending=[
                                False,
                                True,
                                True
                            ]
                    )
                    .select(
                            [
                                'start',
                                'stop',
                            ]
                    )
                    .row(
                            index=0,
                            named=True
                    )
                )
                result['start'] = hourly_df['start']
                result['stop']  = hourly_df['stop']
                pass
            
            ind = self.get_index_within_start_stop(
                    start_epoch=result['start'],
                    stop_epoch=result['stop']
            )
            if len(ind) > 0:
                channel = df.get_channel_from_channel_column(self.beat_df['CHANNEL'][ind].to_numpy())
                result['channel']  = int(df.find_most_frequency_occurring_values(channel) + 1)
                
            result['start']     = self.convert_epoch_time_to_timestamp(result['start'])
            result['stop']      = self.convert_epoch_time_to_timestamp(result['stop'])
            
            show_log and self._log(result)

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return result
