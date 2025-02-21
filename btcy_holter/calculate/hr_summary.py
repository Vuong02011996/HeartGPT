from btcy_holter import *


class DailyHeartRateSummary(
        pt.Summary        
):
    BEAT_OFFSET:            Final[int]      = int((cf.MIN_RR_INTERVAL / 2) * df.MILLISECOND)
    MAX_STRIP_SUGGESTION:   Final[int]      = 10

    MIN_STRIP_LENGTH:       Final[int]      = 7                          # seconds
    MAX_STRIP_LENGTH:       Final[int]      = 10                         # seconds

    MAX_STRIP_POLARS:       Final[int]      = 20

    def __init__(
            self,
            beat_df:            pl.DataFrame,
            event_df:           pl.DataFrame,
            record_config:      sr.RecordConfigurations,
            all_files:          List | str                  = None,
            show_log:           bool                        = True
    ) -> None:
        """
        Initializes the DailyHeartRateSummary class with the given parameters.

        Args:
            beat_df (pl.DataFrame): DataFrame containing beat data.
            event_df (pl.DataFrame): DataFrame containing event data.
            record_config (sr.RecordConfigurations): Configuration for the record.
            all_files (List | str, optional): List of all files or a single file path.
            show_log (bool, optional): Flag to show log messages.
        """
        
        try:
            super(DailyHeartRateSummary, self).__init__(
                beat_df=beat_df,
                record_config=record_config,
                event_df=event_df,
                all_files=all_files,
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
            
            self._study_epoch:          Final[NDArray]      = self.beat_df['EPOCH'].to_numpy()
            self._show_log:             Final[bool]         = show_log
            
            self._minute_hr_df:         pl.DataFrame        = pl.DataFrame()
            self._summary:              Dict                = dict()
            
            self._valid_index: Final[NDArray] = self.get_valid_region(
                    event_invalid=['ARTIFACT', 'PAUSE'],
                    beat_valid=[df.HolterBeatTypes.N.value, df.HolterBeatTypes.IVCD.value]
            )
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
            
    def __get_id_event(
            self
    ) -> str:
        
        _id = None 
        try:
            _id = df.generate_event_id(
                    event_id_existed=self.event_df['id'].unique().to_list()
            )
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            
        return _id
    
    def __get_date_update(
            self,
            file_index: NDArray
    ) -> List:
        
        date = list()
        try:
            for i, all_files in enumerate(self.all_files):
                if i not in file_index:
                    continue
                    
                start = df.convert_timestamp_to_epoch_time(
                        timestamp=all_files['start'],
                        timezone=self.record_config.timezone,
                        dtype=float,
                        ms=True
                )
                start = df.convert_epoch_time_to_timestamp(
                        epoch_time=start,
                        timezone=self.record_config.timezone,
                        is_iso_format=False
                )
                date.append(start.replace(hour=0, minute=0, second=0).isoformat())
            
            date = list(set(date))
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return date
    
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
            start_epochs:       List,
            stop_epochs:        List,
    ) -> [List, List, List]:
        
        try:
            epoch = epoch - self.BEAT_OFFSET
            if len(epoch) <= df.LIMIT_BEAT_CALCULATE_HR:
                return avg_hrs, start_epochs, stop_epochs
            
            stop_epoch = epoch + self.MAX_STRIP_LENGTH * df.MILLISECOND
            ind = np.flatnonzero(stop_epoch < np.max(epoch))
            if len(ind) == 0:
                return avg_hrs, start_epochs, stop_epochs
            
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
                start_epochs.append(start_strip)
                stop_epochs.append(stop_strip)

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return avg_hrs, start_epochs, stop_epochs
    
    # @df.timeit
    def __calculate_minutely_hr(
            self,
            study_epoch:    NDArray,
            valid:          NDArray,
            date:           str = None,
    ) -> Dict | None:
        
        result = dict()
        result['date']:              Any = date
        result['avgHr']:             Any = df.DEFAULT_HR_VALUE
        
        result['maxHr']:             Any = df.DEFAULT_HR_VALUE
        result['maxHrStart']:        float = 0.0
        result['maxHrStop']:         float = 0.0
        
        result['minHr']:             Any = df.DEFAULT_HR_VALUE
        result['minHrStart']:        float = 0.0
        result['minHrStop']:         float = 0.0

        try:
            # region Get Valid Index
            if len(study_epoch) == 0:
                return result
            
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
            avg_hrs         = list()
            start_epochs    = list()
            stop_epochs     = list()
            
            for (duration, epoch) in group_epoch_valid:
                if self.MIN_STRIP_LENGTH <= duration <= self.MAX_STRIP_LENGTH:
                    hr = self.__cal_hr(epoch)
                    if hr == df.DEFAULT_HR_VALUE:
                        continue
                     
                    avg_hrs.append(hr)
                    start_epochs.append(epoch[0] - self.BEAT_OFFSET)
                    stop_epochs.append(epoch[-1] + self.BEAT_OFFSET)
                    
                else:
                    avg_hrs, start_epochs, stop_epochs = self.__get_strip_hr(
                            epoch=epoch,
                            avg_hrs=avg_hrs,
                            start_epochs=start_epochs,
                            stop_epochs=stop_epochs,
                    )
                    pass
            # endregion Process through Valid Region
            
            # region Calculate HR
            if len(avg_hrs) == 0:
                return result
            
            result['avgHr'] = int(round(geometric_mean(avg_hrs)))
            
            avg_hrs = np.array(avg_hrs)
            ind_max = np.argmax(avg_hrs)
            
            result['maxHr']              = avg_hrs[ind_max]
            result['maxHrStart']         = start_epochs[ind_max]
            result['maxHrStop']          = stop_epochs[ind_max]
            
            ind_min = np.argmin(avg_hrs)
            result['minHr']              = avg_hrs[ind_min]
            result['minHrStart']         = start_epochs[ind_min]
            result['minHrStop']          = stop_epochs[ind_min]
            # endregion Calculate HR
            pass
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return result
    
    # @df.timeit
    def __process_filter_and_range_index(
            self
    ) -> None:
        try:
            if self._minute_hr_df.height == 0:
                return
            
            start_time = time.time()
            self._minute_hr_df = (
                self._minute_hr_df
                .sort(
                        'date'
                )
            )
            
            ranges = df.get_index_within_multiple_ranges(
                    nums=self._study_epoch,
                    low=self._minute_hr_df['startEpoch'],
                    high=self._minute_hr_df['stopEpoch'],
                    is_filter_index=False
            )
            
            self._minute_hr_df = (
                self._minute_hr_df
                .with_columns(
                        [
                            pl.Series('startIndex', ranges[:, 0]),
                            pl.Series('stopIndex', ranges[:, -1]),
                        ]
                )
                .filter(
                        [
                            (pl.col('startIndex') != -1)
                            & (pl.col('stopIndex') != -1)
                            & (pl.col('startIndex') < pl.col('stopIndex'))
                            & (pl.col('stopIndex') - pl.col('startIndex') > df.LIMIT_BEAT_CALCULATE_HR)
                        ]
                )
                .select(
                        [
                            'date',
                            'startIndex',
                            'stopIndex'
                        ]
                )
            )
            
            total_times = time.time() - start_time
            self._show_log and st.LOGGING_SESSION.info(f'--- Collect: {total_times}s -> ({self._minute_hr_df.height})')
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    def cal(self, z):
        return self.__calculate_minutely_hr(z['startIndex'], z['stopIndex'] + 1, z['date'])
    
    # @df.timeit
    def __process_strips(
            self
    ) -> None:
        try:
            if self._minute_hr_df.height == 0:
                return
            
            start_time = time.time()
            minutes = list(filter(
                    lambda y: y['avgHr'] != df.DEFAULT_HR_VALUE,
                    map(
                        lambda x: self.__calculate_minutely_hr(
                                self._study_epoch[x['startIndex']: x['stopIndex'] + 1],
                                self._valid_index[x['startIndex']: x['stopIndex'] + 1],
                                x['date']
                        ),
                        self._minute_hr_df.to_dicts()
                    )
            ))
            
            self._minute_hr_df = self._minute_hr_df.clear()
            if len(minutes) == 0:
                return
                
            self._minute_hr_df = pl.DataFrame(minutes)
            total_times = time.time() - start_time
            step_time = total_times / self._minute_hr_df.height if self._minute_hr_df.height > 0 else 0
            
            self._show_log and st.LOGGING_SESSION.info(
                f'--- Processing minute strips: {total_times}s   '
                f'|| step: {step_time}s '
                f'({self._minute_hr_df.height})'
            )
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def __process_hr_strips(
            self
    ) -> None:
        try:
            if self._minute_hr_df.height == 0:
                return
            
            self._minute_hr_df = (
                self._minute_hr_df
                .lazy()
                .filter(
                        pl.col('avgHr') != df.DEFAULT_HR_VALUE
                )
                .group_by(
                        'date'
                )
                .agg(
                        [
                            pl.struct(['maxHr', 'maxHrStart', 'maxHrStop'])
                            .sort_by('maxHr', descending=True)
                            .head(self.MAX_STRIP_SUGGESTION)
                            .alias('maxHrStrips'),
                            
                            pl.struct(['minHr', 'minHrStart', 'minHrStop'])
                            .sort_by('minHr', descending=False)
                            .head(self.MAX_STRIP_SUGGESTION)
                            .alias('minHrStrips'),
                            
                            pl.col('avgHr')
                        ]
                )
                .collect()
            )
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _get_time_dfs(
            self,
            file_index: NDArray | List = None
    ) -> None:
        try:
            if file_index is not None:
                if len(file_index) == 0:
                    file_index = None
                else:
                    file_index = self.__get_date_update(file_index)
            
            for i, all_files in enumerate(self.all_files):
                start = self.convert_epoch_time_to_datetime(
                    epoch_time=self.convert_timestamp_to_epoch_time(all_files['start']),
                )
                date = deepcopy(start).replace(hour=0, minute=0, second=0).isoformat()
                if file_index is not None and date not in file_index:
                    continue
                    
                stop = self.convert_epoch_time_to_datetime(
                        epoch_time=self.convert_timestamp_to_epoch_time(all_files['stop']) - df.MILLISECOND_IN_MINUTE,
                )
                
                min_df = pl.DataFrame(
                    {
                        "start": pl.datetime_range(
                                start=datetime(start.year, start.month, start.day, start.hour, start.minute, 0),
                                end=datetime(stop.year, stop.month, stop.day, stop.hour, stop.minute, 59),
                                interval="1m",
                                time_unit="ms",
                                eager=True,
                        ).to_list(),
                    }
                )
                
                if min_df.height == 0:
                    min_df = pl.DataFrame(
                            {
                                'start': [datetime(start.year, start.month, start.day, start.hour, start.minute, 0)]
                            }
                    )
                
                pass
                min_df = (
                    min_df
                    .lazy()
                    .with_columns(
                        [
                            (pl.col("start") + pl.duration(minutes=1)).alias("stop")
                        ]
                    )
                    .with_columns(
                        [
                            pl.lit(date)
                            .alias('date'),
                            
                            pl.col("start")
                            .map_elements(lambda x: x.replace(tzinfo=start.tzinfo).timestamp() * df.MILLISECOND)
                            .alias("startEpoch"),
                            
                            pl.col("stop")
                            .map_elements(lambda x: x.replace(tzinfo=start.tzinfo).timestamp() * df.MILLISECOND)
                            .alias("stopEpoch"),
                        ]
                    )
                    .collect()
                )
                
                self._minute_hr_df = pl.concat([self._minute_hr_df, min_df])
                pass
            pass
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _get_minute_strips(
            self
    ) -> None:
        try:
            if self._minute_hr_df.height == 0:
                return
            
            self.__process_filter_and_range_index()
            self.__process_strips()
            self.__process_hr_strips()
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    def _format_suggest_strip(
            self,
            strip: Dict
    ) -> Dict:
        
        fmt_strip = dict()
        try:
            fmt_strip['avgHr'] = strip['avgHr']
            fmt_strip['head'] = self.convert_epoch_time_to_timestamp(epoch_time=strip['head'])
            fmt_strip['tail'] = self.convert_epoch_time_to_timestamp(epoch_time=strip['tail'])
            pass
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            
        return fmt_strip
    
    def __get_strip(
            self,
            strip: Dict,
    ) -> [float, float]:
        """
        Gets the strip start and stop times for the given strip.

        Args:
            strip (Dict): Dictionary containing strip information.

        Returns:
            [float, float]: Tuple containing strip start and stop times.
        """
        
        strip_start = deepcopy(strip['head'])
        strip_stop = deepcopy(strip['tail'])
        try:
            if strip_stop - strip_start != self.MAX_STRIP_LENGTH * df.MILLISECOND:
                center = strip_start + ((strip_stop - strip_start) / 2)
                strip_start = center - ((self.MAX_STRIP_LENGTH * df.MILLISECOND) / 2)
                strip_start = max([self.beat_df['EPOCH'][0], strip_start])
                
                strip_start = float(strip_start)
                strip_stop = float(strip_start + (self.MAX_STRIP_LENGTH * df.MILLISECOND))
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return strip_start, strip_stop
    
    def _get_event(
            self,
            event_strip: Tuple[str, Dict]
    ) -> Dict:
        
        event: Dict = dict()
        try:
            event_type, event_strip = event_strip
            event_type = 'MAX_HR' if event_type == 'maxHr' else 'MIN_HR'
            
            if all(event_strip[x] != df.DEFAULT_INVALID_VALUE for x in ['head', 'tail']):
                ind = self.get_index_within_start_stop(
                        start_epoch=event_strip['head'],
                        stop_epoch=event_strip['tail']
                )
                channel = df.get_channel_from_channel_column(self.beat_df['CHANNEL'][ind].to_numpy())
                channel = df.find_most_frequency_occurring_values(channel) + 1

                duration = (event_strip['tail'] - event_strip['head']) / df.MILLISECOND
                duration = round(duration, 2)

            else:
                channel = df.get_channel_from_channel_column(self.beat_df['CHANNEL'].to_numpy())
                channel = df.find_most_frequency_occurring_values(channel) + 1
                ind = list()
                duration = 0
                
            noise_channels = df.get_noise_channels(
                study_df=self.beat_df,
                record_config=self.record_config,
                all_files=self.all_files,
                epoch_start=event_strip['head'],
                epoch_stop=event_strip['tail'],
                strip_channel=channel - 1
            )
            
            comment = df.HOLTER_ALL_EVENT_TITLES[event_type] + f' with {int(event_strip["value"])} bpm.'
            
            start = self.convert_epoch_time_to_timestamp(event_strip['head'])
            stop = self.convert_epoch_time_to_timestamp(event_strip['tail'])
            strip_start, strip_stop = self.__get_strip(event_strip)

            event['id']:                        Any = self.__get_id_event()
            event['start']:                     Any = start
            event['stop']:                      Any = stop

            event['type']:                      Any = event_type
            event['isIncludedToReport']:        Any = True
            event['maxHr']:                     Any = event_strip['value']
            event['minHr']:                     Any = event_strip['value']
            event['avgHr']:                     Any = event_strip['value']

            event['channel']:                   Any = channel
            event['countBeats']:                Any = len(ind)
            event['duration']:                  Any = duration
            event['comment']:                   Any = comment

            strip = dict()
            strip['start']:                     Any = self.convert_epoch_time_to_timestamp(strip_start)
            strip['stop']:                      Any = self.convert_epoch_time_to_timestamp(strip_stop)
            strip['avgHr']:                     Any = event['avgHr']
            strip['channel']:                   Any = event['channel']
            strip['comment']:                   Any = event['comment']
            strip['noiseChannels']:             Any = noise_channels
            event['strips'] = [strip]
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return event
    
    # @df.timeit
    def _process_min(
            self,
            file_index: NDArray | List = None
    ) -> None:
        try:
            self._get_time_dfs(file_index)
            self._get_minute_strips()
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _process_daily(
            self,
            file_index: NDArray | List = None
    ) -> None:
        """
        Gets the date heart rate for the study period.

        Returns:
            pl.DataFrame: DataFrame containing date heart rates.
        """
        self._summary = dict()
        self._summary['hrSummary']         = list()
        self._summary['hrMinMaxEvents']    = dict()
        
        try:
            self._summary['hrMinMaxEvents']['maxHr'] = None
            self._summary['hrMinMaxEvents']['minHr'] = None
            
            total_dates = self.__get_date_update(
                    file_index
                    if file_index is not None
                    else np.arange(len(self.all_files))
            )
            if self._minute_hr_df.height > 0:
                total_dates = list(set(total_dates) - set(self._minute_hr_df['date'].unique()))
                
                # region Valid date
                for date in self._minute_hr_df.sort('date').to_dicts():
                    summary = dict()
                    summary['date']  = date['date']
                    summary['avgHr'] = {
                        'value': ut.calculate_hr_by_geometric_mean(date['avgHr'])
                    }
                    
                    max_hr_strip = date['maxHrStrips'][0]
                    
                    summary['maxHr'] = {
                        'time'     : self.convert_epoch_time_to_timestamp(epoch_time=max_hr_strip['maxHrStart']),
                        'value'    : max_hr_strip['maxHr'],
                        'hrsOnBeat': {
                            'avgHr': max_hr_strip['maxHr'],
                            'head' : self.convert_epoch_time_to_timestamp(epoch_time=max_hr_strip['maxHrStart']),
                            'tail' : self.convert_epoch_time_to_timestamp(epoch_time=max_hr_strip['maxHrStop']),
                        }
                    }
                    
                    if (
                            self._summary['hrMinMaxEvents']['maxHr'] is None
                            or (
                                self._summary['hrMinMaxEvents']['maxHr'] is not None
                                and self._summary['hrMinMaxEvents']['maxHr']['value'] < summary['maxHr']['value']
                        )
                    ):
                        self._summary['hrMinMaxEvents']['maxHr'] = {
                            'head':     max_hr_strip['maxHrStart'],
                            'tail':     max_hr_strip['maxHrStop'],
                            'value':    summary['maxHr']['value']
                        }
        
                    min_hr_strip = date['minHrStrips'][0]
                    summary['minHr'] = {
                        'time'     : self.convert_epoch_time_to_timestamp(epoch_time=min_hr_strip['minHrStart']),
                        'value'    : min_hr_strip['minHr'],
                        'hrsOnBeat': {
                            'avgHr': min_hr_strip['minHr'],
                            'head' : self.convert_epoch_time_to_timestamp(epoch_time=min_hr_strip['minHrStart']),
                            'tail' : self.convert_epoch_time_to_timestamp(epoch_time=min_hr_strip['minHrStop']),
                        }
                    }
                    
                    if (
                            self._summary['hrMinMaxEvents']['minHr'] is None
                            or (
                                self._summary['hrMinMaxEvents']['minHr'] is not None
                                and self._summary['hrMinMaxEvents']['minHr']['value'] > summary['minHr']['value']
                            )
                    ):
                        self._summary['hrMinMaxEvents']['minHr'] = {
                            'head':     min_hr_strip['minHrStart'],
                            'tail':     min_hr_strip['minHrStop'],
                            'value':    min_hr_strip['minHr']
                        }
                    pass
        
                    max_hrs_suggestion = list(map(
                            lambda x: self._format_suggest_strip(
                                    strip = {
                                        'avgHr':    x['maxHr'],
                                        'head':     x['maxHrStart'],
                                        'tail':     x['maxHrStop']
                                    }
                            ),
                            date['maxHrStrips']
                    ))
                    
                    min_hrs_suggestion = list(map(
                            lambda x: self._format_suggest_strip(
                                    strip = {
                                        'avgHr':    x['minHr'],
                                        'head':     x['minHrStart'],
                                        'tail':     x['minHrStop']
                                    }
                            ),
                            date['minHrStrips']
                    ))
                    
                    summary['suggestHrsOnBeat'] = {
                        'maxHr': max_hrs_suggestion,
                        'minHr': min_hrs_suggestion
                    }
                    self._summary['hrSummary'].append(summary)
                    pass
                # endregion Valid date

            # region Invalid dates
            for date in total_dates:
                summary = dict()
                summary['date'] = date
                summary['avgHr'] = {}
                summary['minHr'] = {}
                summary['maxHr'] = {}
                summary['suggestHrsOnBeat'] = {
                    'maxHr': [],
                    'minHr': []
                }
                self._summary['hrSummary'].append(summary)
            # endregion Invalid dates

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _process_events(
            self
    ) -> None:
        try:
            pass
            if 'hrMinMaxEvents' not in self._summary.keys():
                return
            
            self._summary['hrMinMaxEvents'] = list(map(
                    self._get_event,
                    filter(lambda x: x[1] is not None, self._summary['hrMinMaxEvents'].items())
            ))
            pass
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__) 
        
    @df.timeit
    def process(
            self,
            file_index: NDArray | List = None,
    ) -> Dict:
        """
        Processes the heart rate data and generates a summary.
        """
        
        try:
            self._process_min(file_index)
            self._process_daily(file_index)
            self._process_events()
            
            file_index is not None and self._summary.pop('hrMinMaxEvents', None)
            self._show_log and st.LOGGING_SESSION.info(
                    f'--- Processed Heart Rate Summary: {json.dumps(self._summary, cls=df.NumpyEncoder)}'
            )

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return self._summary