from btcy_holter import *


class Summary(
        ABC
):
    """
    Abstract base class for summarizing data from a Holter monitor.
    """
    
    INVALID:                Final[int] = 0
    VALID:                  Final[int] = 1

    def __init__(
            self,
            beat_df:                pl.DataFrame,
            record_config:          sr.RecordConfigurations,
            all_files:              List[Dict]                  = None,
            event_df:               pl.DataFrame                = None,
            pvc_morphology:         pl.DataFrame                = None,
            num_of_processes:       int                         = os.cpu_count(),
            **kwargs
    ) -> None:
        """
        Initializes the Summary class with the provided parameters.
        """

        try:
            self.all_files:        Final[List[Dict]] = all_files
            self.record_config:    Final[sr.RecordConfigurations] = record_config
            self.num_of_processes: Final[int] = num_of_processes
            
            self.beat_df:          pl.DataFrame = beat_df

            self.event_df:         pl.DataFrame = df.pl_get_valid_events(event_df) if event_df is not None else None
            self.pvc_mor_df:       pl.DataFrame = pvc_morphology if pvc_morphology is not None else None
            
            self.log_func:         Any = kwargs.get('log_func', st.LOGGING_SESSION.info)
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    def get_value(
            self,
            column: str,
            index:  Any
    ) -> Any:
        
        if column not in self.beat_df.columns:
            st.get_error_exception('Column not found in the dataframe.', class_name=self.__class__.__name__)
            
        return self.beat_df[column][int(index)]
    
    def convert_timestamp_to_epoch_time(
            self,
            timestamp: str
    ) -> Any:
        
        epoch_time = None
        try:
            if timestamp is None:
                return epoch_time
            
            if not isinstance(timestamp, str):
                return epoch_time
            
            epoch_time = df.convert_timestamp_to_epoch_time(
                timestamp=timestamp,
                timezone=self.record_config.timezone,
                dtype=int,
                ms=True
            )
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
            
        return epoch_time
    
    def convert_epoch_time_to_timestamp(
            self,
            epoch_time:     float | int,
    ) -> Any:
        
        timestamp = None
        try:
            if epoch_time is None:
                return timestamp
            
            if isinstance(epoch_time, str):
                return timestamp
            
            timestamp = df.convert_epoch_time_to_timestamp(
                    epoch_time=epoch_time,
                    timezone=self.record_config.timezone,
                    is_iso_format=True
            )
                
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
            
        return timestamp
    
    def convert_epoch_time_to_datetime(
            self,
            epoch_time: float | int,
    ) -> Any:
        
        dt = None
        try:
            if epoch_time is None:
                return dt
            
            if isinstance(epoch_time, str):
                return dt
            
            dt = df.convert_epoch_time_to_timestamp(
                    epoch_time=epoch_time,
                    timezone=self.record_config.timezone,
                    is_iso_format=False
            )
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return dt
    
    def convert_all_file_to_dataframe(
            self,
    ) -> pl.DataFrame:
        
        all_file_df = pl.DataFrame()
        try:
            if self.all_files is None:
                return all_file_df
            
            all_file_df = (
                pl.DataFrame(self.all_files)
                .with_columns(
                    [
                        pl.col("start")
                        .map_elements(lambda x: int(self.convert_timestamp_to_epoch_time(x)), pl.Int64)
                        .alias("startEpoch"),

                        pl.col("stop")
                        .map_elements(lambda x: int(self.convert_timestamp_to_epoch_time(x)), pl.Int64)
                        .alias("stopEpoch")
                    ]
                )
            )

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return all_file_df
    
    def mark_invalid_beat_types(
            self,
            event_invalid: List[str] = ('ARTIFACT', )
    ) -> None:

        try:
            beat_types = self.beat_df['BEAT_TYPE'].to_numpy().copy()
            beat_types[beat_types == df.HolterBeatTypes.MARKED.value] = df.HolterBeatTypes.OTHER.value
            
            if self.event_df.height > 0:
                df_events = (
                    self.event_df
                    .filter(
                            pl.col('type').is_in(event_invalid)
                    )
                    .select(
                            [
                                'start',
                                'stop'
                            ]
                    )
                    .sort(
                            'start'
                    )
                )
                
                if df_events.height > 0:
                    artifact_index = df.get_flattened_index_within_multiple_ranges(
                            nums=self.beat_df['EPOCH'],
                            low=df_events['start'].to_numpy(),
                            high=df_events['stop'].to_numpy()
                    )
                    if len(artifact_index) > 0:
                        beat_types[artifact_index] = df.HolterBeatTypes.OTHER.value
            
            self.beat_df = (
                self.beat_df
                .with_columns(
                        [
                            pl.Series('BEAT_TYPE', beat_types)
                            .alias('BEAT_TYPE')
                        ]
                )
            )

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    def get_valid_region(
            self,
            event_invalid:  List[str] = ('ARTIFACT',),
            beat_valid:     List[int] = None
    ) -> NDArray:
        
        region = np.ones(self.beat_df.height, dtype=int) * self.VALID
        try:
            if beat_valid is None:
                beat_valid = df.VALID_HES_BEAT_TYPE
                
            self.mark_invalid_beat_types(event_invalid=event_invalid)
            
            ind = np.flatnonzero(np.logical_or(
                (~self.beat_df['BEAT_TYPE'].is_in(beat_valid)).to_numpy(),
                np.insert(np.diff(self.beat_df['EPOCH'].to_numpy()), 0, False) > df.THRESHOLD_MIN_PAUSE_RR_INTERVALS
            ))
            region[ind] = self.INVALID
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return region
    
    def get_index_within_start_stop(
            self,
            start_epoch:    int | float,
            stop_epoch:     int | float,
            epoch:          pl.Series | NDArray = None
    ) -> NDArray:
        
        index = np.array([], dtype=int)
        try:
            if epoch is None:
                epoch = self.beat_df['EPOCH']
            
            if len(epoch) == 0:
                return index
                
            index = df.get_index_within_range(
                    nums=epoch,
                    low=start_epoch,
                    high=stop_epoch
            )
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return index
    
    def calculate_duration(
            self,
            index: NDArray
    ) -> float:
        
        duration = 0
        try:
            duration = (self.beat_df['EPOCH'][index].max() - self.beat_df['EPOCH'][index].min()) / df.MILLISECOND
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return duration
    
    @abstractmethod
    def process(self) -> Any:
        pass
