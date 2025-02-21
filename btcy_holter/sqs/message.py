from btcy_holter import *


class MetaData(
    ABC
):
    """
    Parent class for all commands
    """
    
    EVENT_COLS:                         Final[List[str]] = list(df.HOLTER_EVENT_DATAFRAME.keys())
    PVC_DFS:                            Final[Dict]      = df.HOLTER_PVC_DATAFRAME
    
    def __init__(
            self,
            message:                        Dict,
            num_cores:                      int = os.cpu_count(),
    ):
        try:
            self.message:                       Final[Dict] = message
            self.num_cores:                     Final[int]  = num_cores
            
            if 'cmd' in message['Body'].keys():
                self.command: Final[str] = message['Body']['cmd'].upper()
                self.kwargs = dict(
                    addition=self.command,
                    class_name=self.__class__.__name__
                )
            else:
                self.kwargs = dict(
                    class_name=self.__class__.__name__
                )

            if not self._check_valid_message():
                st.get_error_exception('Invalid message', **self.kwargs)

            try:
                cf.GLOBAL_STUDY_ID = int(self.message['Body']['meta']['studyFid'])
            except (Exception,):
                cf.GLOBAL_STUDY_ID = None
            
            self.study_fid:                     Final[str] = str(self.message['Body']['meta']['studyFid'])
            self.study_id:                      Final[str] = self.message['Body']['meta']['studyId']
            self.profile_id:                    Final[str] = self.message['Body']['meta']['profileId']

            self.filename:                      Final[str] = f'study-{self.study_fid}-1'
            self.s3_filename:                   Final[str] = f'{self.profile_id}_true'
            self.s3_data_structure:             Final[df.S3DataStructure] = self.init_s3_data_structures()

            self.work_dir:                      Final[str] = join(cf.RECORD_VOLUME, self.filename, self.profile_id)
            
            prefix = join(self.work_dir, self.filename)
            self.local_parquet_file_path:       Final[str] = prefix + self.s3_data_structure.beat_df_prefix
            self.local_event_parqet_file_path:  Final[str] = prefix + self.s3_data_structure.event_df_prefix
            self.local_pvc_mor_file_path:       Final[str] = prefix + self.s3_data_structure.pvc_df_prefix
            
            self.is_pvc_mor:                    Final[bool] = self.init_flag_for_pvc_morphology_support()

        except (Exception,) as error:
            st.get_error_exception(error, **self.kwargs)
            
    def _check_valid_message(
            self
    ) -> bool:
        """
        Check if the message contains all required fields in the 'meta' section.
        """
        
        try:
            self.message['Body']['meta']['profileId']
            
        except (Exception,):
            st.get_error_exception('Missing: profileId', **self.kwargs)

            return False

        try:
            self.message['Body']['meta']['studyFid']
            
        except (Exception,):
            st.get_error_exception('Missing: studyFid', **self.kwargs)

            return False

        try:
            self.message['Body']['meta']['studyId']
            
        except (Exception,):
            st.get_error_exception('Missing: studyId', **self.kwargs)

            return False

        try:
            if (
                    'cmd' in self.message['Body'].keys()
                    and self.message['Body']['cmd'] not in ['assign-profile', 'migrateVer1ToVer2', 'migrateBeatFinalPath']
                    and self.message['Body']['meta'].get('version', df.DATA_VERSION) != df.DATA_VERSION
            ):
                st.get_error_exception('The old data version.', **self.kwargs)
                
                return False
            
        except (Exception,):
            return False
        
        return True
    
    def init_s3_data_structures(
            self
    ) -> df.S3DataStructure:
        structure = df.S3DataStructure()
        try:
            structure = structure(self.message['Body']['meta'])
            
        except (Exception, ) as error:
            st.write_error_log(error, **self.kwargs)
            
        return structure
    
    def init_df_events(
            self
    ) -> pl.DataFrame:
        return pl.DataFrame(schema={col: pl.Int64 for col in self.EVENT_COLS})
    
    def init_df_pvc(
            self
    ) -> pl.DataFrame:
        return pl.DataFrame(schema={col: pl.Int64 for col in self.PVC_DFS.keys()})
    
    def init_flag_for_pvc_morphology_support(
            self
    ) -> bool:
        
        status = False
        try:
            status = self.message['Body']['meta'].get('isPVCMorphologyEnabled', False)
            if status is None:
                status = False
                
        except (Exception, ) as error:
            st.write_error_log(error, **self.kwargs)
    
        return status
    
    def load_data(
            self
    ) -> [pl.DataFrame]:
        """
        Load the study data from the local parquet file.
        """
        
        dataframe = None
        try:
            dataframe = df.pl_load_dataframe_from_parquet_file(self.local_parquet_file_path)
            
        except (Exception,) as error:
            st.get_error_exception(error, **self.kwargs)

        return dataframe

    def load_event_data(
            self
    ) -> pl.DataFrame:
        """
        Load the event data from the local event parquet file.
        """

        dataframe = None
        try:
            if not df.check_file_exists(self.local_event_parqet_file_path):
                return self.init_df_events()
            
            dataframe = df.pl_load_dataframe_from_parquet_file(self.local_event_parqet_file_path)

        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)

        return dataframe
    
    def load_pvc_morphology_data(
            self
    ) -> pl.DataFrame | None:
        """
        Load the event data from the local event parquet file.
        """

        dataframe = None
        try:
            if not df.check_file_exists(self.local_pvc_mor_file_path):
                if self.is_pvc_mor:
                    return self.init_df_pvc()
                return dataframe
            
            dataframe = df.pl_load_dataframe_from_parquet_file(self.local_pvc_mor_file_path)

        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)

        return dataframe

    def logging(
            self,
            message: str
    ) -> None:
        """
        Log a message with the current command and message ID.
        """
        st.LOGGING_SESSION.info(f'[{self.message.get("date")}] [{self.command}] {message}')

    @abstractmethod
    def process(
            self,
    ) -> Any:
        """
        Abstract method to process the command. Must be implemented by subclasses.
        """
        pass


class SqsMessage(
    MetaData
):
    """
    Initialize the SetCommand with the provided parameters.
    """

    def __init__(
            self,
            message:                        Dict,
            num_cores:                      int = os.cpu_count(),
    ):
        try:
            super(SqsMessage, self).__init__(
                message=message,
                num_cores=num_cores,
            )
            
            self.all_file_path:                 Final[str]          = self._get_all_file_path()
            self.all_files:                     Final[List[Dict]]   = self._get_all_hourly_data()
            self.record_config:                 Final[Any]          = self._get_record_config()

            self.s3_parquet_file_path:          Final[str]          = self._get_s3_data_path()
            self.s3_event_parqet_file_path:     Final[str]          = self._get_s3_event_path()
            self.s3_pvc_mor_parquet_file_path:  Final[str]          = self._get_s3_pvc_morphology_path()

            os.makedirs(self.work_dir, exist_ok=True)

        except (Exception,) as error:
            st.get_error_exception(error, **self.kwargs)
            
    def _get_all_file_path(
            self
    ) -> str:
        path = None
        try:
            path = join(self.work_dir, df.HolterFilenames.ALL_HOURLY_DATA.value)
            
        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)
        
        return path
    
    def _get_s3_data_path(
            self
    ) -> str | None:
        s3_parqet_file_path = join(
            self.s3_data_structure.study_data_dir,
            self.s3_filename + self.s3_data_structure.beat_df_prefix
        )
        try:
            if df.check_file_exists(self.local_parquet_file_path):
                return s3_parqet_file_path

            if hl.check_s3_file_exists(s3_parqet_file_path):
                return s3_parqet_file_path

            s3_npy_path = join(
                    self.s3_data_structure.study_data_dir,
                    self.s3_filename + '.npy'
            )
            
            if hl.check_s3_file_exists(s3_npy_path):
                hl.download_files_by_s5cmd(
                    s3_files=[s3_npy_path],
                    dest_dir=self.work_dir,
                    show_log=False
                )

                local_npy_path = join(self.work_dir, f'{self.s3_filename}.npy')
                if not df.check_file_exists(local_npy_path):
                    return None

                try:
                    pl.read_parquet(local_npy_path)
                    hl.upload_file_to_aws_s3(
                        local_file=local_npy_path,
                        s3_file=s3_parqet_file_path,
                        method='boto3'
                    )

                except (Exception,):
                    return None

        except (Exception, ) as error:
            st.get_error_exception(error, **self.kwargs)

        return s3_parqet_file_path

    def _get_s3_event_path(
            self
    ) -> str:
        s3_event_parqet_file_path = join(
                self.s3_data_structure.study_data_dir,
                self.s3_filename + self.s3_data_structure.event_df_prefix
        )
        try:
            if df.check_file_exists(self.local_event_parqet_file_path):
                return s3_event_parqet_file_path

            s3_events_path = f'{self.study_id}/airp/event/{self.filename}-events.parquet'
            if (
                    hl.check_s3_file_exists(s3_events_path)
                    and not hl.check_s3_file_exists(s3_event_parqet_file_path)
            ):
                
                # region old event data
                local_file = join(self.work_dir, 'tmp.parquet')
                
                hl.download_file_from_aws_s3(
                    s3_file=s3_events_path,
                    local_file=local_file
                )
                
                hl.upload_file_to_aws_s3(
                    local_file=local_file,
                    s3_file=s3_event_parqet_file_path,
                    method='boto3'
                )
                shutil.rmtree(local_file, ignore_errors=True)
                # endregion old event data
                
                # region old event data
                local_file = join(self.work_dir, 'tmp.parquet')
                bk_s3_events_path = s3_events_path.replace('.parquet', '-bk.parquet')
                hl.download_file_from_aws_s3(
                        s3_file=bk_s3_events_path,
                        local_file=local_file
                )
                
                hl.upload_file_to_aws_s3(
                        local_file=local_file,
                        s3_file=bk_s3_events_path,
                        method='boto3',
                )
                shutil.rmtree(local_file, ignore_errors=True)
                # endregion old event data

                st.LOGGING_SESSION.warning(f'--- Uploaded {s3_event_parqet_file_path} to S3.')
                
        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)
        
        return s3_event_parqet_file_path
    
    def _get_s3_pvc_morphology_path(
            self
    ) -> str | None:
        
        s3_pvc_mor_file_path = join(
                self.s3_data_structure.study_data_dir,
                self.s3_filename + self.s3_data_structure.pvc_df_prefix
        )

        try:
            if df.check_file_exists(self.local_pvc_mor_file_path):
                return s3_pvc_mor_file_path
            
            if self.is_pvc_mor:
                return s3_pvc_mor_file_path
            
            if not hl.check_s3_file_exists(s3_pvc_mor_file_path):
                return None
        
        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)
        
        return s3_pvc_mor_file_path
    
    def _get_all_hourly_data(
            self
    ) -> List:
        """
        Get all hourly data files.
        """
        
        all_files = None
        try:
            if df.check_file_exists(self.all_file_path):
                all_files = df.load_all_files(self.all_file_path)

        except (Exception, ) as error:
            st.get_error_exception(error, **self.kwargs)

        return all_files

    def _get_record_config(
            self
    ) -> sr.RecordConfigurations:
        """
        Get the record configurations.
        """

        record_config = df.get_record_configurations_by_meta_data(
                meta_data=self.message['Body']['meta']
        )
        try:
            record_config.record_path = self.work_dir

        except (Exception,) as error:
            st.get_error_exception(error, **self.kwargs)

        return record_config

    def save_data(
            self,
            dataframe: pl.DataFrame
    ) -> None:
        """
        Save the study data to the local parquet file.
        """

        try:
            if cf.DEBUG_MODE:
                file = self.local_parquet_file_path.replace('.parquet', '-updated.parquet')
            else:
                file = self.local_parquet_file_path
            dataframe.write_parquet(file)

        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)

    def upload_data(
            self
    ) -> None:
        """
        Upload the study data to AWS S3.
        """

        try:
            hl.upload_file_to_aws_s3(
                    local_file=self.local_parquet_file_path,
                    s3_file=self.s3_parquet_file_path,
                    method='boto3',
                    show_log=False
            )

        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)

    def save_event_data(
            self,
            dataframe: pl.DataFrame
    ) -> None:
        """
        Save the event data to the local event parquet file.
        """

        try:
            if cf.DEBUG_MODE:
                file = self.local_event_parqet_file_path.replace('.parquet', '-updated.parquet')
            else:
                file = self.local_event_parqet_file_path
                
            if len(dataframe) == 0:
                dataframe = self.init_df_events()
            else:
                dataframe = (
                    dataframe
                    .with_columns(
                        [
                            pl.when(pl.col('start').map_elements(lambda x: isinstance(x, str), return_dtype=pl.Boolean))
                            .then(pl.col('start').map_elements(self.convert_timestamp_to_epoch, return_dtype=pl.Int64))
                            .otherwise(pl.col('start'))
                            .alias('start'),

                            pl.when(pl.col('stop').map_elements(lambda x: isinstance(x, str), return_dtype=pl.Boolean))
                            .then(pl.col('stop').map_elements(self.convert_timestamp_to_epoch, return_dtype=pl.Int64))
                            .otherwise(pl.col('stop'))
                            .alias('stop'),

                            pl.col('isIncludedToReport').replace(None, False)
                        ]
                    )
                    .select(
                            self.EVENT_COLS
                    )
                )
            dataframe.write_parquet(file)

        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)
    
    def upload_event_data(
            self
    ) -> None:
        """
        Upload the event parquet file to AWS S3.
        """
        try:
            hl.upload_file_to_aws_s3(
                    local_file=self.local_event_parqet_file_path,
                    s3_file=self.s3_event_parqet_file_path,
                    method='boto3',
                    show_log=False
            )

        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)
    
    def save_pvc_morphology_data(
            self,
            dataframe: pl.DataFrame
    ) -> None:
        """
        Save the event data to the local event parquet file.
        """

        try:
            if cf.DEBUG_MODE:
                file = self.local_pvc_mor_file_path.replace('.parquet', '-updated.parquet')
            else:
                file = self.local_pvc_mor_file_path
                
            if dataframe.height == 0:
                dataframe = self.init_df_pvc()
                
            dataframe.write_parquet(file)

        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)
    
    def upload_pvc_morphology_data(
            self
    ) -> None:
        """
        Upload the PVC Morphology parquet file to AWS S3.
        """
        try:
            if self.is_pvc_mor is None:
                return
            
            if not df.check_file_exists(self.local_pvc_mor_file_path):
                return
            
            hl.upload_file_to_aws_s3(
                    local_file=self.local_pvc_mor_file_path,
                    s3_file=self.s3_pvc_mor_parquet_file_path,
                    method='boto3',
                    show_log=False
            )
        
        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)
            
    def convert_timestamp_to_epoch(
            self,
            timestamp: str
    ) -> Any:
        """
        Convert a timestamp string to epoch time.
        """

        epoch = 0
        try:
            if not isinstance(timestamp, str):
                return int(timestamp)

            epoch = df.convert_timestamp_to_epoch_time(
                timestamp=timestamp,
                timezone=self.record_config.timezone,
                dtype=int,
                ms=True
            )

        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)

        return epoch

    def convert_epoch_to_timestamp(
            self,
            epoch_time: int | float
    ) -> Any:
        """
        Convert an epoch time to a timestamp string.
        """

        timestamp = None
        try:
            if isinstance(epoch_time, str):
                return epoch_time

            timestamp = df.convert_epoch_time_to_timestamp(
                epoch_time=epoch_time,
                timezone=self.record_config.timezone,
            )

        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)

        return timestamp

    def generate_study_data(
            self,
            dataframe: pl.DataFrame
    ) -> Any:
        """
        Generate study data and column names from the given DataFrame.
        """

        study_data = None
        columns = None
        try:
            study_data = dataframe.to_numpy().copy()
            columns = df.get_study_data_columns(list(dataframe.columns))

        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)

        return study_data, columns

    # @abstractmethod
    def run(self) -> Any:
        pass
    
    @abstractmethod
    def process(self) -> Any:
        pass
