from btcy_holter import *


class BeatFinalPaths(
        sq.SqsMessage
):
    def __init__(
            self,
            message:    Dict,
            num_cores:  int = os.cpu_count(),
            file_index: List = None
    ) -> None:
        try:
            super(BeatFinalPaths, self).__init__(
                    message=message,
                    num_cores=num_cores
            )
            
            self._summary = dict()
            self._summary['summaries'] = list()
            
            self._file_index = file_index

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _download_files(
            self,
    ) -> Any:
        try:
            hl.download_file_from_aws_s3(
                    s3_file=self.message['Body']['meta']['allHourlyDataPath'],
                    local_file=self.all_file_path,
                    method='boto3'
            )
            
            hl.download_file_from_aws_s3(
                    s3_file=self.s3_parquet_file_path,
                    local_file=self.local_parquet_file_path,
                    method='boto3'
            )
        
        except (Exception,) as error:
            st.get_error_exception(error, **self.kwargs)
    
    def _write_final_beats(
            self
    ) -> None:
        
        try:
            saved_path = join(self.work_dir, 'final-beats')
            os.makedirs(saved_path, exist_ok=True)
            
            study_data, col = df.generate_study_data(self.load_data())
            total_file_index = np.sort(np.unique(study_data[:, col.file_index]))
            
            all_hourly_files = df.load_all_files(self.all_file_path)
            if len(all_hourly_files) != len(total_file_index):
                st.get_error_exception('Mismatch in hourly files', **self.kwargs)
            
            if max(total_file_index) >= len(all_hourly_files):
                st.get_error_exception('Mismatch in hourly files', **self.kwargs)
            
            for file_index, hourly_info in zip(total_file_index, all_hourly_files):
                if self._file_index is not None and file_index not in self._file_index:
                    continue
                    
                study_hourly_data = study_data[study_data[:, col.file_index] == file_index]
                filename = df.get_filename(hourly_info['path'])
                
                final_beat_file = join(
                        saved_path,
                        filename + f'-final-{df.generate_id_with_current_epoch_time()}.beat'
                )
                
                pvc_morphology = None
                if self.is_pvc_mor:
                    pvc_morphology = study_hourly_data[:, col.pvc_morphology]
                
                final_beat_file = ut.write_final_beat_file(
                        beat_file=final_beat_file,
                        beat_samples=study_hourly_data[:, col.beat],
                        beat_types=study_hourly_data[:, col.beat_type],
                        beat_channels=df.get_channel_from_channel_column(study_hourly_data[:, col.channel]),
                        rr_heat_maps=study_hourly_data[:, col.rr_heatmap],
                        pvc_morphology=pvc_morphology
                )
                
                self._summary['summaries'].append({
                    'id': hourly_info['id'],
                    'beatFinalPath': join(self.s3_data_structure.beat_final_dir, basename(final_beat_file))
                })
        
        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)
    
    @df.timeit
    def process(
            self
    ) -> Dict:
        try:
            self._download_files()
            if (
                    all(not df.check_file_exists(x)
                    for x in [self.all_file_path, self.local_parquet_file_path])
            ):
                st.get_error_exception('Missing files', **self.kwargs)
            
            self._write_final_beats()
        
        except (Exception,) as error:
            st.write_error_log(error, **self.kwargs)
        
        return self._summary
