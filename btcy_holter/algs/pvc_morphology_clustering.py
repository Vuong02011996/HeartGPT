from btcy_holter import *


class Configuration:
    PERCENTAGE:     float   = 90 # %
    MAX_CLUSTER:    int     = 10


class PVCMorphologyClustering(
        pt.Summary
):
    BEAT_OFFSET:                        Final[float]    = 0.25             # seconds
    MOR_BEAT_OFFSET:                    Final[float]    = 0.12             # seconds
    BEAT_TYPE:                          Final[int]      = df.HolterBeatTypes.VE.value
    
    CHUNK_SIZE:                         Final[int]      = 5000
    OFFSET:                             Final[int]      = 2                 # %
    MIN_PERCENTAGE:                     Final[int]      = 90                # %
    
    TOTAL_CLUSTER_INCLUDED_TO_REPORT:   Final[int]      = 5
    
    LOWPASS_CUTOFF:                     Final[int]      = 30                # Hz
    HIGHPASS_CUTOFF:                    Final[int]      = 1                 # Hz

    def __init__(
            self,
            data_path:          str,
            beat_df:            pl.DataFrame,
            all_files:          List[Dict],
            record_config:      sr.RecordConfigurations,
            dtype:              str                         = 'float64',
            num_of_processes:   int                         = None,
            review:             bool                        = False,
    ) -> None:
        try:
            super(PVCMorphologyClustering, self).__init__(
                    beat_df=beat_df,
                    all_files=all_files,
                    record_config=record_config,
                    num_of_processes=num_of_processes,
            )
            
            self.data_path:             Final[str]              = df.get_path(data_path) + df.S3DataStructure.pvc_df_prefix
            self._dtype:                Final[str]              = dtype
            self._review:               Final[bool]             = review
            self._config:               Configuration           = Configuration()
            
            self._beat_samples:         Final[int]              = int(self.BEAT_OFFSET * self.record_config.sampling_rate)
            self._mor_beat_samples:     Final[int]              = int(self.MOR_BEAT_OFFSET * self.record_config.sampling_rate)

            self._beat_offset_frames:   Final[NDArray]          = self._get_frames()
            
            self._mor_ids:              NDArray                 = np.zeros(self.beat_df.height, dtype=int)
            self._mor_corrcoef:         NDArray                 = np.zeros(self.beat_df.height, dtype=float)
            self._ids:                  Final[NDArray]          = np.arange(1, self._config.MAX_CLUSTER + 1, 1)
            
            self.__morphology:          pl.DataFrame            = pl.DataFrame([])
            self.__templates:           List[Dict]              = list()
            
        except (Exception, ) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    def _get_frames(
            self
    ) -> NDArray:
        frames = np.array([])
        try:
            frames = np.arange(-self._beat_samples, self._beat_samples + 1)[None, :]
            pass
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return frames
    
    def refactor(
            self,
            ecg_morphology: NDArray
    ) -> NDArray:
        
        corr_ecg_mor = deepcopy(ecg_morphology)
        try:
            offset = self._beat_samples - self._mor_beat_samples
            corr_ecg_mor = ecg_morphology[:, offset: -offset - 1]
    
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return corr_ecg_mor
    
    def calculate(
            self,
            ecg_morphology: NDArray
    ) -> NDArray:
        
        values = np.zeros(len(ecg_morphology))
        try:
            corr_ecg_mor = self.refactor(ecg_morphology)
            values = np.nan_to_num(np.corrcoef(corr_ecg_mor))
            pass
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return values
    
    # @df.timeit
    def __get_ecg_signal(
            self,
            file_index: int
    ) -> Any:
        
        ecg_signals = None
        try:
            data_path = join(
                    dirname(self.record_config.record_path)
                    if isfile(self.record_config.record_path)
                    else self.record_config.record_path,
                    basename(self.all_files[file_index]['path'])
            )
            
            raw_ecg_signals = ut.get_data_from_dat(
                    data_path,
                    record_config=self.record_config
            )
            
            lp_ecg_signals = ut.butter_lowpass_filter(
                    raw_ecg_signals.T,
                    cutoff=self.LOWPASS_CUTOFF,
                    fs=self.record_config.sampling_rate,
            )
            
            ecg_signals = ut.butter_highpass_filter(
                    lp_ecg_signals,
                    cutoff=self.HIGHPASS_CUTOFF,
                    fs=self.record_config.sampling_rate
            )
            
            ecg_signals = ecg_signals.astype(self._dtype)
            pass
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return ecg_signals
    
    def __cluster_with_chunk_size(
            self,
            begin:  int,
    ) -> List[Dict]:
        
        template_clusters = list()
        try:
            ecg_mor_df = (
                self.__morphology
                .slice(
                        offset=begin,
                        length=self.CHUNK_SIZE
                )
            )
            if ecg_mor_df.height == 0:
                return template_clusters
            
            ecg_mor = np.array(ecg_mor_df['MORPHOLOGY'].to_numpy())
            indexes = np.array(ecg_mor_df['INDEX'].to_numpy())
            if ecg_mor_df.height == 1:
                template_clusters.append(
                        {
                            'centerVector': np.nan_to_num(np.mean(ecg_mor, axis=0)),
                            'corrCoeff'   : np.ones(ecg_mor_df.height, dtype=int),
                            'index'       : indexes,
                        }
                )
                return template_clusters
            
            corr_coeff_values = self.calculate(ecg_mor)
            while True:
                if len(ecg_mor) <= 1:
                    corr_values = np.ones(ecg_mor_df.height, dtype=int)
                    index = np.arange(len(indexes))
                else:
                    corr_values = corr_coeff_values[0]
                    if not np.all(corr_values):
                        index = np.arange(len(indexes))
                    else:
                        index = np.flatnonzero(corr_values >= self._config.PERCENTAGE / 100)
                
                template_clusters.append(
                        {
                            'centerVector': np.nan_to_num(np.mean(ecg_mor[index], axis=0)),
                            'corrCoeff'   : corr_values[index],
                            'index'       : indexes[index],
                        }
                )
                
                ecg_mor = np.delete(ecg_mor, index, axis=0)
                indexes = np.delete(indexes, index, axis=0)
                
                corr_coeff_values = np.delete(corr_coeff_values, index, axis=0)
                corr_coeff_values = np.delete(corr_coeff_values, index, axis=1)
                if len(ecg_mor) == 0:
                    break
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return template_clusters
    
    def __cluster_with_center_vector(
            self,
            templates: List[Dict],
    ) -> List[Dict]:
        
        template_clusters = list()
        try:
            if len(templates) == 0:
                return template_clusters
            
            others_index = list()
            if len(templates) > self.CHUNK_SIZE:
                templates = np.array(list(sorted(
                        templates,
                        key=lambda x: len(x['index']),
                        reverse=True
                )))
                others_index = list(chain.from_iterable(map(lambda x: x['index'], templates[self.CHUNK_SIZE:])))
                templates = templates[:self.CHUNK_SIZE]
                
            raw_ecg_mor = np.array(list(map(
                lambda x: x['centerVector'],
                templates
            )))
            raw_corr_coeff_values = self.calculate(raw_ecg_mor)
            
            count = 0
            while True:
                clusters = list()
                indexes = np.arange(len(templates))
                
                ecg_mor             = deepcopy(raw_ecg_mor)
                corr_coeff_values   = deepcopy(raw_corr_coeff_values)
                
                threshold = (self._config.PERCENTAGE - count * self.OFFSET) / 100
                while True:
                    if len(ecg_mor) <= 1:
                        index = np.arange(len(indexes))
                    else:
                        corr_values = corr_coeff_values[0]
                        if not np.all(corr_values):
                            index = np.arange(len(indexes))
                        else:
                            index = np.flatnonzero(corr_values >= threshold)
                    
                    _ = list(chain.from_iterable(map(lambda x: templates[x]['index'], indexes[index])))
                    __ = list(chain.from_iterable(map(lambda x: templates[x]['corrCoeff'], indexes[index])))
                    clusters.append(
                            {
                                'centerVector': np.nan_to_num(np.mean(ecg_mor[index], axis=0)),
                                'corrCoeff'   : __,
                                'index'       : _
                            }
                    )

                    ecg_mor = np.delete(ecg_mor, index, axis=0)
                    indexes = np.delete(indexes, index, axis=0)
                    
                    corr_coeff_values = np.delete(corr_coeff_values, index, axis=0)
                    corr_coeff_values = np.delete(corr_coeff_values, index, axis=1)
                    if len(ecg_mor) == 0:
                        break
                
                count += 1
                if (self._config.PERCENTAGE - count * self.OFFSET) < self.MIN_PERCENTAGE:
                    clusters = sorted(
                            clusters,
                            key=lambda x: len(x['index']),
                            reverse=True
                    )
                    
                    if len(clusters) > self._config.MAX_CLUSTER - 1:
                        clusters = np.array(clusters)
                        others = clusters[self._config.MAX_CLUSTER - 1:]
                        index = list(chain.from_iterable(map(lambda x: x['index'], others))) + others_index
    
                        others = [{
                            'index'         : index,
                            'centerVector'  : np.zeros(clusters[0]['centerVector'].shape),
                            'corrCoeff'     : np.zeros(len(index)),
                            'type'          : -1
                        }]
                        template_clusters = list(clusters[:self._config.MAX_CLUSTER - 1]) + others
                    else:
                        template_clusters = list(clusters)
                    break
                
                if len(clusters) > self._config.MAX_CLUSTER - 1:
                    continue
                
                if len(clusters) <= self._config.MAX_CLUSTER - 1:
                    others = list()
                    if len(others_index) > 0:
                        others = [{
                            'index'       : others_index,
                            'centerVector': np.zeros(clusters[0]['centerVector'].shape),
                            'corrCoeff'   : np.zeros(len(others_index)),
                            'type'        : -1
                        }]
                    
                    template_clusters = clusters + others
                    break
                    
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return template_clusters
    
    @df.timeit
    def _get_ecg_mor(
            self
    ) -> None:
        try:
            
            pvc_df = (
                self.beat_df
                .lazy()
                .with_row_index(
                        'INDEX'
                )
                .filter(
                    pl.col('BEAT_TYPE') == self.BEAT_TYPE
                )
                .select(
                        [
                            'BEAT',
                            'CHANNEL',
                            'FILE_INDEX',
                            'INDEX',
                        ]
                )
                .with_columns(
                        [
                            pl.col('CHANNEL')
                            .map_elements(df.get_channel_from_channel_column)
                            .alias('CHANNEL'),
                        ]
                )
                .collect()
            )
            
            if pvc_df.height == 0:
                return
            
            ecg_mor = list()
            for file_index in pvc_df['FILE_INDEX'].unique():
                dfs = (
                    pvc_df
                    .lazy()
                    .filter(
                        pl.col('FILE_INDEX') == file_index
                    )
                    .select(
                            [
                                'BEAT',
                                'CHANNEL',
                                'INDEX',
                            ]
                    )
                    .sort(
                        'BEAT'
                    )
                    .collect()
                )
                if dfs.height == 0:
                    continue
                
                # region read ecg signals
                ecg_signals = self.__get_ecg_signal(file_index)
                if ecg_signals is None:
                    st.get_error_exception(f'No data found in {file_index}')
                
                # endregion read ecg signals
                
                len_segment = len(ecg_signals[0])
                frame_ind = self._beat_offset_frames + dfs['BEAT'].to_numpy().reshape(-1, 1)
                frame_ind[frame_ind < 0] = 0
                frame_ind[frame_ind >= len_segment] = len_segment - 1
                
                tmp = ecg_signals[dfs['CHANNEL'].to_numpy()[:, None], frame_ind]
                amp = np.max(tmp, axis=1) - np.min(tmp, axis=1)
                
                dfs = (
                    dfs
                    .with_columns(
                            pl.Series('MORPHOLOGY', tmp)
                            .alias('MORPHOLOGY'),
                            
                            pl.Series('AMP', amp)
                            .alias('AMP'),
                    )
                    .select(
                            [
                                'MORPHOLOGY',
                                'AMP',
                                'INDEX',
                            ]
                    )
                )
                ecg_mor.append(dfs)
                pass
            
            self.__morphology = (
                pl.concat(ecg_mor)
                .sort(
                        'AMP',
                        descending=False
                )
                .drop('AMP')
            )
            pass
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    @df.timeit
    def _clustering(
            self
    ) -> None:
        try:
            if self.__morphology.height == 0:
                return 
             
            # region cluster morphology
            templates = list()
            total_beat_raw = self.__morphology.height
            total_groups = (self.__morphology.height // self.CHUNK_SIZE) + 1
            for i in range(total_groups):
                _ = self.__cluster_with_chunk_size(i * self.CHUNK_SIZE)
                templates.extend(_)
            
            templates = sorted(
                    templates,
                    key=lambda x: len(x['index']),
                    reverse=True
            )
            not self._review and self.__morphology.clear()
            if len(templates) == 0:
                return
            
            if len(templates) <= self._config.MAX_CLUSTER - 1:
                self.__templates = templates
                return
            pass
            # endregion cluster morphology
            
            # region cluster center vector
            self.__templates = sorted(
                    self.__cluster_with_center_vector(templates),
                    key=lambda x: [x.get('type', 0), len(x['index'])],
                    reverse=True
            )
            total_beats = sum(list(map(lambda x: len(x['index']), self.__templates)))
            if total_beats != total_beat_raw:
                st.LOGGING_SESSION.debug(f'--- Check beats: {total_beats} != {total_beat_raw}')
            # endregion cluster center vector
            pass
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    @df.timeit
    def _update(
            self
    ) -> None:
        try:
            if len(self.__templates) == 0:
                return
            
            # region update results
            center_vector_mor_df = list()
            for i, template in enumerate(self.__templates):
                template_id = self._ids[i]
                
                index = np.array(template['index'])
                
                self._mor_ids[index] = template_id
                self._mor_corrcoef[index] = np.array(template['corrCoeff'])
                
                unclassified = template.get('type', 0) == -1
                # is_include_to_report = (i < self.TOTAL_CLUSTER_INCLUDED_TO_REPORT) and not unclassified
                
                center_vector_mor_df.append(
                        {
                            'ID':                       template_id,
                            'CENTER_VECTOR':            list(np.nan_to_num(template['centerVector'])),
                            'IS_INCLUDED_TO_REPORT':    False,
                            'UNCLASSIFIED':             unclassified
                        }
                )
            self.__templates = center_vector_mor_df
            self._mor_corrcoef = np.round(self._mor_corrcoef * 100, 4)
            pass
            # endregion update results
            
            # region save morphology
            _ = (
                pl.DataFrame(center_vector_mor_df)
                .write_parquet(
                        self.data_path
                )
            )
            # endregion save morphology
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    def _plot(
            self
    ) -> None:
        try:
            if not self._review:
                return
            
            total_cols = 2
            total_rows = int(np.ceil(len(self.__templates) / 2))
            
            fig, (axis) = plt.subplots(total_rows, total_cols, sharex='all', sharey='all', figsize=(19.2, 10.08))
            fig.subplots_adjust(hspace=0.25, wspace=0.2, left=0.02, right=0.99, bottom=0.03, top=0.95)
            
            for i, template in enumerate(self.__templates):
                if total_rows == 1:
                    ax = axis[i]
                else:
                    row = i // total_cols
                    col = i % total_cols
                    ax = axis[row, col]
                
                index = np.flatnonzero(self._mor_ids == template['ID'])
                if len(index) == 0:
                    continue
                    
                ecg_mor = (
                    self.__morphology
                    .filter(
                            pl.col('INDEX').is_in(index)
                    )
                            [
                                'MORPHOLOGY'
                            ]
                    .to_numpy()
                )
                
                ecg_mor = self.refactor(ecg_mor)
                center_vector = self.refactor(np.array([template['CENTER_VECTOR']]))
                
                # values = np.corrcoef(center_vector, ecg_mor)[0, 1:]
                values = self._mor_corrcoef[index]
                
                # ax_main.plot(ecg_mor[np.argmin(values)].T, color='blue', label='min')
                # ax_main.plot(ecg_mor[np.argmax(values)].T, color='red', label='max')
                # ax.plot(center_vector.T)

                ax.plot(ecg_mor.T)
                ax.plot(center_vector.T, color='k', label='centerVector')
                
                # ax_main.set_title(
                #         f'Template: {template["ID"]}'
                #         f'- values: {np.min(values) * 100:.2f}%/ {np.max(values) * 100:.2f}%'
                # )
                ax.set_title(
                        f'Template: {template["ID"]}'
                        f'- Count: {len(index)} '
                        f'- Include: {template["IS_INCLUDED_TO_REPORT"]}'
                        f'- unclassified: {template["UNCLASSIFIED"]}\n'
                        f'- corrcoef: {np.min(values):.2f}%/ {np.max(values):.2f}%'
                )
                
                i == 0 and ax.set_xlim(0, len(ecg_mor[0]))

            plt.show()
            plt.close()
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    @df.timeit
    def process(
            self
    ) -> [NDArray, NDArray]:

        try:
            if self.beat_df.height == 0:
                return self._mor_ids, self._mor_corrcoef
            
            if not self.beat_df['BEAT_TYPE'].is_in(self.BEAT_TYPE).any():
                st.LOGGING_SESSION.info('PVCMorphologyClustering: No PVCs found.')
                
                _ = (
                    pl.DataFrame(schema={col: pl.Int64 for col in df.HOLTER_PVC_DATAFRAME.keys()})
                    .write_parquet(
                        self.data_path
                    )
                )
                
                return self._mor_ids, self._mor_corrcoef
            
            self._get_ecg_mor()
            self._clustering()
            self._update()
            
            cf.DEBUG_MODE and self._review and self._plot()
            st.LOGGING_SESSION.info(
                    f'PVCMorphologyClustering: {np.count_nonzero(self._mor_ids)} PVCs found'
                    f'-- total templates: {len(self.__templates)} templates'
            )

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return self._mor_ids, (self._mor_corrcoef * 100).astype(int)
