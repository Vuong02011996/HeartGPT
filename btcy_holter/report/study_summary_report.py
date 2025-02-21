from btcy_holter import *


class HolterReport(
        pt.Summary
):
    DEFAULT_VALUE:                  Final[str]   = 'NA'
    INVALID_PERCENT:                Final[float] = 0.01  # %
    
    def __init__(
            self,
            beat_df:            pl.DataFrame,
            event_df:           pl.DataFrame,
            all_files:          List[Dict],
            record_config:      sr.RecordConfigurations,
            pvc_morphology_df:  pl.DataFrame = None,
            num_of_processes:   int = os.cpu_count()
    ) -> None:
        try:
            super(HolterReport, self).__init__(
                beat_df=beat_df,
                record_config=record_config,
                event_df=event_df,
                pvc_morphology=pvc_morphology_df,
                all_files=all_files,
                num_of_processes=num_of_processes
            )
            
            self.summary = dict()
            self.summary['start_study']: Final[str] = self.record_config.record_start_time
            self.summary['stop_study']:  Final[str] = self.record_config.record_stop_time
    
            self.start_study:            Final[Any] = self.convert_timestamp_to_epoch_time(self.summary['start_study'])
            self.stop_study:             Final[Any] = self.convert_timestamp_to_epoch_time(self.summary['stop_study'])
            
            self.study_data, self.npy_col = df.generate_study_data(self.beat_df)
            
            self.single_ve_epoch:        NDArray = np.array([])
            self.single_sve_epoch:       NDArray = np.array([])
            
            epoch = self.convert_epoch_time_to_datetime(self.start_study)
            self.start_date = datetime(epoch.year, epoch.month, epoch.day)

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
            
    def _preprocess(
            self
    ) -> None:
        try:
            beat_types = self.beat_df['BEAT_TYPE'].to_numpy().copy()
            
            index = np.flatnonzero(beat_types == df.HolterBeatTypes.MARKED.value)
            artifact_index = df.pl_get_index_events(
                    epochs=self.beat_df['EPOCH'].to_numpy(),
                    dataframe=self.event_df,
                    event_type='ARTIFACT',
            )
            beat_types[np.concatenate((index, artifact_index)).astype(int)] = df.HolterBeatTypes.OTHER.value
            
            self.beat_df = (
                self.beat_df
                .with_columns(
                        [
                            pl.Series('BEAT_TYPE', beat_types).alias('BEAT_TYPE')
                        ]
                )
            )
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    def get_time(
            self,
            offset: int,
            **kwargs
    ) -> NDArray:
        if all(x in kwargs.keys() for x in ['start', 'stop']):
            start = kwargs['start']
            stop = kwargs['stop']
        else:
            start = self.start_study
            stop = self.stop_study

        offset = offset * df.MILLISECOND
        end = stop % offset
        end = stop - end
        end += offset

        if end - start <= offset:
            rs = [start, start + offset]
        else:
            rs = np.arange(start, end, offset)

        if offset == df.SECOND_IN_DAY * df.MILLISECOND and rs[-1] < self.stop_study:
            rs = np.concatenate((np.array(rs), np.array([rs[-1] + offset])))

        return rs
    
    def get_format_datetime(
            self,
            epoch: int
    ) -> Any:

        timestamp = self.convert_epoch_time_to_datetime(epoch)
        timestamp = timestamp.time().strftime('%H:%M:%S')

        return timestamp
    
    # @df.timeit
    def _verify_artifact_report(
            self
    ) -> bool:
        
        is_artifact_report = False
        try:
            count = (self.beat_df['BEAT_TYPE'] == df.HolterBeatTypes.OTHER.value).sum()
            is_artifact_report = ((self.beat_df.height - count) / self.beat_df.height) <= self.INVALID_PERCENT

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return is_artifact_report
        
    @df.timeit
    def process(
            self,
            show_log: bool = False
    ) -> Any:
        
        is_artifact_report = False
        try:
            self._preprocess()
            is_artifact_report = self._verify_artifact_report()
            
            Event(self).process()
            Date(self).process()

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return is_artifact_report, self.summary


class Event:
    HES_DICT = dict()

    HES_DICT['VE'] = dict()
    HES_DICT['VE']['SINGLE']:           Final[int] = df.HOLTER_SINGLE_VES
    HES_DICT['VE']['COUPLET']:          Final[int] = df.HOLTER_VES_COUPLET
    HES_DICT['VE']['BIGEMINY']:         Final[int] = df.HOLTER_VES_BIGEMINY
    HES_DICT['VE']['TRIGEMINAL']:       Final[int] = df.HOLTER_VES_TRIGEMINAL
    HES_DICT['VE']['QUADRIGEMINY']:     Final[int] = df.HOLTER_VES_QUADRIGEMINY
    HES_DICT['VE']['RUN']:              Final[int] = [df.HOLTER_VES_RUN, df.HOLTER_VT]

    HES_DICT['SVE'] = dict()
    HES_DICT['SVE']['SINGLE']:          Final[int] = df.HOLTER_SINGLE_SVES
    HES_DICT['SVE']['COUPLET']:         Final[int] = df.HOLTER_SVES_COUPLET
    HES_DICT['SVE']['BIGEMINY']:        Final[int] = df.HOLTER_SVES_BIGEMINY
    HES_DICT['SVE']['TRIGEMINAL']:      Final[int] = df.HOLTER_SVES_TRIGEMINAL
    HES_DICT['SVE']['QUADRIGEMINY']:    Final[int] = df.HOLTER_SVES_QUADRIGEMINY
    HES_DICT['SVE']['RUN']:             Final[int] = [df.HOLTER_SVES_RUN, df.HOLTER_SVT]

    def __init__(
            self,
            var: HolterReport
    ) -> None:
        self.var = var

    @staticmethod
    def __cal_burden(
            count_beats: int,
            total_beats: int
    ) -> float | str:
        if count_beats > 0:
            burden = df.round_burden(burden=count_beats / total_beats)
            burden = burden if burden >= 0.01 else '<0.01%'
        else:
            burden = 0

        return burden
    
    def _get_time_and_date_index_events(
            self,
            epoch: int | float
    ):
        date_index = None
        time_start = None
        try:
            time_start = self.var.get_format_datetime(epoch)
            timestamp = self.var.convert_epoch_time_to_datetime(epoch)
            timestamp = datetime(timestamp.year, timestamp.month, timestamp.day)
            date_index = (timestamp - self.var.start_date).days + 1
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return date_index, time_start
    
    def __get_longest(
            self,
            event_df: pl.DataFrame
    ) -> str:
        longest_event: str = self.var.DEFAULT_VALUE
        try:
            longest = (
                event_df
                .sort(
                        [
                            'countBeats',
                            'start'
                        ],
                        descending=[
                            True,
                            False
                        ]
                )
                .row(
                        index=0,
                        named=True
                )
            )
            
            (date_index, time_start) = self._get_time_and_date_index_events(longest['start'])
            longest_event = f'{longest["countBeats"]} beats @{time_start} ({date_index})'
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            
        return longest_event
    
    def __get_fastest(
            self,
            event_df: pl.DataFrame
    ) -> str:
        fastest_event:  str = self.var.DEFAULT_VALUE
        try:
            fastest = (
                event_df
                .sort(
                        [
                            'avgHr',
                            'start'
                        ],
                        descending=[
                            True,
                            False
                        ]
                )
                .row(
                        index=0,
                        named=True
                )
            )
            
            (date_index, time_start) = self._get_time_and_date_index_events(fastest['start'])
            fastest_event = f'{fastest["avgHr"]} bpm @{time_start} ({date_index})'
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            
        return fastest_event
    
    # @df.timeit
    def _get_fastest_longest_events(
            self,
            hes_id: list,
    ) -> Any:
        
        total_events:   int = 0
        count_beats:    int = 0
        
        longest_event:  str = self.var.DEFAULT_VALUE
        fastest_event:  str = self.var.DEFAULT_VALUE
        try:
            event_types = list(map(lambda x: df.HOLTER_ALL_EVENT_SUMMARIES_INVERT[x], hes_id))
            run_events_df = (
                self.var.event_df
                .filter(
                        pl.col('type').is_in(event_types)
                )
            )
            if run_events_df.height == 0:
                return total_events, count_beats, fastest_event, longest_event
            
            total_events    = run_events_df.height
            count_beats     = run_events_df['countBeats'].sum()
            
            longest_event = self.__get_longest(run_events_df)
            fastest_event = self.__get_fastest(run_events_df)
            
        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return total_events, count_beats, fastest_event, longest_event

    def _get_start_event(
            self,
            hes_id: int
    ) -> List:
        start_events = list()
        try:
            if self.var.event_df.height == 0:
                return start_events

            if isinstance(hes_id, list):
                event_types = list(map(
                        lambda x: df.HOLTER_ALL_EVENT_SUMMARIES_INVERT[x],
                        hes_id
                ))
                
                dfs = (
                    self.var.event_df
                    .filter(
                            self.var.event_df['type'].is_in(event_types)
                    )
                )
                
            else:
                dfs = (
                    self.var.event_df
                    .filter(
                            self.var.event_df['type'] == df.HOLTER_ALL_EVENT_SUMMARIES_INVERT[hes_id]
                    )
                    
                )
            
            start_events = (
                dfs
                .select(
                        [
                            'start'
                        ]
                )
                .to_numpy()
            )

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return start_events

    def _count_beat_events(
            self,
            hes_id:         int
    ) -> int:

        count = df.DEFAULT_INVALID_VALUE
        try:
            if self.var.event_df.height == 0:
                return count
            
            event_type = df.HOLTER_ALL_EVENT_SUMMARIES_INVERT[hes_id]
            count = (
                self.var.event_df
                .filter(
                        self.var.event_df['type'] == event_type
                )
                .select(
                        [
                            'countBeats'
                        ]
                )
                .sum()
                .item()
            )
            pass

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return count
    
    def _get_event_format(
            self,
            hes_id:     int,
            beat_type:  int = None
    ) -> Any:
        
        text = self.var.DEFAULT_VALUE
        try:
            if self.var.event_df.height == 0:
                return text
            
            count_df = (
                self.var.event_df
                .filter(
                        self.var.event_df['type'] == df.HOLTER_ALL_EVENT_SUMMARIES_INVERT[hes_id]
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
            
            if beat_type is None:
                return count_df.height
            
            index = df.get_flattened_index_within_multiple_ranges(
                    nums=self.var.study_data[:, self.var.npy_col.epoch],
                    low=count_df['start'],
                    high=count_df['stop']
            )
            
            total_ectopic_beats = 0
            if len(index) > 0:
                total_ectopic_beats = np.count_nonzero(self.var.study_data[index, self.var.npy_col.beat_type] == beat_type)

            text = self.generate_fmt(
                    count_df.height,
                    total_beat=total_ectopic_beats
            )
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return text
    
    def generate_fmt(
            self,
            total_event: int,
            total_beat:  int = None
    ) -> Any:
        
        result = self.var.DEFAULT_VALUE
        try:
            if total_event == 0:
                return result
            
            text_event = 'times' if total_event > 1 else 'time'
            result = f'{total_event} {text_event}'
            
            if total_beat is not None:
                text_beat_type = 'beats' if total_beat > 1 else 'beat'
                result += f' ({total_beat} {text_beat_type})'
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return result
    
    def _get_single_event(
            self,
            index:       NDArray,
            hes_dict:    Dict,
            is_sve_beat: bool = False
    ) -> None:
        try:
            list_event_types = list()
            for i in hes_dict.values():
                if isinstance(i, list):
                    list_event_types.extend(list(map(lambda x: df.HOLTER_ALL_EVENT_SUMMARIES_INVERT[x], i)))
                else:
                    list_event_types.append(df.HOLTER_ALL_EVENT_SUMMARIES_INVERT[i])
                    
            list_event_types = list(filter(
                    lambda x: 'single' not in x.lower(),
                    list_event_types
            ))
            
            event_index = df.pl_get_index_events(
                    dataframe=self.var.event_df,
                    event_type=list_event_types,
                    epochs=self.var.study_data[:, self.var.npy_col.epoch]
            )
            idx_single = np.setdiff1d(index, event_index)
            if is_sve_beat:
                self.var.single_sve_epoch = self.var.study_data[idx_single, self.var.npy_col.epoch]
            else:
                self.var.single_ve_epoch = self.var.study_data[idx_single, self.var.npy_col.epoch]
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    def _payload(
            self,
            event_df: pl.DataFrame,
            beat_type: int
    ) -> str | int:
        
        payload = df.DEFAULT_INVALID_VALUE
        try:
            if event_df.height == 0:
                return payload
            
            index = df.get_flattened_index_within_multiple_ranges(
                    nums=self.var.study_data[:, self.var.npy_col.epoch],
                    low=event_df['start'],
                    high=event_df['stop']
            )
            count = np.count_nonzero(self.var.study_data[index, self.var.npy_col.beat_type] == beat_type)
            payload = self.generate_fmt(event_df.height, count)
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            
        return payload
    
    # @df.timeit
    def _ectopic(
            self,
            beat_type: int  # 60 / 70
    ) -> Dict:
        
        summary = dict()
        _ = beat_type == df.HolterBeatTypes.SVE.value
        if _:
            summary['total_sve']:           int = 0
            summary['sve_beats']:           int = 0
            summary['sve_burden']:          int = 0
            summary['sve_percent']:         int = 0
            summary['single_sves']:         int = 0
            summary['single_sve']:          int = 0
            summary['sve_couplet']:         str = self.var.DEFAULT_VALUE
            summary['sve_bigeminy']:        str = self.var.DEFAULT_VALUE
            summary['sve_trigeminal']:      str = self.var.DEFAULT_VALUE
            summary['sve_quadrigeminy']:    str = self.var.DEFAULT_VALUE
            summary['sve_run']:             str = self.var.DEFAULT_VALUE
            summary['sve_run_beats']:       int = 0
            summary['fastest_sve_run']:     str = self.var.DEFAULT_VALUE
            summary['longest_sve_run']:     str = self.var.DEFAULT_VALUE
            
        else:
            summary['total_ves']:           int = 0
            summary['ve_beats']:            int = 0
            summary['ves_burden']:          int = 0
            summary['ve_percent']:          int = 0
            summary['single_ves']:          int = 0
            summary['single_ve']:           int = 0
            summary['ve_couplet']:          str = self.var.DEFAULT_VALUE
            summary['ve_bigeminy']:         str = self.var.DEFAULT_VALUE
            summary['ve_trigeminal']:       str = self.var.DEFAULT_VALUE
            summary['ve_quadrigeminy']:     str = self.var.DEFAULT_VALUE
            summary['ve_run']:              str = self.var.DEFAULT_VALUE
            summary['ve_run_beats']:        int = 0
            summary['fastest_ve_run']:      str = self.var.DEFAULT_VALUE
            summary['longest_ve_run']:      str = self.var.DEFAULT_VALUE
        
        try:
            beat_hes_dict = self.HES_DICT['SVE' if _ else 'VE']
            index = np.flatnonzero(self.var.study_data[:, self.var.npy_col.beat_type] == beat_type)
            burden = self.__cal_burden(len(index), self.var.beat_df.height)
            
            summary['total_sve' if _ else 'total_ves']                = len(index)
            summary['sve_beats' if _ else 've_beats']                 = len(index)

            summary['sve_burden' if _ else 'ves_burden']              = burden
            summary['sve_percent' if _ else 've_percent']             = burden
            
            text = self._get_event_format(beat_hes_dict['COUPLET'], beat_type=beat_type)
            summary['sve_couplet' if _ else 've_couplet']             = text

            text = self._get_event_format(beat_hes_dict['BIGEMINY'], beat_type=beat_type)
            summary['sve_bigeminy' if _ else 've_bigeminy']           = text

            text = self._get_event_format(beat_hes_dict['TRIGEMINAL'], beat_type=beat_type)
            summary['sve_trigeminal' if _ else 've_trigeminal']       = text

            text = self._get_event_format(beat_hes_dict['QUADRIGEMINY'], beat_type=beat_type)
            summary['sve_quadrigeminy' if _ else 've_quadrigeminy']   = text

            total_events, count_beats, fastest, longest = self._get_fastest_longest_events(beat_hes_dict['RUN'])
            summary['sve_run' if _ else 've_run']                     = self.generate_fmt(total_events, count_beats)
            summary['sve_run_beats' if _ else 've_run_beats']         = count_beats
            summary['fastest_sve_run' if _ else 'fastest_ve_run']     = fastest
            summary['longest_sve_run' if _ else 'longest_ve_run']     = longest
            
            self._get_single_event(index, beat_hes_dict, is_sve_beat=_)
            if _:
                count = len(self.var.single_sve_epoch)
            else:
                count = len(self.var.single_ve_epoch)
                
            summary['single_sves' if _ else 'single_ves'] = count
            summary['single_sve' if _ else 'single_ve'] = f"{count} {'beats' if count > 1 else 'beat'}"
            
        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return summary
    
    # @df.timeit
    def _event(
            self
    ) -> Dict:
        summary = dict()
        
        summary['total_afls']:  int = 0
        summary['afl_burden']:  int = 0
        summary['max_afl']:     int = 0
        summary['long_rr']:     int = 0
        
        summary['max_long_rr'] = self.var.DEFAULT_VALUE
        try:
            group = df.pl_get_group_index_events(
                    dataframe=self.var.event_df,
                    event_type='LONG_RR',
                    epochs=self.var.study_data[:, self.var.npy_col.epoch]
            )
            if len(group) == 0:
                return summary
        
            rr_duration = np.array(list(map(
                    lambda x: np.sum(np.diff(self.var.study_data[x, self.var.npy_col.beat])),
                    group
            )))

            index_max = np.argmax(rr_duration)
            max_dur = round(np.max(rr_duration[index_max]) / self.var.record_config.sampling_rate, 2)
            
            summary['max_long_rr'] = dict()
            summary['max_long_rr']['value'] = max_dur
            summary['max_long_rr']['time'] = self.var.convert_epoch_time_to_timestamp(
                self.var.get_value('EPOCH', group[index_max][0]))
            
            summary['long_rr'] = len(group)

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return summary

    # @df.timeit
    def _other(
            self
    ) -> Dict:
        
        summary = dict()
        summary['max_ves_in_hour']:     int = df.DEFAULT_INVALID_VALUE
        summary['max_sves_in_hour']:    int = df.DEFAULT_INVALID_VALUE
        
        summary['max_ves_in_min']:      int = df.DEFAULT_INVALID_VALUE
        summary['max_sves_in_min']:     int = df.DEFAULT_INVALID_VALUE
        
        summary['ront']:                str = self.var.DEFAULT_VALUE
        
        try:
            ve_hour = list()
            sve_hour = list()
            ve_min = list()
            sve_min = list()

            hour_ranges = self.var.get_time(df.SECOND_IN_HOUR)
            hour_ranges = np.row_stack((hour_ranges[:-1], hour_ranges[1:])).transpose()

            group_hourly_index = df.get_group_index_within_multiple_ranges(
                nums=self.var.study_data[:, self.var.npy_col.epoch],
                low=hour_ranges[:, 0],
                high=hour_ranges[:, -1],
                is_filter_index=False
            )
            
            for (start, stop), index in zip(hour_ranges, group_hourly_index):
                if len(index) == 0:
                    continue

                beat_types = self.var.study_data[index, self.var.npy_col.beat_type]
                ve_hour_count = np.count_nonzero(beat_types == df.HolterBeatTypes.VE.value)
                sve_hour_count = np.count_nonzero(beat_types == df.HolterBeatTypes.SVE.value)

                minutes = np.arange(start, stop, df.SECOND_IN_MINUTE * df.MILLISECOND)
                index_minutes = df.get_group_index_within_multiple_ranges(
                    nums=self.var.study_data[index, self.var.npy_col.epoch],
                    low=minutes[:-1],
                    high=minutes[1:]
                )

                max_ve_minute_count = 0
                max_sve_minute_count = 0
                for idx in index_minutes:
                    ve_minutes_count = np.count_nonzero(beat_types[idx] == df.HolterBeatTypes.VE.value)
                    if max_ve_minute_count < ve_minutes_count:
                        max_ve_minute_count = ve_minutes_count

                    sve_minute_count = np.count_nonzero(beat_types[idx] == df.HolterBeatTypes.SVE.value)
                    if max_sve_minute_count < sve_minute_count:
                        max_sve_minute_count = sve_minute_count

                ve_hour.append(ve_hour_count)
                sve_hour.append(sve_hour_count)

                ve_min.append(max_ve_minute_count)
                sve_min.append(max_sve_minute_count)

            ve_min = np.array(ve_min)
            sve_min = np.array(sve_min)
            ve_hour = np.array(ve_hour)
            sve_hour = np.array(sve_hour)

            summary['max_ves_in_hour']:     int = np.max(ve_hour) if len(ve_hour) > 0 else df.DEFAULT_INVALID_VALUE
            summary['max_sves_in_hour']:    int = np.max(sve_hour) if len(sve_hour) > 0 else df.DEFAULT_INVALID_VALUE

            summary['max_ves_in_min']:      int = np.max(ve_min) if len(ve_min) > 0 else df.DEFAULT_INVALID_VALUE
            summary['max_sves_in_min']:     int = np.max(sve_min) if len(sve_min) > 0 else df.DEFAULT_INVALID_VALUE

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return summary

    # @df.timeit
    def _hrv(
            self
    ) -> Dict:
        
        summary: Dict = dict()
        summary['sdnn']:                int = 0
        summary['sdann']:               int = 0
        summary['rmssd']:               int = 0
        summary['pnn50']:               int = 0
        summary['tp']:                  int = 0
        summary['vlf']:                 int = 0
        summary['lf']:                  int = 0
        summary['hf']:                  int = 0
        summary['lf_norm']:             int = 0
        summary['hf_norm']:             int = 0
        summary['lf_hf']:               int = 0
        
        summary['max_rr']:              int = 0
        summary['avg_rr']:              int = 0
        summary['min_rr']:              int = 0
        
        summary['msd']:                 int = 0
        summary['cv']:                  int = 0
        
        summary['total_beats']:         int = 0
        summary['beats_use_hrv']:       int = 0
        summary['triangle_index']:      int = 0
        summary['hrv_index']:           int = 0
        
        try:
            rr = np.diff(self.var.study_data[:, self.var.npy_col.beat]) / self.var.record_config.sampling_rate
            valid_index = np.flatnonzero(np.logical_and.reduce((
                rr <= df.SECOND_IN_MINUTE / df.HR_MIN_THR,
                rr >= df.SECOND_IN_MINUTE / df.HR_MAX_THR,
                self.var.study_data[1:, self.var.npy_col.beat_type] != df.HolterBeatTypes.OTHER.value
            )))
            
            if len(valid_index) == 0:
                return summary
            
            summary['total_beats'] = len(valid_index) - 1
            
            rr = rr[valid_index]
            rr_beat_type = self.var.study_data[1:, self.var.npy_col.beat_type][valid_index]
            time_m = self.var.study_data[0, self.var.npy_col.beat] / self.var.record_config.sampling_rate
            time_m = (np.hstack((time_m, rr)).cumsum()[1:]).astype(np.float32)
            rr = np.asarray(np.multiply(rr, df.MILLISECOND), dtype='int32')

            hrv = cl.HRVariability()
            hrv, nn_time, nns = cl.hrv_time_domain(hrv, time_m, rr, rr_beat_type)

            summary['sdnn']     = hrv.sdnn if not math.isnan(hrv.sdnn) else 0
            summary['sdann']    = hrv.sdann if not math.isnan(hrv.sdann) else 0
            summary['rmssd']    = hrv.rmssd if not math.isnan(hrv.rmssd) else 0
            summary['pnn50']    = hrv.pnn50 if not math.isnan(hrv.pnn50) else 0
            summary['msd']      = hrv.msd if not math.isnan(hrv.msd) else 0
            del time_m, rr_beat_type
            
            if len(rr) > 0:
                summary['beats_use_hrv']:   int = len(nns)
                summary['max_rr']:          int = int(round(np.max(nns), 2))
                summary['min_rr']:          int = int(round(np.min(nns), 2))
                summary['avg_rr']:          float = float(np.round(np.mean(nns), 2))
          
            if len(nn_time) > 1:
                if len(nns) > df.LIMIT_BEAT_TO_CALCULATE_HRV:
                    nn_time = nn_time[:df.LIMIT_BEAT_TO_CALCULATE_HRV]
                    nns = nns[:df.LIMIT_BEAT_TO_CALCULATE_HRV]
                
                (summary['ulf'], summary['vlf'], summary['lf'], summary['hf'],
                 summary['tp'], summary['lf_hf']) = cl.hrv_freq_domain(nn_time, nns)
                
                summary['lf_norm'] = summary['lf'] / (summary['tp'] - summary['vlf'] + np.finfo(float).eps)
                summary['lf_norm'] = df.round_burden(summary['lf_norm'])
                
                summary['hf_norm'] = summary['hf'] / (summary['tp'] - summary['vlf'] + np.finfo(float).eps)
                summary['hf_norm'] = df.round_burden(summary['hf_norm'])
                
                summary['cv'] = round(summary['sdnn'] / summary['avg_rr'], 2)

            cal_hrv = np.asarray(np.divide(df.SECOND_IN_MINUTE, rr), dtype=int)
            max_number_hr = np.count_nonzero(cal_hrv == np.argmax(np.bincount(cal_hrv)))
            
            text = str(summary["beats_use_hrv"])
            
            burden = self.__cal_burden(summary['beats_use_hrv'], len(rr))
            if isinstance(burden, str):
                text += f' ({burden})'
            elif burden == 0:
                pass
            else:
                text += f' ({burden}%)'
                
            summary['beats_use_hrv']:   str = text
            summary['triangle_index']:  float = float(np.round(len(rr) / max_number_hr, 1))
            summary['hrv_index']:       float = float(np.round(summary['triangle_index'] / 2, 1))

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return summary
    
    # @df.timeit
    def _beats_burden(
            self
    ) -> Dict:
        
        summary = dict()
        summary['total_beats'] = self.var.beat_df.height
        
        summary['sve_beats']:   int = 0
        summary['sve']:         int = 0
        
        summary['ve_beats']:    int = 0
        summary['ve']:          int = 0
        
        summary['afib']:        int = 0
        summary['brady']:       int = 0
        summary['tachy']:       int = 0
        summary['pause']:       int = 0
        summary['ve']:          int = 0
        summary['others']:      int = 0
        
        try:
            sve_ind = np.flatnonzero(self.var.study_data[:, self.var.npy_col.beat_type] == df.HolterBeatTypes.SVE.value)
            summary['sve_beats'] = len(sve_ind)
            summary['sve'] = self.__cal_burden(summary['sve_beats'], summary['total_beats'])
            
            ve_ind = np.flatnonzero(self.var.study_data[:, self.var.npy_col.beat_type] == df.HolterBeatTypes.VE.value)
            summary['ve_beats'] = len(ve_ind)
            summary['ve'] = self.__cal_burden(summary['ve_beats'], summary['total_beats'])
            
            summary['afib'] = self.__cal_burden(
                    self._count_beat_events(df.HOLTER_AFIB),
                    summary['total_beats']
            )
            
            summary['brady'] = self.__cal_burden(
                    self._count_beat_events(df.HOLTER_BRADY),
                    summary['total_beats']
            )
            
            summary['tachy'] = self.__cal_burden(
                    self._count_beat_events(df.HOLTER_TACHY),
                    summary['total_beats']
            )
            
            summary['pause'] = self.__cal_burden(
                    self._count_beat_events(df.HOLTER_PAUSE),
                    summary['total_beats']
            )
            
            event_index = df.pl_get_index_events(
                    dataframe=self.var.event_df,
                    event_type=['AFIB', 'BRADY', 'TACHY', 'PAUSE'],
                    epochs=self.var.study_data[:, self.var.npy_col.epoch]
            )
            if event_index is None:
                event_index = np.array([], dtype=int)
            
            index = np.unique(np.concatenate((sve_ind, ve_ind, event_index)).flatten())
            summary['others'] = self.__cal_burden(summary['total_beats'] - len(index), summary['total_beats'])

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return summary
    
    # @df.timeit
    def _ectopic_burden(
            self,
            beat_type: int
    ) -> Dict:

        _ = beat_type == df.HolterBeatTypes.SVE.value

        summary = dict()
        
        ectopic = dict()
        if _:
            ectopic['sve_beats']:           int = 0
            ectopic['sve_burden']:          int = 0
            ectopic['single_sves']:         int = 0
            ectopic['single_sve']:          int = 0
            ectopic['sve_couplet']:         str = self.var.DEFAULT_VALUE
            ectopic['sve_bigeminy']:        str = self.var.DEFAULT_VALUE
            ectopic['sve_trigeminal']:      str = self.var.DEFAULT_VALUE
            ectopic['sve_quadrigeminy']:    str = self.var.DEFAULT_VALUE
            ectopic['sve_run']:             str = self.var.DEFAULT_VALUE
            ectopic['sve_run_beats']:       str = self.var.DEFAULT_VALUE
            ectopic['fastest_sve_run']:     str = self.var.DEFAULT_VALUE
            ectopic['longest_sve_run']:     str = self.var.DEFAULT_VALUE
            summary['total_sve_analysis']:  Dict = ectopic
            summary['daily_sve_analysis']:  Dict = list()

        else:
            ectopic['ve_beats']:            int = 0
            ectopic['ve_burden']:           int = 0
            ectopic['single_ves']:          int = 0
            ectopic['single_ve']:           int = 0
            ectopic['ve_couplet']:          str = self.var.DEFAULT_VALUE
            ectopic['ve_bigeminy']:         str = self.var.DEFAULT_VALUE
            ectopic['ve_trigeminal']:       str = self.var.DEFAULT_VALUE
            ectopic['ve_quadrigeminy']:     str = self.var.DEFAULT_VALUE
            ectopic['ve_run']:              str = self.var.DEFAULT_VALUE
            ectopic['ve_run_beats']:        str = self.var.DEFAULT_VALUE
            ectopic['fastest_ve_run']:      str = self.var.DEFAULT_VALUE
            ectopic['longest_ve_run']:      str = self.var.DEFAULT_VALUE
            summary['total_ve_analysis']:   Dict = ectopic
            summary['daily_ve_analysis']:   Dict = list()
        
        try:
            # region total analysis
            ectopic = dict()
            hes_dict = self.HES_DICT['SVE' if _ else 'VE']
            
            count_ectopic = np.count_nonzero(self.var.study_data[:, self.var.npy_col.beat_type] == beat_type)
            ectopic['sve_beats' if _ else 've_beats'] = count_ectopic
            
            burden = self.__cal_burden(count_ectopic, self.var.beat_df.height)
            ectopic['sve_burden' if _ else 've_burden'] = burden

            ectopic['sve_couplet' if _ else 've_couplet'] = self._get_event_format(hes_dict['COUPLET'])
            ectopic['sve_bigeminy' if _ else 've_bigeminy'] = self._get_event_format(hes_dict['BIGEMINY'])
            ectopic['sve_trigeminal' if _ else 've_trigeminal'] = self._get_event_format(hes_dict['TRIGEMINAL'])
            ectopic['sve_quadrigeminy' if _ else 've_quadrigeminy'] = self._get_event_format(hes_dict['QUADRIGEMINY'])

            total_events, count_beats, fastest_event, longest_event = self._get_fastest_longest_events(hes_dict['RUN'])
            ectopic['sve_run' if _ else 've_run']                 = total_events
            ectopic['sve_run_beats' if _ else 've_run_beats']     = count_beats
            ectopic['fastest_sve_run' if _ else 'fastest_ve_run'] = fastest_event
            ectopic['longest_sve_run' if _ else 'longest_ve_run'] = longest_event
            
            count = len(self.var.single_sve_epoch if _ else self.var.single_ve_epoch)
            ectopic['single_sves' if _ else 'single_ves'] = count
            ectopic['single_sve' if _ else 'single_ve']   = f"{count} {'beats' if count > 1 else 'beat'}"
            summary['total_sve_analysis' if _ else 'total_ve_analysis'] = ectopic
            # endregion total analysis

            # region daily analysis
            date_start = self.var.convert_epoch_time_to_datetime(self.var.start_study)
            date_start = date_start.replace(hour=0, minute=0, second=0)
            epoch_date_start = self.var.convert_timestamp_to_epoch_time(str(date_start))
            
            date_end = self.var.convert_epoch_time_to_datetime(self.var.stop_study)
            date_end = date_end.replace(hour=23, minute=59, second=59)
            epoch_date_end = self.var.convert_timestamp_to_epoch_time(str(date_end))

            day_offset = df.SECOND_IN_DAY * df.MILLISECOND
            hour_offset = df.SECOND_IN_HOUR * df.MILLISECOND

            daily_ectopic_analysis = list()
            date_ranges = np.arange(epoch_date_start, epoch_date_end + day_offset + 1, day_offset)
            date_ranges = np.column_stack((date_ranges[:-1], date_ranges[1:]))

            start_run_events            = self._get_start_event(hes_dict['RUN'])
            start_couplet_events        = self._get_start_event(hes_dict['COUPLET'])
            start_bigeminy_events       = self._get_start_event(hes_dict['BIGEMINY'])
            start_trigeminal_events     = self._get_start_event(hes_dict['TRIGEMINAL'])
            start_quadrigeminy_events   = self._get_start_event(hes_dict['QUADRIGEMINY'])
            for (start, stop) in date_ranges:
                day = dict()
                day['start_time'] = self.var.convert_epoch_time_to_timestamp(start)

                ind = self.var.get_index_within_start_stop(
                        start_epoch=start,
                        stop_epoch=stop
                )
                day['total_beats'] = len(ind)

                for comp, epoch_events in [
                    ('couplet', start_couplet_events),
                    ('run', start_run_events),
                    ('bigeminy', start_bigeminy_events),
                    ('trigeminal', start_trigeminal_events),
                    ('quadrigeminy', start_quadrigeminy_events),
                ]:
                    if len(epoch_events) == 0:
                        count = 0
                    else:
                        count = np.count_nonzero(np.logical_and(epoch_events >= start, epoch_events <= stop))
                    day[f'sve_{comp}' if _ else f've_{comp}'] = count

                count = np.count_nonzero(self.var.study_data[ind, self.var.npy_col.beat_type] == beat_type)
                day['sve_beats' if _ else 've_beats'] = count

                day['hourly_burden_histogram'] = list()
                if len(ind) == 0:
                    day['sve_burden' if _ else 've_burden'] = 0
                    day['single_sves' if _ else 'single_ves'] = 0
                    day['hourly_burden_histogram'] = np.zeros(df.HOUR_IN_DAY, dtype=int).tolist()

                else:
                    count_single = self.var.get_index_within_start_stop(
                            epoch=self.var.single_sve_epoch if _ else self.var.single_ve_epoch,
                            start_epoch=start,
                            stop_epoch=stop
                    )
                    
                    day['single_sves' if _ else 'single_ves'] = len(count_single)

                    day['sve_burden' if _ else 've_burden'] = self.__cal_burden(count, len(ind))
                    hour_ranges = np.arange(start, stop + hour_offset, hour_offset)
                    hour_ranges = np.column_stack((hour_ranges[:-1], hour_ranges[1:]))
                    for (begin, end) in hour_ranges:
                        index = self.var.get_index_within_start_stop(
                                epoch=self.var.study_data[ind, self.var.npy_col.epoch],
                                start_epoch=begin,
                                stop_epoch=end
                        )
                        if len(index) == 0:
                            hour_burden = 0
                        else:
                            count = np.count_nonzero(
                                self.var.study_data[ind[index], self.var.npy_col.beat_type] == beat_type)
                            hour_burden = self.__cal_burden(count, len(index)) if count > 0 else 0
                        day['hourly_burden_histogram'].append(hour_burden)

                daily_ectopic_analysis.append(day)
            summary['daily_sve_analysis' if _ else 'daily_ve_analysis'] = daily_ectopic_analysis
            # endregion daily analysis

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return summary
        
    # @df.timeit
    def _pvc_morphology_summary(
            self,
    ) -> Dict:
        
        summary = dict()
        try:
            summary['totalBeats'] = np.count_nonzero(
                    self.var.study_data[:, self.var.npy_col.beat_type] == df.HolterBeatTypes.VE.value
            )
            summary['histogram'] = list()
            if self.var.pvc_mor_df.height == 0:
                return summary
            
            mor_df = (
                self.var.pvc_mor_df
                .filter(
                        [
                            pl.col('IS_INCLUDED_TO_REPORT')
                        ]
                )
            )
            if mor_df.height == 0:
                return summary
            
            date_start = self.var.convert_epoch_time_to_datetime(self.var.start_study)
            date_start = date_start.replace(hour=0, minute=0, second=0)
            epoch_date_start = self.var.convert_timestamp_to_epoch_time(str(date_start))
            
            date_end = self.var.convert_epoch_time_to_datetime(self.var.stop_study)
            date_end = date_end.replace(hour=23, minute=59, second=59)
            epoch_date_end = self.var.convert_timestamp_to_epoch_time(str(date_end))
            
            day_offset = df.SECOND_IN_DAY * df.MILLISECOND
            hour_offset = df.SECOND_IN_HOUR * df.MILLISECOND
            
            date_ranges = np.arange(epoch_date_start, epoch_date_end + day_offset + 1, day_offset)
            date_ranges = np.column_stack((date_ranges[:-1], date_ranges[1:]))
            
            for mor in mor_df.to_dicts():
                histogram = dict()
                histogram['id']             = mor['ID']
                histogram['centerVector']   = mor['CENTER_VECTOR']
                
                template_df = (
                    self.var.beat_df
                    .filter(
                            pl.col('PVC_TEMPLATE') == histogram['id']
                    )
                    .select(
                            'EPOCH'
                    )
                )
                histogram['totalBeats']     = template_df.height
                histogram['burden']         = self.__cal_burden(histogram['totalBeats'], summary['totalBeats'])
                
                histogram['date'] = list()
                
                epochs_in_template = template_df['EPOCH'].to_numpy()
                for (start, stop) in date_ranges:
                    hour_ranges = np.arange(start, stop + hour_offset, hour_offset)
                    hour_ranges = np.column_stack((hour_ranges[:-1], hour_ranges[1:]))
                    
                    burden = list()
                    burden_ve_beats = list()
                    for (hour_start, hour_stop) in hour_ranges:
                        total_beat_in_template_in_hour = np.count_nonzero(np.logical_and(
                                epochs_in_template >= hour_start,
                                epochs_in_template <= hour_stop
                        ))
                        
                        index = np.flatnonzero(np.logical_and(
                                self.var.study_data[:, self.var.npy_col.epoch] >= hour_start,
                                self.var.study_data[:, self.var.npy_col.epoch] <= hour_stop,
                        ))
                        total_beat_in_hour = len(index)
                        
                        total_v_beat_in_hour = np.count_nonzero(
                                self.var.study_data[index, self.var.npy_col.beat_type] == df.HolterBeatTypes.VE.value
                        )
                        
                        burden.append(self.__cal_burden(total_beat_in_template_in_hour, total_beat_in_hour))
                        burden_ve_beats.append(self.__cal_burden(total_beat_in_template_in_hour, total_v_beat_in_hour))
                    
                    histogram['date'].append({
                        'time':         self.var.convert_epoch_time_to_timestamp(start),
                        'burden':       burden,
                        'burdenVbeats': burden_ve_beats
                    })
                    
                summary['histogram'].append(histogram)
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return summary
        
    # @df.timeit
    def _hr(
            self
    ) -> Dict:
        
        summary = dict()
        summary['beats']: int = self.var.beat_df.height
        try:
            pass
            
        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return summary
    
    # @df.timeit
    def _afib(
            self
    ) -> Dict:
        
        summary = dict()
        summary['hr_under_60']:     int = 0     # HR < 60 bpm
        summary['hr_under_100']:    int = 0     # 60 bpm <= HR < 100 bpm
        summary['hr_under_140']:    int = 0     # 100 bpm <= HR < 140 bpm
        summary['hr_over_140']:     int = 0     # HR >= 140 bpm
        try:
            group = df.pl_get_group_index_events(
                    dataframe=self.var.event_df,
                    event_type='AFIB',
                    epochs=self.var.study_data[:, self.var.npy_col.epoch]
            )
            if len(group) == 0:
                return summary
            
            for ind in group:
                cons = df.remove_channel_from_channel_column(self.var.study_data[ind, self.var.npy_col.channel])
                group_index = np.split(ind, np.flatnonzero(np.diff(cons) != 0) + 1)
                
                hr = np.array([])
                rr = np.array([])
                for index in group_index:
                    frame_index = np.column_stack((index[:-1], index[1:]))
                    tmp_hr = np.array(list(map(
                        lambda x: ut.calculate_hr_by_geometric_mean(
                            df.MILLISECOND_IN_MINUTE / np.diff(self.var.study_data[x, self.var.npy_col.epoch])
                        ),
                        frame_index
                    )))
                    
                    tmp_rr = np.diff(self.var.study_data[:, self.var.npy_col.epoch][frame_index], axis=1)
                    tmp_rr = tmp_rr.flatten() / df.MILLISECOND
                    
                    hr = np.append(hr, tmp_hr)
                    rr = np.append(rr, tmp_rr)

                ind = df.get_ind_hr_valid(hr)
                hr = hr[ind]
                rr = rr[ind]
                if len(hr) == 0:
                    continue
                
                value = round(np.sum(rr[np.flatnonzero(hr < 60)]))
                summary['hr_under_60'] += value
                
                value = round(np.sum(rr[np.flatnonzero(np.logical_and(hr >= 60, hr < 100))]))
                summary['hr_under_100'] += value
                
                value = round(np.sum(rr[np.flatnonzero(np.logical_and(hr >= 100, hr < 140))]))
                summary['hr_under_140'] += value
                
                value = round(np.sum(rr[np.flatnonzero(hr >= 140)]))
                summary['hr_over_140'] += value
                pass
            pass
        
        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return summary
    
    # @df.timeit
    def _daily_afib(
            self
    ) -> List:
        
        summary = list()
        try:
            if self.var.event_df.height == 0:
                return summary
            
            date_start = self.var.convert_epoch_time_to_datetime(self.var.start_study)
            date_start = date_start.replace(hour=0, minute=0, second=0)
            date_start = self.var.convert_timestamp_to_epoch_time(str(date_start))
            
            date_end = self.var.convert_epoch_time_to_datetime(self.var.stop_study)
            date_end = date_end.replace(hour=23, minute=59, second=59)
            date_end = self.var.convert_timestamp_to_epoch_time(str(date_end))
            
            day_offset = df.SECOND_IN_DAY * df.MILLISECOND
            
            date_ranges = np.arange(date_start, date_end + day_offset + 1, day_offset)
            date_ranges = np.column_stack((date_ranges[:-1], date_ranges[1:]))
            
            afib_event_dfs = (
                self.var.event_df
                .filter(
                        pl.col('type') == 'AFIB'
                )
                .select(
                        [
                            'start',
                            'stop'
                        ]
                )
                .sort(
                        [
                            'start'
                        ]
                )
            )
            
            for (start, stop) in date_ranges:
                day = dict()
                day['start_time'] = self.var.convert_epoch_time_to_timestamp(start)
                
                day['hr_under_60']      = 0
                day['hr_under_100']     = 0
                day['hr_under_140']     = 0
                day['hr_over_140']      = 0
                day['longest_episode'] = {
                    'value': df.DEFAULT_INVALID_VALUE,
                    'time': self.var.DEFAULT_VALUE
                }
                
                day['heart_rate']: Dict = {
                    'average': df.DEFAULT_HR_VALUE,
                    'slowest': {
                        'value': df.DEFAULT_HR_VALUE
                    },
                    'fastest': {
                        'value': df.DEFAULT_HR_VALUE
                    }
                }
                
                if afib_event_dfs.height == 0:
                    summary.append(day)
                    continue
                
                index_epoch = df.get_index_within_range(
                        nums=self.var.study_data[:, self.var.npy_col.epoch],
                        low=start,
                        high=stop
                )
                if len(index_epoch) == 0:
                    summary.append(day)
                    continue
                
                # region calculate 1RR
                _ = df.get_flattened_index_within_multiple_ranges(
                    nums=self.var.study_data[index_epoch, self.var.npy_col.epoch],
                    low=afib_event_dfs['start'],
                    high=afib_event_dfs['stop'],
                )
                if len(_) == 0:
                    summary.append(day)
                    continue
                    
                index_epoch = index_epoch[_]
                cons = df.remove_channel_from_channel_column(self.var.study_data[index_epoch, self.var.npy_col.channel])
                
                group_index = np.split(
                        index_epoch, 
                        np.flatnonzero(np.logical_or(np.diff(index_epoch) != 1, np.diff(cons) != 0)) + 1
                )
                group_index = list(filter(lambda x: len(x) > 0, group_index))
                
                hr = np.array([])
                rr = np.array([])
                afib_events = list()
                for index in group_index:
                    frame_index = np.column_stack((index[:-1], index[1:]))
                    tmp_hr = np.array(list(map(
                        lambda x: ut.calculate_hr_by_geometric_mean(
                            df.MILLISECOND_IN_MINUTE / np.diff(self.var.study_data[x, self.var.npy_col.epoch])
                        ),
                        frame_index
                    )))
                    
                    tmp_rr = np.diff(self.var.study_data[:, self.var.npy_col.epoch][frame_index], axis=1)
                    tmp_rr = tmp_rr.flatten() / df.MILLISECOND
                    
                    hr = np.append(hr, tmp_hr)
                    rr = np.append(rr, tmp_rr)
                    
                    tmp_event = dict()
                    tmp_event['start'] = self.var.study_data[index[0], self.var.npy_col.epoch]
                    tmp_event['stop'] = self.var.study_data[index[-1], self.var.npy_col.epoch]
                    
                    tmp_event['duration'] = (tmp_event['stop'] - tmp_event['start']) / df.MILLISECOND
                    tmp_event['countBeats'] = len(index)
                    
                    heart_rate = ut.HeartRate(
                            beats=self.var.study_data[index, self.var.npy_col.beat],
                            symbols=self.var.study_data[index, self.var.npy_col.beat_type],
                            sampling_rate=self.var.record_config.sampling_rate
                    ).process()
                    
                    tmp_event['maxHr'] = heart_rate['maxHr']
                    tmp_event['minHr'] = heart_rate['minHr']
                    tmp_event['avgHr'] = heart_rate['avgHr']
                    afib_events.append(tmp_event)
                    
                if len(hr) > 0:
                    ind = df.get_ind_hr_valid(hr)
                    hr = hr[ind]
                    rr = rr[ind]
                    if len(hr) == 0:
                        continue
                    
                    value = round(np.sum(rr[np.flatnonzero(hr < 60)]))
                    day['hr_under_60'] += value
                    
                    value = round(np.sum(rr[np.flatnonzero(np.logical_and(hr >= 60, hr < 100))]))
                    day['hr_under_100'] += value
                    
                    value = round(np.sum(rr[np.flatnonzero(np.logical_and(hr >= 100, hr < 140))]))
                    day['hr_under_140'] += value
                    
                    value = round(np.sum(rr[np.flatnonzero(hr >= 140)]))
                    day['hr_over_140'] += value
                    pass
                
                if len(afib_events) > 0:
                    event_dfs = pl.DataFrame(afib_events)
                    
                    longest = (
                        event_dfs
                        .sort(
                                'duration',
                                descending=True
                        )
                        .row(
                                index=0,
                                named=True
                        )
                    )
                    
                    duration = int(round(longest['duration'])) \
                        if longest['duration'] > df.SECOND_IN_MINUTE \
                        else round(longest['duration'], 1)
                    
                    day['longest_episode'] = {
                        'time' : self.var.convert_epoch_time_to_timestamp(longest['start']),
                        'value': duration
                    }
                    
                    hr_dfs = (
                        event_dfs
                        .filter(
                                pl.col('avgHr') != df.DEFAULT_HR_VALUE
                        )
                    )
                    if hr_dfs.height > 0:
                        min_hr, max_hr = (
                            hr_dfs
                            .sort(
                                    'avgHr',
                                    descending=False
                            )[[0, -1]]
                            .to_dicts()
                        )
                        
                        day['heart_rate'] = {
                            'average': ut.calculate_hr_by_geometric_mean(hr_dfs['avgHr'].to_numpy()),
                            'slowest': {
                                'value': min_hr['avgHr']
                            },
                            'fastest': {
                                'value': max_hr['avgHr']
                            }
                        }
                summary.append(day)
            pass
        
        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return summary
    
    def process(
            self
    ) -> None:
        try:
            self.var.summary['ves_summary']:                    Dict = self._ectopic(beat_type=df.HolterBeatTypes.VE.value)
            self.var.summary['sves_summary']:                   Dict = self._ectopic(beat_type=df.HolterBeatTypes.SVE.value)
            self.var.summary['event_summary']:                  Dict = self._event()

            self.var.summary['other_summary']:                  Dict = self._other()
            self.var.summary['hrv_summary']:                    Dict = self._hrv()

            self.var.summary['beats_burden_summary']:           Dict = self._beats_burden()
            self.var.summary['sve_burden_summary']:             Dict = self._ectopic_burden(df.HolterBeatTypes.SVE.value)
            self.var.summary['ve_burden_summary']:              Dict = self._ectopic_burden(df.HolterBeatTypes.VE.value)
            self.var.summary['hr_summary']:                     Dict = self._hr()
    
            self.var.summary['afib']:                           Dict = dict()
            self.var.summary['afib']['total_afib_analysis']:    Dict = self._afib()
            self.var.summary['afib']['daily_afib_analysis']:    Dict = self._daily_afib()
            
            if self.var.pvc_mor_df is not None:
                self.var.summary['pvc_morphology_summary']:     Dict = self._pvc_morphology_summary()
        
        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)


class Date:
    def __init__(
            self,
            var:    HolterReport
    ) -> None:
        
        try:
            self.var = var

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    def __cal_qtc_histograms(
            self,
            qts:            NDArray,
            qtc_range:      int = 300     # ms
    ) -> Dict:

        qtc_range_histogram = dict()
        qtc_range_histogram['titles'] = list()
        qtc_range_histogram['values'] = list()

        try:
            for index in range(0, df.SECOND_IN_MINUTE):
                count = np.count_nonzero(np.logical_and(
                        qts >= qtc_range + index * 5,
                        qts < qtc_range + (index + 1) * 5
                ))
                qtc_range_histogram['titles'].append(qtc_range + index * 5)
                qtc_range_histogram['values'].append(count)
                
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return qtc_range_histogram

    def __cal_qtc_variability(
            self,
            qt_min:     NDArray,
            qtc_min:    NDArray
    ) -> Dict:

        qtc_variability = dict()
        qtc_variability['range']    = ['<300MS', '<350MS', '<400MS', '<450MS', '<500MS', '<550MS', '>=550', 'MAX']

        qtc_variability['qts']      = [0] * len(qtc_variability['range'])
        qtc_variability['qtcs']     = [0] * len(qtc_variability['range'])
        try:
            qt = deepcopy(qt_min)
            qt = qt[qt > 0]
            
            qtc = deepcopy(qtc_min)
            qtc = qtc[qtc > 0]
            
            qt_values = list()
            qtc_values = list()
            
            for index in range(0, len(qtc_variability['range']) - 1):
                if index == len(qtc_variability['range']) - 2:
                    stop_range = np.inf
                else:
                    stop_range = index * 50

                if index == 0:
                    start_range = -300
                    stop_range = index * 50
                else:
                    start_range = (index - 1) * 50
                
                if len(qt) > 0:
                    num_qts = np.count_nonzero(
                            np.logical_and(
                                    qt >= (300 + start_range),
                                    qt < (300 + stop_range)
                            )
                    )
                    qt_values.append(df.round_burden(num_qts / len(qt)))
                else:
                    qt_values.append(0)
                
                if len(qtc) > 0:
                    num_qtc = np.count_nonzero(
                            np.logical_and(
                                    qtc >= (300 + start_range),
                                    qtc < (300 + stop_range)
                            )
                    )
                    qtc_values.append(df.round_burden(num_qtc / len(qtc)))
                else:
                    qtc_values.append(0)
            
            # Max
            if len(qt) > 0:
                qtc_variability['qts']:     List = qt_values + [str(np.max(qt))]
            else:
                qtc_variability['qts']:     List = qt_values + [str(0)]
                
            if len(qtc) > 0:
                qtc_variability['qtcs']:    List = qtc_values + [str(np.max(qtc))]
            else:
                qtc_variability['qtcs']:    List = qtc_values + [str(0)]
            
        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return qtc_variability

    def __count_ectopic(
            self,
            index:          NDArray,
            is_sve_beat:    bool
    ) -> int:
        value = 0
        try:
            if len(index) == 0:
                return value

            epoch = self.var.single_sve_epoch if is_sve_beat else self.var.single_ve_epoch
            value = np.count_nonzero(np.isin(self.var.study_data[index, self.var.npy_col.epoch], epoch))

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return value
    
    def __hrv(
            self,
            index: NDArray
    ) -> Dict:
        
        hrv_dict = dict()
        hrv_dict['sdnn']:       int = 0
        hrv_dict['pnn50']:      int = 0
        hrv_dict['sdann']:      int = 0
        hrv_dict['rmssd']:      int = 0
        hrv_dict['sdnn_index']: int = 0
        hrv_dict['msd']:        int = 0
        hrv_dict['mean_rr']:    int = 0
        
        try:
            if len(index) == 0:
                return hrv_dict
            
            if np.all(self.var.study_data[index, self.var.npy_col.beat_type] == df.HolterBeatTypes.OTHER.value):
                return hrv_dict
                
            hrv = cl.calculate_hrv(
                    x=self.var.study_data[index, self.var.npy_col.epoch],
                    beat_types=self.var.study_data[index, self.var.npy_col.beat_type],
                    sampling_rate=self.var.record_config.sampling_rate,
                    is_epoch_time=True
            )

            hrv_dict['sdnn']:       int = hrv.sdnn if not math.isnan(hrv.sdnn) else 0
            hrv_dict['sdann']:      int = hrv.sdann if not math.isnan(hrv.sdann) else 0
            hrv_dict['rmssd']:      int = hrv.rmssd if not math.isnan(hrv.rmssd) else 0
            hrv_dict['pnn50']:      int = hrv.pnn50 if not math.isnan(hrv.pnn50) else 0
            hrv_dict['sdnn_index']: int = hrv.sdnnindx if not math.isnan(hrv.sdnnindx) else 0
            hrv_dict['msd']:        int = hrv.msd if not math.isnan(hrv.msd) else 0
            hrv_dict['mean_rr']:    int = int(hrv.mean_rr) if not math.isnan(hrv.mean_rr) else 0

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return hrv_dict
    
    def __pqst(
            self,
            index: NDArray
    ) -> Dict:
        
        data = dict()
        data['st_slope']:   int = 0
        data['st_level']:   int = 0
        data['qtc']:        int = 0
        data['qt']:         int = 0

        try:
            if len(index) == 0:
                return data
                                                
            data['st_slope'] = np.mean(np.divide(self.var.study_data[index, self.var.npy_col.st_slope], df.MILLISECOND))
            data['st_level'] = np.mean(np.divide(self.var.study_data[index, self.var.npy_col.st_level], df.MILLISECOND))
            
            qtc = self.var.study_data[index, self.var.npy_col.qtc]
            qtc = qtc[qtc > 0]
            if len(qtc) > 0:
                data['qtc'] = np.mean(qtc)
            
            qt = self.var.study_data[index, self.var.npy_col.qt]
            qt = qt[qt > 0]
            if len(qt) > 0:
                data['qt'] = np.mean(qt)

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return data

    def __hr_qt_qtc_t_wave(
            self,
            start_hour:     int,
            index:          NDArray,
            date_index:     int
    ) -> Dict:

        hour = dict()
        
        hour['time']:       int = f'{self.var.get_format_datetime(int(start_hour))} ({date_index + 1})'
        hour['max_qt']:     int = 0
        hour['min_qt']:     int = 0
        hour['avg_qt']:     int = 0

        hour['max_qtc']:    int = 0
        hour['min_qtc']:    int = 0
        hour['avg_qtc']:    int = 0

        hour['max_t']:      int = 0
        hour['min_t']:      int = 0
        hour['avg_t']:      int = 0

        hour['beats']:      int = 0
        try:
            if len(index) == 0:
                return hour

            qtc_hour    = self.var.study_data[index, self.var.npy_col.qtc]
            qtc_hour    = qtc_hour[qtc_hour > 0]

            qt_hour     = self.var.study_data[index, self.var.npy_col.qt]
            qt_hour     = qt_hour[qt_hour > 0]
            
            t_peak_hour = np.divide(self.var.study_data[index, self.var.npy_col.t_amp], df.VOLT_TO_MV)
            t_peak_hour = t_peak_hour[t_peak_hour != 0]

            if len(qt_hour) > 0:
                hour['max_qt']:     int = int(np.round((np.max(qt_hour)), 2))
                hour['min_qt']:     int = int(np.round((np.min(qt_hour)), 2))
                hour['avg_qt']:     int = int(np.round((np.mean(qt_hour)), 2))
            
            if len(qtc_hour) > 0:
                hour['max_qtc']:    int = int(np.round((np.max(qtc_hour)), 2))
                hour['min_qtc']:    int = int(np.round((np.min(qtc_hour)), 2))
                hour['avg_qtc']:    int = int(np.round((np.mean(qtc_hour)), 2))
            
            if len(t_peak_hour) > 0:
                hour['max_t']:      float = round(np.round((np.max(t_peak_hour)), 2), 2)
                hour['min_t']:      float = round(np.round((np.min(t_peak_hour)), 2), 2)
                hour['avg_t']:      float = round(np.round((np.mean(t_peak_hour)), 2), 2)

            hour['beats']:      int = len(index)

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return hour
    
    def _ectopic_single_by_minutes(
            self,
            index:          List | NDArray,
            is_sve_beat:    bool
    ) -> Dict:

        data = dict()
        if is_sve_beat:
            data['sves_report'] = dict()
            data['sves_report']['sve_single_array'] = list()
        else:
            data['ves_report'] = dict()
            data['ves_report']['ve_single_array'] = list()

        try:
            minutes = list(map(
                self.__count_ectopic,
                index,
                [is_sve_beat] * len(index)
            ))

            if is_sve_beat:
                data['sves_report']['sve_single_array'] = minutes
            else:
                data['ves_report']['ve_single_array'] = minutes

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return data

    # @df.timeit
    def _hrv_by_minutes(
            self,
            index: List
    ) -> Dict:

        hrv_dict = dict()
        hrv_dict['sdnn_array']:         NDArray = list()
        hrv_dict['rmssd_array']:        NDArray = list()
        hrv_dict['pnn50_array']:        NDArray = list()
        hrv_dict['mean_rr_array']:      NDArray = list()
        hrv_dict['sdnn_index_array']:   NDArray = list()

        try:
            hrv_df: pl.DataFrame = pl.DataFrame(list(map(self.__hrv, index)))
            if hrv_df.height == 0:
                return hrv_dict
            
            hrv_dict['sdnn_array']:         NDArray = hrv_df['sdnn'].to_numpy().astype(int)
            hrv_dict['rmssd_array']:        NDArray = hrv_df['rmssd'].to_numpy().astype(int)
            hrv_dict['pnn50_array']:        NDArray = hrv_df['pnn50'].to_numpy().astype(int)
            hrv_dict['mean_rr_array']:      NDArray = hrv_df['mean_rr'].to_numpy().astype(int)
            hrv_dict['sdnn_index_array']:   NDArray = hrv_df['sdnn_index'].to_numpy().astype(int)

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return hrv_dict

    # @df.timeit
    def _pqst_by_minutes(
            self,
            minutes: List
    ) -> Dict:
        
        data = dict()
        data['st_slope_array']:             Dict = dict()
        data['st_slope_array']['name']:     str = 'ST_SLOPE'
        data['st_slope_array']['value']:    List = list()
        
        data['st_level_array']:             Dict = dict()
        data['st_level_array']['name']:     str = 'ST_LEVEL'
        data['st_level_array']['value']:    List = list()

        data['qt_array']:                   NDArray = np.array([])
        data['qtc_array']:                  NDArray = np.array([])

        data['qtc_variability']:            List = list()
        data['qtc_range_histogram']:        List = list()

        try:
            pt_df: pl.DataFrame = pl.DataFrame(list(map(self.__pqst, minutes)))
            if pt_df.height == 0:
                return data

            data['st_slope_array']['value']:    NDArray = np.round(pt_df['st_slope'].to_numpy().astype(float), 4)
            data['st_level_array']['value']:    NDArray = np.round(pt_df['st_level'].to_numpy().astype(float), 4)

            data['qt_array']:                   NDArray = pt_df['qt'].to_numpy().astype(int)
            data['qtc_array']:                  NDArray = pt_df['qtc'].to_numpy().astype(int)
            
            data['qtc_variability']:            List = self.__cal_qtc_variability(data['qt_array'], data['qtc_array'])
            data['qtc_range_histogram']:        List = self.__cal_qtc_histograms(data['qtc_array'])

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return data

    # @df.timeit
    def _hrv_by_hour(
            self,
            index: List
    ) -> Dict:
        
        hrv_dict = dict()
        hrv_dict['hrv_report']:                                     Dict = dict()
        hrv_dict['hrv_report']['hour_info_array']:                  Dict = dict()
        hrv_dict['hrv_report']['hour_info_array']['sdnn']:          NDArray = list()
        hrv_dict['hrv_report']['hour_info_array']['sdann']:         NDArray = list()
        hrv_dict['hrv_report']['hour_info_array']['rmssd']:         NDArray = list()
        hrv_dict['hrv_report']['hour_info_array']['pnn50']:         NDArray = list()
        hrv_dict['hrv_report']['hour_info_array']['sdnn_index']:    NDArray = list()
        hrv_dict['hrv_report']['hour_info_array']['mean_rr']:       NDArray = list()
        try:
            hrv_df = pl.DataFrame(list(map(self.__hrv, index)))
            if hrv_df.height == 0:
                return hrv_dict

            hrv_dict['hrv_report'] = dict()
            hrv_dict['hrv_report']['hour_info_array'] = dict()
            hrv_dict['hrv_report']['hour_info_array']['sdnn']       = hrv_df['sdnn'].to_numpy().astype(float)
            hrv_dict['hrv_report']['hour_info_array']['sdann']      = hrv_df['sdann'].to_numpy().astype(float)
            hrv_dict['hrv_report']['hour_info_array']['rmssd']      = hrv_df['rmssd'].to_numpy().astype(float)
            hrv_dict['hrv_report']['hour_info_array']['pnn50']      = hrv_df['pnn50'].to_numpy().astype(float)
            hrv_dict['hrv_report']['hour_info_array']['sdnn_index'] = hrv_df['sdnn_index'].to_numpy().astype(float)
            hrv_dict['hrv_report']['hour_info_array']['mean_rr']    = hrv_df['mean_rr'].to_numpy().astype(int)

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return hrv_dict
        
    # @df.timeit
    def _hr_qt_qtc_t_wave_by_date(
            self,
            date_index: int,
            date_times: List
    ) -> Dict:

        data = dict()
        data['hr_qt_qtc_t_array'] = list()
        try:
            start, stop = date_times
            start = start - (start % (df.SECOND_IN_HOUR * df.MILLISECOND))

            epoch_hours = self.var.get_time(offset=df.SECOND_IN_HOUR, start=start, stop=stop)
            epoch_hours = np.concatenate((epoch_hours, [date_times[-1]]))

            epoch_hours[0] = date_times[0]
            epoch_hours = epoch_hours[epoch_hours <= self.var.stop_study]
            if len(epoch_hours) == 0:
                return data
            
            group_hour_index = df.get_group_index_within_multiple_ranges(
                nums=self.var.study_data[:, self.var.npy_col.epoch],
                low=epoch_hours[:-1],
                high=epoch_hours[1:],
                is_filter_index=False
            )
            for epoch, index in zip(epoch_hours, group_hour_index):
                tmp = self.__hr_qt_qtc_t_wave(
                    start_hour=epoch,
                    index=index,
                    date_index=date_index
                )
                data['hr_qt_qtc_t_array'].append(tmp)

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return data
    
    def _minutes(
            self,
            epoch_start:    int,
            epoch_stop:     int
    ) -> Dict:

        results = dict()
        try:
            epoch_minutes = self.var.get_time(
                offset=df.SECOND_IN_MINUTE,
                start=epoch_start,
                stop=epoch_stop
            )

            group_minutes_index = df.get_group_index_within_multiple_ranges(
                nums=self.var.study_data[:, self.var.npy_col.epoch],
                low=epoch_minutes[:-1],
                high=epoch_minutes[1:],
                is_filter_index=False
            )

            # region VE
            ve_date = self._ectopic_single_by_minutes(
                index=group_minutes_index,
                is_sve_beat=False
            )
            results.update(ve_date)
            # endregion VE

            # region SVE
            sve_date = self._ectopic_single_by_minutes(
                index=group_minutes_index,
                is_sve_beat=True
            )
            results.update(sve_date)
            # endregion SVE

            # region HRV
            hrv_min = self._hrv_by_minutes(group_minutes_index)
            results.update(hrv_min)
            # endregion HRV

            # region PQST
            pt_date = self._pqst_by_minutes(group_minutes_index)
            results.update(pt_date)
            # endregion PQST

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
        return results
    
    def _hour(
            self,
            epoch_start:    int,
            epoch_stop:     int
    ) -> Dict:

        results = dict()
        try:
            epoch_hours = self.var.get_time(
                offset=df.SECOND_IN_HOUR,
                start=epoch_start,
                stop=epoch_stop,
            )

            group_hour_index = df.get_group_index_within_multiple_ranges(
                nums=self.var.study_data[:, self.var.npy_col.epoch],
                low=epoch_hours[:-1],
                high=epoch_hours[1:],
                is_filter_index=False
            )

            hrv_hour = self._hrv_by_hour(group_hour_index)
            results.update(hrv_hour)

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return results

    # @df.timeit
    def _date(
            self,
            date_index: int,
            date_times: NDArray | List
    ) -> Dict:

        results = dict()
        try:
            date_start, date_stop = date_times
            results['start_time']   = self.var.convert_epoch_time_to_timestamp(date_start)
            results['end_time']     = self.var.convert_epoch_time_to_timestamp(date_stop)
            
            minutes:    Dict = self._minutes(date_start, date_stop)
            results.update(minutes)

            hours:      Dict = self._hour(date_start, date_stop)
            results.update(hours)

            hrs:        Dict = self._hr_qt_qtc_t_wave_by_date(date_index, date_times)
            results.update(hrs)

        except (Exception, ) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return results

    # @df.timeit
    def process(
            self
    ) -> None:
        try:
            date_ranges = self.var.get_time(offset=df.SECOND_IN_DAY)
            date_ranges = np.row_stack((date_ranges[:-1], date_ranges[1:])).transpose()

            self.var.summary['date'] = [
                self._date(date_index=index, date_times=date)
                for index, date in enumerate(date_ranges)
            ]

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
