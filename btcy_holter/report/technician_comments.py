from btcy_holter import *


class TechnicianComments(
        pt.Summary
):
    LIMIT:                          Final[int] = 500
    PREDOMINANT_BURDEN:             Final[int] = 50  # %
    PREDOMINANT_BURDEN_RECORD:      Final[int] = 75  # %
    
    LIST_OF_RHYTHM_EVENT_BURDENS:   Final[str] = [
        'AFIB',
        'VT',
        'SVT',
        'TACHY',
        'BRADY',
        'AVB3',
        'AVB2',
        'ARTIFACT'
    ]
    
    ECTOPIC_BEAT_EVENT_DICT: Final[Dict] = {
        df.HolterSymbols.SVE.value: [
            'SVE_COUPLET',
            'SVE_BIGEMINY',
            'SVE_TRIGEMINAL',
            'SVE_QUADRIGEMINY'
        ],
        df.HolterSymbols.VE.value: [
            'VE_COUPLET',
            'VE_BIGEMINY',
            'VE_TRIGEMINAL',
            'VE_QUADRIGEMINY'
        ]
    }
    
    ECTOPIC_RUN_EVENT_DICT: Final[Dict] = {
        df.HolterSymbols.SVE.value: [
            'SVT',
            'SVE_RUN'
        ],
        df.HolterSymbols.VE.value: [
            'VT',
            'VE_RUN'
        ]
    }
    
    def __init__(
            self,
            beat_df:            pl.DataFrame,
            event_df:           pl.DataFrame,
            all_files:          [str, List],
            record_config:      sr.RecordConfigurations,
            **kwargs
    ) -> None:
        try:
            super(TechnicianComments, self).__init__(
                beat_df=beat_df,
                event_df=event_df,
                record_config=record_config,
                **kwargs
            )
            
            self._study_duration:   Final[float] = self._cal_study_duration(all_files=all_files)
            
            self._event_burden_df:  pl.DataFrame = pl.DataFrame()
            self._is_afib_event:    bool = False
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    def _cal_study_duration(
            self,
            all_files: List
    ) -> float:
        
        duration = 0
        try:
            duration = sum(list(map(
                    lambda x: self.convert_timestamp_to_epoch_time(x['stop']) -
                              self.convert_timestamp_to_epoch_time(x['start']),
                    all_files
            )))
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            
        return duration
    
    def _get_burden_of_rhythm_events(
            self
    ) -> pl.DataFrame:
        
        burden_df = pl.DataFrame()
        try:
            if self.event_df.height == 0:
                return burden_df
            
            burdens = list()
            for event_type in self.LIST_OF_RHYTHM_EVENT_BURDENS:
                dur = (
                    self.event_df
                    .filter(
                            pl.col('type') == event_type
                    )
                    .select(
                            [
                                'duration'
                            ]
                    )
                    .sum()
                    .item()
                )
                    
                if dur > 0:
                    burdens.append({
                        'type':     event_type,
                        'burden':   df.round_burden(burden=(dur * df.MILLISECOND) / self._study_duration)
                    })
            
            sinus_burden = self._study_duration - sum(list(map(lambda x: x['burden'], burdens)))
            burdens.append({
                'type':     'SINUS',
                'burden':   df.round_burden(burden=(sinus_burden * df.MILLISECOND) / self._study_duration)
            })
            
            burden_df = pl.DataFrame(burdens)
            burden_df = (
                burden_df
                .filter(
                        ~pl.col('type').is_in(['ARTIFACT'])
                )
            )
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return burden_df
    
    def _generate_rhythm_titles(
            self,
            event_type: str = 'SINUS'
    ) -> str:
        
        text = ''
        try:
            match event_type:
                case 'SINUS':
                    if self.event_df['type'].unique().is_in(['BRADY']).any():
                        text += f'{"" if len(text) == 0 else "/"}Sinus Bradycardia'
                        
                    text += f'{"" if len(text) == 0 else "/"}Sinus Rhythm'
                    
                    if self.event_df['type'].unique().is_in(['TACHY']).any():
                        text += f'{"" if len(text) == 0 else "/"}Sinus Tachycardia'
                
                case _:
                    text = df.HOLTER_ALL_EVENT_TITLES[event_type]
            
            text = f'Predominant Rhythm was {text}'

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return text

    def _check_contains_ivcd_and_avb1(
            self
    ) -> str:
        
        comment = ''
        try:
            is_exists_r = self.beat_df['BEAT_TYPE'].is_in(df.HolterBeatTypes.IVCD.value).any()
            if is_exists_r:
                comment += ' with IVCD'
            
            is_exists_avb1 = self.event_df['type'].unique().is_in(['AVB1']).any()
            if is_exists_avb1:
                comment += f'{", " if is_exists_r else " with "}1st AV Block'

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return comment
    
    def _get_sinus_index(
            self
    ) -> NDArray:
        sinus_index = np.arange(self.beat_df.height)
        try:
            if self.event_df.height > 0:
                events = (
                    self.event_df
                    .filter(
                            pl.col('type').is_in(self.LIST_OF_RHYTHM_EVENT_BURDENS)
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
                if events.height > 0:
                    index_events = df.get_flattened_index_within_multiple_ranges(
                            nums=self.beat_df['EPOCH'].to_numpy(),
                            low=events['start'].to_numpy(),
                            high=events['stop'].to_numpy()
                    )
                    if len(index_events) > 0:
                        sinus_index = np.setdiff1d(sinus_index, index_events)
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return sinus_index
    
    def _get_beat_events_info(
            self,
            comment:        str,
            ectopic:        str,
            ectopic_type:   str
    ) -> str:
        try:
            list_events_ids = self.ECTOPIC_BEAT_EVENT_DICT[ectopic]
            
            sub = list()
            for event_type in list_events_ids:
                ((self.event_df['type'].unique().is_in([event_type]).any())
                 and sub.append(df.HOLTER_ALL_EVENT_TITLES[event_type]))
            
            if len(sub) > 0:
                sub.insert(0, ectopic_type)
                comment += f' including {", ".join(sub)}'
                
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return comment
        
    def _get_run_info(
            self,
            comment:        str,
            ectopic:        str,
            ectopic_type:   str
    ) -> str:
        
        try:
            ectopic_run_df = (
                self.event_df
                .filter(
                        pl.col('type').is_in(self.ECTOPIC_RUN_EVENT_DICT[ectopic])
                )
                .select(
                        [
                            'countBeats',
                            'avgHr'
                        ]
                )
            )
            
            if ectopic_run_df.height > 0:
                max_count_beats = ectopic_run_df['countBeats'].max()
                min_count_beats = ectopic_run_df['countBeats'].min()
                
                if max_count_beats == min_count_beats:
                    count_run = str(max_count_beats)
                else:
                    count_run = f'{min_count_beats}-{max_count_beats}'
                
                title = f'{ectopic_type[:-1]} Run'
                comment += f', and {ectopic_run_df.height} {title} lasting {count_run} beats'
                
                hr = (
                    ectopic_run_df
                    .filter(
                            (pl.col('avgHr') != df.DEFAULT_HR_VALUE)
                    )
                    .select(
                            [
                                'avgHr'
                            ]
                    )
                    .max()
                    .item()
                )
                if hr != df.DEFAULT_HR_VALUE:
                    comment += f' with HR to {hr} bpm'
                    
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
   
        return comment
    
    # @df.timeit
    def _get_predominant_rhythm(
            self
    ) -> Any:
        
        text:               str = ''
        try:
            self._event_burden_df = self._get_burden_of_rhythm_events()
            if self._event_burden_df.height == 0:
                return
            
            self._event_burden_df = (
                self._event_burden_df
                .sort(
                        [
                            'burden'
                        ],
                        descending=True
                )
            )
            
            main = self._event_burden_df[0].rows(named=True)[0]
            self._is_afib_event = main['type'] == 'AFIB'
            
            self._event_burden_df = (
                self._event_burden_df
                .filter(
                        ~pl.col('type').is_in([main['type']])
                )
            )
            
            if main['burden'] > self.PREDOMINANT_BURDEN:
                text = self._generate_rhythm_titles(str(main['type']))
                if main['burden'] > self.PREDOMINANT_BURDEN_RECORD:
                    text += f' ({main["burden"]} % of the recorded time)'
            text += self._check_contains_ivcd_and_avb1()
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        finally:
            text = self._generate_rhythm_titles()
        
        return text
    
    # @df.timeit
    def _get_secondary_arrhythmias(
            self,
    ) -> str:
        
        text = ''
        try:
            if self._event_burden_df.height == 0:
                return text
            
            filter_event_types = (
                self._event_burden_df
                .filter(
                        ~pl.col('type').is_in(['SINUS'])
                )
                .select(
                        [
                            'type'
                        ]
                )
            )
            
            text = ' / '.join(list(map(
                    lambda x: df.HOLTER_ALL_EVENT_TITLES[x],
                    filter_event_types['type']
            )))
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return text

    # @df.timeit
    def _get_ectopic_summary(
            self,
            ectopic: str
    ) -> str:
        
        ectopic_type = 'SVEs' if ectopic == 'S' else 'VEs'
        comment = f'NO {ectopic_type} detected'
        
        try:
            count_beats = self.beat_df['BEAT_TYPE'].is_in([df.SYMBOL_TO_HES[ectopic]]).sum()
            if count_beats == 0:
                return comment
            
            units = 'beats' if count_beats > 1 else 'beat'
            text = f"Total {ectopic_type}:"
            
            burden = df.round_burden(burden=count_beats / self.beat_df.height)
            comment = f'{text} {count_beats} {units} '
            comment += '(<0.01%)' if burden < 0.01 else f'({burden}%)'
            
            comment = self._get_beat_events_info(
                    comment=comment,
                    ectopic=ectopic,
                    ectopic_type=ectopic_type
            )
            
            comment = self._get_run_info(
                    comment=comment,
                    ectopic=ectopic,
                    ectopic_type=ectopic_type
            )
            
            comment = comment + '.'
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return comment
    
    # @df.timeit
    def _get_slowest_heart_rate(
            self,
    ) -> Dict:
    
        intervals = dict()
        intervals['hr']:    int = df.DEFAULT_HR_VALUE
        intervals['pr']:    int = 0
        intervals['qrs']:   int = 0
        intervals['qt']:    int = 0
        
        try:
            if self._is_afib_event:
                return intervals

            if (~self.beat_df['BEAT_TYPE'].is_in(df.VALID_HES_BEAT_TYPE)).all():
                return intervals
            
            # region Get sinus region
            sinus_index = self._get_sinus_index()
            index_valid = np.flatnonzero(self.beat_df['BEAT_TYPE'][sinus_index].is_in(df.VALID_HES_BEAT_TYPE))
            group_sinus_index = df.get_group_from_index_event(sinus_index[index_valid])
            
            group_heart_rate = list()
            for i, index in enumerate(sorted(group_sinus_index, key=lambda x: len(x), reverse=True)):
                if len(index) <= df.LIMIT_BEAT_CALCULATE_HR:
                    continue
                
                if i >= self.LIMIT:
                    break
                
                rrs = df.MILLISECOND_IN_MINUTE / np.diff(self.beat_df['EPOCH'][index].to_numpy())
                heart_rate = ut.calculate_hr_by_geometric_mean(rrs)
                if df.HR_MIN_THR <= heart_rate <= df.HR_MAX_THR:
                    group_heart_rate.append([heart_rate, index])
                    
            if len(group_heart_rate) == 0:
                return intervals
            # endregion Get sinus region

            # region Get slowest heart rate
            for hr, index in sorted(group_heart_rate, key=lambda x: x[0]):
                index = index[np.flatnonzero(self.beat_df['P_ONSET'][index] != 0)]
                if len(index) == 0:
                    continue
                
                hrs = list()
                group = filter(lambda x: len(x) > df.LIMIT_BEAT_CALCULATE_HR, df.get_group_from_index_event(index))
                for i, idx in enumerate(sorted(group, key=lambda x: len(x), reverse=True)):
                    if i >= self.LIMIT:
                        break
                        
                    values = np.column_stack([np.diff(self.beat_df['BEAT'][idx].to_numpy()), idx[1:]])
                    if len(hrs) == 0:
                        hrs = values
                    else:
                        hrs = np.row_stack([hrs, values])
                
                if len(hrs) == 0:
                    continue
                
                index = int(sorted(hrs, key=lambda x: x[0])[0][-1])
                
                pr = (self.beat_df['QRS_ONSET'][index] - self.beat_df['P_ONSET'][index])
                pr = round(int(pr) / self.record_config.sampling_rate, 2)
                pr = min(max(pr, 0), 1.0)
                
                qrs = (self.beat_df['QRS_OFFSET'][index] - self.beat_df['QRS_ONSET'][index])
                qrs = round(int(qrs) / self.record_config.sampling_rate, 2)
                qrs = min(max(qrs, 0), 1.0)
                
                qt = round(int(self.beat_df['QT'][index]) / df.MILLISECOND, 2)
                qt = min(max(qt, 0), 1.0)
                
                intervals['hr']:    int = int(round(hr))
                intervals['pr']:    float = pr
                intervals['qrs']:   float = qrs
                intervals['qt']:    float = qt
                break
            # endregion Get slowest heart rate

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return intervals
    
    # @df.timeit
    def _log(
            self,
            result: Dict
    ) -> None:
        
        try:
            if not bool(result):
                return
            
            tables = PrettyTable(field_names=['Key', 'Value'], align='l')
            tables.sheet_names = ['Technician Comment']
            for key, values in result.items():
                tables.add_row([key, values])
            
            list(map(
                lambda x: self.log_func(message='\t' + x),
                tables.get_string().splitlines()
            ))
        
        except (Exception,) as error:
            cf.DEBUG_MODE and st.write_error_log(error, class_name=self.__class__.__name__)
            pass
    
    @df.timeit
    def process(
            self,
            show_log: bool = False
    ) -> Dict:
        
        result = {
            'predominantRhythm': None,
            'predominantRhythmHr': {
                'min': df.DEFAULT_HR_VALUE,
                'avg': df.DEFAULT_HR_VALUE,
                'max': df.DEFAULT_HR_VALUE
            },
            'secondaryArrhythmias': None,
            'sve': None,
            've': None,
            'slowestSinusHr': {
                'hr':   df.DEFAULT_HR_VALUE,
                'pr':   df.DEFAULT_HR_VALUE,
                'qrs':  df.DEFAULT_HR_VALUE,
                'qt':   df.DEFAULT_HR_VALUE
            }
        }

        try:
            self.mark_invalid_beat_types()
            
            result['predominantRhythm']:    str  = self._get_predominant_rhythm()
            result['secondaryArrhythmias']: str = self._get_secondary_arrhythmias()
            result['sve']:                  str = self._get_ectopic_summary(ectopic=df.HolterSymbols.SVE.value)
            result['ve']:                   str = self._get_ectopic_summary(ectopic=df.HolterSymbols.VE.value)
            result['slowestSinusHr']:       Dict = self._get_slowest_heart_rate()

            show_log and self._log(result)
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return result
