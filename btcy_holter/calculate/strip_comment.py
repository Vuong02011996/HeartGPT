from btcy_holter import *


class StripComment:
    LIST_EVENT_NOT_CALCULATE_HR: Final[List[str]] = [
        'SINGLE',
        'BIGEMINY',
        'TRIGEMINAL',
        'QUADRIGEMINY',
        'COUPLET'
    ]
    
    DEFINE_EVENT_COUNT_THRESHOLD: Final[Dict] = {
        'VE_RUN'          : 3,
        'VE_BIGEMINY'     : 6 - 1,
        'VE_TRIGEMINAL'   : 9 - 2,
        'VE_QUADRIGEMINY' : 12 - 3,
        'VE_COUPLET'      : 2,
        'SINGLE_VE'       : 1,
        
        'SVE_RUN'         : 3,
        'SVE_BIGEMINY'    : 6 - 1,
        'SVE_TRIGEMINAL'  : 9 - 2,
        'SVE_QUADRIGEMINY': 12 - 3,
        'SVE_COUPLET'     : 2,
        'SINGLE_SVE'      : 1,
    }

    RHYTHM_EVENTS: Final[List] = ['PAUSE', 'AFIB', 'TACHY', 'BRADY', 'AVB2', 'AVB3', 'VT', 'SVT', 'SINUS']

    # @df.timeit
    def __init__(
            self,
            beat_df:        Union[pl.DataFrame, Tuple[NDArray, Any]],
            event_type:     str,
            record_config:  sr.RecordConfigurations
    ) -> None:
        
        try:
            if isinstance(beat_df, pl.DataFrame):
                self._study_data, self._npy_col = df.generate_study_data(beat_df)
            else:
                self._study_data, self._npy_col = beat_df

            self.event_type:    Final[str]                      = event_type
            self.record_config: Final[sr.RecordConfigurations]  = record_config
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    def _add_sv_into_comment(
            self,
    ) -> str:
        
        text = ''
        try:
            list_sv_events = list(df.HOLTER_BEAT_EVENT.items())
            if self.event_type == 'SVT':
                list_sv_events = list(filter(lambda x: 'SVE' in x[0].split('_'), list_sv_events))
            elif self.event_type == 'VT':
                list_sv_events = list(filter(lambda x: 'VE' in x[0].split('_'), list_sv_events))
            
            subs = list()
            for event_type, hes_id in list_sv_events:
                ind = df.get_index_event(self._study_data[:, self._npy_col.event], hes_id)
                if len(ind) < self.DEFINE_EVENT_COUNT_THRESHOLD[event_type]:
                    continue
                
                temp = deepcopy(df.HOLTER_ALL_EVENT_TITLES[event_type])
                if len(ind) > 1 and 'SINGLE' in event_type:
                    temp += 's'
                subs.append(temp)
            
            if len(subs) > 0:
                text = ', '.join(list(set(subs)))
        
        except Exception as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return text
    
    # @df.timeit
    def _process_strip(
            self,
    ) -> Dict:
        
        comment = None
        try:
            comment = df.HOLTER_ALL_EVENT_TITLES[self.event_type]
            match self.event_type:
                case _ if any(x in self.event_type for x in self.LIST_EVENT_NOT_CALCULATE_HR):
                    pass
                
                case 'ARTIFACT':
                    dur = (self._study_data[-1, self._npy_col.epoch] - self._study_data[0, self._npy_col.epoch])
                    dur = dur / df.MILLISECOND
                    comment += f' {round(float(dur), 2)}s'
                
                case 'PAUSE':
                    if len(self._study_data) > 1:
                        dur = (self._study_data[-1, self._npy_col.epoch] - self._study_data[0, self._npy_col.epoch])
                        dur = dur / df.MILLISECOND
                        comment += f' {round(float(dur), 1)}s'
                
                case _:
                    heart_rate = ut.HeartRate(
                            beats=self._study_data[:, self._npy_col.beat],
                            symbols=self._study_data[:, self._npy_col.beat_type],
                            sampling_rate=self.record_config.sampling_rate
                    ).process()
                    
                    if heart_rate['avgHr'] == df.DEFAULT_HR_VALUE:
                        comment += f' with {len(self._study_data)} beat'
                        if len(self._study_data) > 1:
                            comment += 's'
                    else:
                        comment += f' at {heart_rate["avgHr"]} bpm'
                        if self.event_type not in df.HOLTER_BEAT_EVENT.keys():
                            sv_events = self._add_sv_into_comment()
                            if len(sv_events) > 0:
                                comment += f' with {sv_events}'
            
            if len(comment) > 0 and comment[-1] != '.':
                comment = comment + '.'
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return comment
    
    def _process_event(
            self
    ) -> str:
        
        comment = ''
        try:
            study_duration = (self._study_data[-1, self._npy_col.epoch] - self._study_data[0, self._npy_col.epoch])
            study_duration = study_duration / df.MILLISECOND

            # region Get burden of rhythm events
            rhythm_events = list()
            ignore_event_types = list()
            for event_type in self.RHYTHM_EVENTS:
                if event_type in ignore_event_types:
                    continue
                
                if event_type == 'SINUS' and len(rhythm_events) == 0:
                    group_ind = [np.arange(len(self._study_data))]
                else:
                    group_ind = df.get_group_index_event(self._study_data[:, self._npy_col.event], event_type)

                if len(group_ind) == 0:
                    continue

                duration = list(map(
                    lambda x: (self._study_data[x[-1], self._npy_col.epoch] -
                               self._study_data[x[0], self._npy_col.epoch]),
                    group_ind
                ))

                burden = 0
                if study_duration > 0:
                    burden = df.round_burden(float((sum(duration) / df.MILLISECOND) / study_duration))

                cmt = df.HOLTER_ALL_EVENT_TITLES[event_type]
                match event_type:
                    case 'PAUSE':
                        ind_max = np.argmax(duration)
                        cmt += f' {round(float(duration[ind_max]), 1)}s'
                        ind_pause = group_ind[ind_max]
                        
                        text = ''
                        for _ in self.RHYTHM_EVENTS:
                            if _ == event_type:
                                continue
                                
                            if df.check_exists_event(
                                    self._study_data[int(ind_pause[0]), self._npy_col.event],
                                    df.HOLTER_ALL_EVENT_SUMMARIES[_]
                            ):
                                text = df.HOLTER_ALL_EVENT_TITLES[_]
                                heart_rate = ut.HeartRate(
                                        beats=self._study_data[:, self._npy_col.beat],
                                        symbols=self._study_data[:, self._npy_col.beat_type],
                                        sampling_rate=self.record_config.sampling_rate
                                ).process()
                                
                                if heart_rate['avgHr'] != df.DEFAULT_HR_VALUE:
                                    text += f' at {heart_rate["avgHr"]} bpm'
                                break
                                
                        if len(text) > 0:
                            cmt += f'{text} and {cmt}'

                    case _:
                        heart_rate = ut.HeartRate(
                            beats=self._study_data[:, self._npy_col.beat],
                            symbols=self._study_data[:, self._npy_col.beat_type],
                            sampling_rate=self.record_config.sampling_rate
                        ).process()

                        if heart_rate['avgHr'] != df.DEFAULT_HR_VALUE:
                            cmt += f' at {heart_rate["avgHr"]} bpm'
                            sv_events = self._add_sv_into_comment()
                            if len(sv_events) > 0:
                                cmt += f' with {sv_events}'

                rhythm_events.append([event_type, burden, cmt])
            # endregion Get burden of rhythm events
            
            rhythm_events = sorted(rhythm_events, key=lambda x: x[1], reverse=True)
            comment = ', '.join(list(map(lambda x: x[-1], rhythm_events)))
            if len(comment) > 0 and comment[-1] != '.':
                comment = comment + '.'
            pass

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return comment

    # @df.timeit
    def process(
            self
    ) -> str:
        
        comment = ''
        try:
            if self.event_type is None:
                comment = self._process_event()
            else:
                comment = self._process_strip()
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return comment
