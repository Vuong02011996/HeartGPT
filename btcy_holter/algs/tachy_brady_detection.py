from btcy_holter import *

BRADY: Final[int] = 0x0002
TACHY: Final[int] = 0x0004


class Config:
    pattern:        str
    beat_id:        int
    event_id:       int
    
    def tachy(self):
        self.pattern:       str = '4{5}4*((2|0){0,4}4+)*4*'
        self.beat_id:       int = TACHY
        self.event_id:      int = 0x0040
        
        return self
        
    def brady(self):
        self.pattern:       str = '2{3}2*((4|0){0,3}2+)*2*'
        self.beat_id:       int = BRADY
        self.event_id:      int = 0x0020
        
        return self
    

class TBEvent:
    _EVENT_DICT: Final[Dict] = {
        'TACHY': Config().tachy(),
        'BRADY': Config().brady()
    }

    def __init__(
            self,
            beats:          NDArray,
            beat_types:     NDArray,
    ) -> None:
        try:
            self.beats:             NDArray = beats
            self.beat_types:        NDArray = beat_types.copy()
    
            self._str_symbols:       str = ''
            self._str_symbols_bk:    str = ''
    
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    def _mapping(
            self,
            index:      int,
            conv_type:  str,
            tail:       int
    ) -> None:
        self._str_symbols = self._str_symbols[:index] \
                            + conv_type \
                            + self._str_symbols[len(conv_type) + tail:]

    def _concat(
            self,
            index_find:     List,
            event_config:   Config
    ) -> None:
        try:
            index = np.arange(index_find[0], index_find[1], 1)
            self.beat_types[index] |= event_config.event_id
    
            self._mapping(
                index=index_find[0],
                conv_type=''.join(map(str, (index_find[1] - index_find[0]) * [event_config.beat_id])),
                tail=index_find[0]
            )
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            
    def _sync(
            self
    ) -> None:
        
        try:
            self._str_symbols: str = ''.join(map(str, self.beat_types))
            self._str_symbols_bk: str = deepcopy(self._str_symbols)
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    def _eval(
            self,
            title: str,
    ) -> None:
        try:
            self._sync()
            event_config = self._EVENT_DICT[title]
            
            matches = np.asarray(
                    [(i.start(0), i.end(0))
                     for i in re.finditer(event_config.pattern, self._str_symbols)]
            )
            if len(matches) == 0:
                self.beat_types &= ~event_config.beat_id
                return
            
            for i in matches:
                self._concat(i, event_config)
            
            self.beat_types &= ~event_config.beat_id
            index = df.get_index_event(self.beat_types, event_config.event_id)
            
            self.beat_types &= ~event_config.event_id
            if len(index) == 0:
                return
            
            index = index[np.flatnonzero(np.logical_and(index >= 0, index < len(self.beat_types)))]
            self.beat_types[index] = event_config.beat_id
        
        except (Exception,) as error:
            st.write_error_log(error=f'{title} - {error}', class_name=self.__class__.__name__)
            
    def _brady(
            self
    ) -> None:
        try:
            self._eval(title='BRADY')
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    def _tachy(
            self
    ) -> None:
        try:
            self._eval(title='TACHY')
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            
    def process(
            self
    ) -> NDArray:
        
        try:
            self._brady()
            self._tachy()
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return self.beat_types


class TachyBradyDetection(
    pt.Algorithm
):
    
    TOTAL_BEATS_TO_DETECT:  Final[int] = 6
    TOTAL_TACHY_EVENTS:     Final[int] = 4
    TOTAL_BRADY_EVENTS:     Final[int] = 3
    
    def __init__(
            self,
            data_structure: sr.AIPredictionResult,
            record_config:  sr.RecordConfigurations,
            is_hes_process: bool                    = False,
    ) -> None:
        try:
            super(TachyBradyDetection, self).__init__(
                data_structure=data_structure,
                record_config=record_config,
                is_hes_process=is_hes_process,
            )
            
            self._invalid_region:   NDArray = np.zeros_like(self.data_structure.beat)
            self.sync_beat_type()
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
    def _get_invalid_region(
            self
    ) -> NDArray:
        
        region = np.ones_like(self.data_structure.beat)
        try:
            # region Events
            for (hes_id, rhythm_id) in [
               [df.HOLTER_SVT,          cf.RHYTHMS_DATASTORE['classes']['SVT']],
               [df.HOLTER_VT,           cf.RHYTHMS_DATASTORE['classes']['VT']],
               [df.HOLTER_ARTIFACT,     cf.RHYTHMS_DATASTORE['classes']['OTHER']],
               [df.HOLTER_AFIB,         cf.RHYTHMS_DATASTORE['classes']['AFIB']],
               [df.HOLTER_AV_BLOCK_1,   cf.RHYTHMS_DATASTORE['classes']['AVB1']],
               [df.HOLTER_AV_BLOCK_2,   cf.RHYTHMS_DATASTORE['classes']['AVB2']],
               [df.HOLTER_AV_BLOCK_3,   cf.RHYTHMS_DATASTORE['classes']['AVB3']],
            ]:
                if self.is_hes_process:
                    index = df.check_hes_event(self.data_structure.rhythm, hes_id)
                else:
                    index = self.data_structure.rhythm == rhythm_id
                    
                index = np.flatnonzero(index)
                if len(index) > 0:
                    region[index] = self.INVALID
            # endregion Events
            
            # region LeadOff
            region |= self.data_structure.lead_off
            # endregion LeadOff
            
            # region BeatType
            index = np.flatnonzero(~np.isin(self.data_structure.symbol, df.TB_VALID_BEAT_DETECT))
            if len(index) > 0:
                region[index] = self.INVALID
            # endregion BeatType
            
            # region HR
            hrs = df.SECOND_IN_MINUTE * self.record_config.sampling_rate / np.diff(self.data_structure.beat)
            hrs = np.insert(hrs, 0, df.DEFAULT_HR_VALUE)
            
            index = np.flatnonzero(np.logical_or(hrs < df.HR_MIN_THR, hrs > df.HR_MAX_THR))
            if len(index) > 0:
                region[index] = self.INVALID
            # endregion HR
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
            
        return region
    
    def __cal(
            self,
            index: NDArray
    ) -> int:
        
        heart_rate = df.DEFAULT_HR_VALUE
        try:
            rrs = np.diff(self.data_structure.beat[index])
            heart_rate = (np.sum(rrs) - np.min(rrs) - np.max(rrs)) / 4
            heart_rate = int(round(df.SECOND_IN_MINUTE * self.record_config.sampling_rate / heart_rate))
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return heart_rate
    
    def _calculate_hr_values(
            self,
            index: NDArray
    ) -> Any:
        
        values = np.array([], dtype=int)
        try:
            if len(index) <= self.TOTAL_BEATS_TO_DETECT:
                return values
            
            # region Calculate 6 first RR intervals by GeoMean
            # rr = np.diff(self.data_structure.beat[index[:self.TOTAL_BEATS_TO_DETECT + 1]])
            # hr_before = np.array(list(map(
            #     lambda x: int(round(df.SECOND_IN_MINUTE * self.record_config.sampling_rate / geometric_mean(rr[:x]))),
            #     range(1, len(rr))
            # )))
            # hr_before1 = np.insert(hr_before, 0, df.DEFAULT_HR_VALUE)
            hr_before = np.ones(self.TOTAL_BEATS_TO_DETECT, dtype=int) * df.DEFAULT_HR_VALUE
            # endregion Calculate 6 first RR intervals by GeoMean
            
            # region Calculate RR intervals by 6RR method
            idx = np.arange(len(index) - self.TOTAL_BEATS_TO_DETECT)
            frame = idx.reshape((-1, 1)) + np.arange(0, self.TOTAL_BEATS_TO_DETECT + 1, 1)
            rr_after = np.diff(self.data_structure.beat[index[frame]], axis=1)
            rr_after = (np.sum(rr_after, axis=1) - np.max(rr_after, axis=1) - np.min(rr_after, axis=1)) / 4
            hr_after = np.round(df.SECOND_IN_MINUTE * self.record_config.sampling_rate / rr_after)
            # endregion Calculate RR intervals by 6RR method
            
            values = np.round(np.concatenate((hr_before, hr_after))).astype(int)
            pass
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return values
    
    def _determine_tachy_brady_beats(
            self,
            ind: NDArray
    ) -> NDArray:
        
        tb_beats = np.zeros_like(ind)
        try:
            heart_rate = self._calculate_hr_values(ind)

            brady_ind = np.flatnonzero(np.logical_and(
                    heart_rate < self.record_config.brady_threshold,
                    heart_rate >= df.HR_MIN_THR
            ))
            tb_beats[brady_ind] = BRADY
            pass
            
            tachy_ind = np.flatnonzero(np.logical_and(
                    heart_rate > self.record_config.tachy_threshold,
                    heart_rate <= df.HR_MAX_THR
            ))
            
            tb_beats[tachy_ind] = TACHY
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return tb_beats
    
    def _merge_event(
            self,
            events:                 NDArray,
            hes_id:                 int,
    ) -> NDArray:
        merge_events = deepcopy(events)
        try:
            cons = df.CRITERIA['SINUS']['duration']
            if hes_id not in merge_events:
                return merge_events
            
            index = np.array(list(chain.from_iterable(filter(
                lambda x: df.calculate_duration(
                        beats=self.data_structure.beat,
                        index=x,
                        sampling_rate=self.record_config.sampling_rate
                ) < cons,
                df.get_group_from_index_event(index=np.flatnonzero(merge_events & hes_id != hes_id))
            ))))
            if len(index) == 0:
                return merge_events
            
            index = index[np.flatnonzero(self._invalid_region[index] != self.INVALID)]
            if len(index) > 0:
                merge_events[index] = hes_id
            
            group_index = df.get_group_index_event(merge_events, hes_id)
            group_index = np.array(list(chain.from_iterable(filter(
                    lambda x: len(x) >= self.TOTAL_TACHY_EVENTS if hes_id == TACHY else self.TOTAL_BRADY_EVENTS,
                    group_index
            ))))
            merge_events &= ~hes_id
            if len(group_index) > 0:
                merge_events[group_index] |= hes_id
                
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return merge_events
    
    # @df.timeit
    def process(
            self
    ) -> NDArray:
        
        events = np.zeros_like(self.data_structure.beat)
        try:
            # region Invalid Region
            self._invalid_region = self._get_invalid_region()
            if np.all(self._invalid_region == self.INVALID):
                return events
            
            if np.count_nonzero(self._invalid_region == self.VALID) <= self.TOTAL_BEATS_TO_DETECT:
                return events
            # endregion Invalid Region
            
            # region Detection
            group_index = df.get_group_from_index_event(np.flatnonzero(self._invalid_region != self.INVALID))
            for index in group_index:
                if len(index) <= self.TOTAL_BEATS_TO_DETECT:
                    continue

                events[index] = TBEvent(
                    beats=self.data_structure.beat[index],
                    beat_types=self._determine_tachy_brady_beats(index),
                ).process()
                pass
            # endregion Detection
            
            # region Merge Event
            events = self._merge_event(events, TACHY)
            events = self._merge_event(events, BRADY)
            
            events[df.get_index_event(events, BRADY)] = df.HOLTER_BRADY
            events[df.get_index_event(events, TACHY)] = df.HOLTER_TACHY
            pass
            # endregion Merge Event

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return events
