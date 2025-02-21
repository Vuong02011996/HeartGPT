from btcy_holter import *


class EventPriority:
    MAX_STRIP_LENGTH: Final[int] = df.MAX_STRIP_LEN
    
    FUNCTION_DICT = dict()
    FUNCTION_DICT['max'] = dict()
    FUNCTION_DICT['max']['compare'] = np.greater
    FUNCTION_DICT['max']['select']  = np.argmax
    
    FUNCTION_DICT['min'] = dict()
    FUNCTION_DICT['min']['compare'] = np.less
    FUNCTION_DICT['min']['select']  = np.argmin
    
    def __init__(
            self,
            event:          Dict,
            symptomatic:    bool    = False,
            project_name:   str     = 'hourly',
            sampling_rate:  int     = cf.SAMPLING_RATE
    ) -> None:
        try:
            self.event:             Dict        = event
            self.sampling_rate:     Final[int]  = sampling_rate
            self.project_name:      Final[str]  = project_name
            self.symptomatic:       Final[bool] = symptomatic
            
            self.is_strip_reports:  Final[bool] = project_name == 'hourly'

            self.strip_len:         Final[int] = int(self.MAX_STRIP_LENGTH * self.sampling_rate)
            self.offset:            Final[int] = int(1 * self.sampling_rate)
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    def _get_strip_by_heart_rate(
            self,
            criteria:       Dict,
            priority:       str,
            cal_type:       str     = 'max',
            hr_type:        str     = 'avgHr',
    ) -> bool:

        status = False
        try:
            duration = criteria.get('duration', self.MAX_STRIP_LENGTH)
            if self.event['duration'] < duration:
                return status

            if self.project_name == 'event':
                hr_type = 'avgHr'

            group_index = np.split(
                np.arange(len(self.event['beat'])),
                np.flatnonzero(np.diff(self.event['beat']) < 0) + 1
            )

            strip_frame = duration * self.sampling_rate
            for index in group_index:
                # region strips
                start_region, end_region = self.event['beat'][index][[0, -1]]
                if self.is_strip_reports:
                    if end_region - start_region < self.strip_len:
                        continue

                    frame = np.arange(start_region, end_region - strip_frame, self.offset)
                    seg_frame = frame.reshape((-1, 1)) + [0, strip_frame + 1]
                else:
                    seg_frame = np.arange(start_region, end_region).reshape((1, -1))
                # endregion strips

                for (start_strip, stop_strip) in seg_frame:
                    # region calculate heart rate
                    idx = np.flatnonzero(np.logical_and(
                        self.event['beat'][index] >= start_strip,
                        self.event['beat'][index] <= stop_strip
                    ))

                    hr = None
                    if self.is_strip_reports:
                        hr = ut.HeartRate(
                                beats=self.event['beat'][index][idx],
                                symbols=self.event['beat_type'][index][idx],
                                sampling_rate=self.sampling_rate
                        ).process()
                        
                    elif 'minHr' in self.event.keys():
                        hr = dict()
                        hr['minHr'] = self.event['minHr']
                        hr['maxHr'] = self.event['maxHr']
                        hr['avgHr'] = self.event['avgHr']

                    if hr is None or hr['avgHr'] != df.DEFAULT_HR_VALUE:
                        continue
                    # endregion calculate heart rate

                    # region capture strip
                    if self.FUNCTION_DICT[cal_type]['compare'](hr[hr_type], criteria['heart_rate']):
                        self.event['startStrip'] = self.event['epoch'][0]
                        self.event['priority']   = priority.capitalize()
                        if self.is_strip_reports:
                            frames = np.arange(start_strip, stop_strip - min(self.strip_len, strip_frame), self.offset)
                            frames = frames.reshape((-1, 1)) + [-self.strip_len, self.strip_len]

                            heart_rate_values = list()
                            for (begin, end) in frames:
                                i = np.flatnonzero(np.logical_and(
                                    self.event['beat'] >= begin,
                                    self.event['beat'] <= end
                                ))
                                if len(i) == 0:
                                    continue

                                hr = ut.HeartRate(
                                        beats=self.event['beat'][i],
                                        symbols=self.event['beat_type'][i],
                                        sampling_rate=self.sampling_rate
                                ).process()
                                if hr['avgHr'] != df.DEFAULT_HR_VALUE:
                                    heart_rate_values.append([hr['avgHr'], self.event['epoch'][i[0]]])

                            if len(heart_rate_values) > 0:
                                heart_rate_values = np.array(heart_rate_values)
                                i = self.FUNCTION_DICT[cal_type]['select'](heart_rate_values[:, 0])
                                self.event['startStrip'] = heart_rate_values[i, -1]

                            status = True
                            break
                        # endregion capture strip

                    if status:
                        break

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return status

    def _get_strip_by_count_beats(
            self,
            criteria:       Dict,
            priority:       str,
            cal_type:       str     = 'max',
            hr_type:        str     = 'avgHr',
    ) -> bool:

        status = False
        try:
            if 'num_beat' not in criteria.keys():
                return status
            
            
            if self.event['countBeats'] < criteria['num_beat']:
                return status

            if self.project_name == 'event':
                hr_type = 'avgHr'

            group_index = np.split(
                np.arange(len(self.event['beat'])),
                np.flatnonzero(np.diff(self.event['beat']) < 0) + 1
            )

            for index in group_index:
                start_region, end_region = self.event['beat'][index][[0, -1]]
                if self.is_strip_reports:
                    if end_region - start_region < self.strip_len:
                        continue

                    frame = np.arange(start_region, end_region - self.strip_len, self.offset)
                    seg_frame = frame.reshape((-1, 1)) + [0, self.strip_len + 1]
                else:
                    seg_frame = np.arange(start_region, end_region).reshape((1, -1))

                for (start_strip, stop_strip) in seg_frame:
                    idx = np.flatnonzero(np.logical_and(
                        self.event['beat'][index] >= start_strip,
                        self.event['beat'][index] <= stop_strip
                    ))
                    if len(idx) >= criteria['num_beat']:
                        hr = None
                        if self.is_strip_reports:
                            hr = ut.HeartRate(
                                beats=self.event['beat'][index][idx],
                                symbols=self.event['beat_type'][index][idx],
                                sampling_rate=self.sampling_rate
                            ).process()
                            
                        elif 'minHr' in self.event.keys():
                            hr = dict()
                            hr['minHr'] = self.event['minHr']
                            hr['maxHr'] = self.event['maxHr']
                            hr['avgHr'] = self.event['avgHr']
                        
                        if hr is None or hr['avgHr'] != df.DEFAULT_HR_VALUE:
                            continue

                        if self.FUNCTION_DICT[cal_type]['compare'](hr[hr_type], criteria['heart_rate']):
                            self.event['priority'] = priority.capitalize()
                            if self.is_strip_reports:
                                self.event['startStrip'] = self.event['epoch'][index][idx][0]
                            status = True
                            break

                    if status:
                        break

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return status

    # @df.timeit
    def _vt_priority(
            self,
    ) -> None:
        try:
            if not self.is_strip_reports:
                event_criteria = deepcopy(df.EVENT_STRIP_VT_CRITERIA) \
                    if not self.symptomatic \
                    else deepcopy(df.EVENT_STRIP_VT_CRITERIA_SYMPTOMATIC)
            else:
                event_criteria = deepcopy(df.STRIP_VT_CRITERIA)
            
            if self.event['maxHr'] == df.DEFAULT_HR_VALUE:
                return
            
            for priority, criteria in event_criteria.items():
                if len(criteria) > 1 and isinstance(criteria, list):
                    if (
                            self.event['maxHr'] >= criteria[0]['heart_rate']
                            and self._get_strip_by_heart_rate(criteria[0], priority, cal_type='max', hr_type='maxHr')
                    ):
                        break

                    if (
                            self.event['maxHr'] >= criteria[1]['heart_rate']
                            and self._get_strip_by_count_beats(criteria[1], priority, cal_type='max', hr_type='maxHr')
                    ):
                        break

                elif (
                        'heart_rate' in criteria.keys()
                        and self.event['maxHr'] >= criteria['heart_rate']
                        and self._get_strip_by_count_beats(criteria, priority, cal_type='max', hr_type='maxHr')
                ):
                    break

                elif (
                        'num_beat' in criteria.keys()
                        and self.event['countBeats'] > criteria['num_beat']
                ):
                    break

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def _pause_priority(
            self,
    ) -> None:
        try:
            if not self.is_strip_reports:
                event_criteria = deepcopy(df.EVENT_STRIP_PAUSE_CRITERIA)
            else:
                event_criteria = deepcopy(df.STRIP_PAUSE_CRITERIA)

            for priority, criteria in event_criteria.items():
                if self.event['duration'] < criteria['duration']:
                    continue

                self.event['priority'] = priority.capitalize()
                self.event['startStrip'] = self.event['epoch'][0]
                break

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def _avb3_priority(
            self,
    ) -> None:
        try:
            if not self.is_strip_reports:
                event_criteria = deepcopy(df.EVENT_STRIP_AVB3_CRITERIA)
            else:
                event_criteria = deepcopy(df.STRIP_AVB3_CRITERIA)
            
            if self.event['minHr'] == df.DEFAULT_HR_VALUE:
                return 
            
            for priority, criteria in event_criteria.items():
                if self.event['minHr'] > criteria['heart_rate']:
                    continue

                if self._get_strip_by_heart_rate(criteria,  priority, cal_type='min', hr_type='minHr'):
                    break

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def _avb2_priority(
            self,
    ) -> None:
        try:
            if not self.is_strip_reports:
                event_criteria = deepcopy(df.EVENT_STRIP_AVB2_CRITERIA)
            else:
                event_criteria = deepcopy(df.STRIP_AVB2_CRITERIA)

            for priority, criteria in event_criteria.items():
                if self.event['minHr'] > criteria['heart_rate']:
                    continue

                if self._get_strip_by_heart_rate(criteria, priority, cal_type='min', hr_type='minHr'):
                    break

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def _afib_priority(
            self,
    ) -> None:

        try:
            if not self.is_strip_reports:
                event_criteria = deepcopy(df.EVENT_STRIP_AFIB_CRITERIA)
            else:
                event_criteria = deepcopy(df.STRIP_AFIB_CRITERIA)

            for priority, criteria in event_criteria.items():
                if (
                        self.event['maxHr'] >= criteria['max_hr']['heart_rate']
                        and self._get_strip_by_heart_rate(criteria['max_hr'], priority, cal_type='max', hr_type='maxHr')
                ):
                    break

                if (
                        self.event['minHr'] <= criteria['min_hr']['heart_rate']
                        and self._get_strip_by_heart_rate(criteria['min_hr'], priority, cal_type='min', hr_type='minHr')
                ):
                    break
            
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def _svt_priority(
            self
    ) -> None:
        
        try:
            if not self.is_strip_reports:
                event_criteria = deepcopy(df.EVENT_STRIP_SVT_CRITERIA) \
                    if not self.symptomatic \
                    else deepcopy(df.EVENT_STRIP_SVT_CRITERIA_SYMPTOMATIC)
            else:
                event_criteria = deepcopy(df.STRIP_SVT_CRITERIA)
            
            if self.event['maxHr'] == df.DEFAULT_HR_VALUE:
                return

            for priority, criteria in event_criteria.items():
                if self.event['maxHr'] < criteria['heart_rate']:
                    continue

                if (
                    'duration' in criteria.keys()
                    and self.event['duration'] >= criteria['duration']
                    and self._get_strip_by_heart_rate(criteria, priority, cal_type='max', hr_type='maxHr')
                ):
                    break
                    
                elif self._get_strip_by_count_beats(criteria, priority, cal_type='max', hr_type='maxHr'):
                    break
                    
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def _brady_priority(
            self,
    ) -> None:
        try:
            if not self.is_strip_reports:
                event_criteria = deepcopy(df.EVENT_STRIP_BRADY_CRITERIA)
            else:
                event_criteria = deepcopy(df.STRIP_BRADY_CRITERIA)

            for priority, criteria in event_criteria.items():
                if self.event['duration'] < criteria['duration']:
                    continue

                if self.event['minHr'] > criteria['heart_rate']:
                    continue

                if self._get_strip_by_heart_rate(criteria, priority, cal_type='min', hr_type='minHr'):
                    break
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def _tachy_priority(
            self,
    ) -> None:
        
        try:
            if not self.is_strip_reports:
                event_criteria = deepcopy(df.EVENT_STRIP_TACHY_CRITERIA) \
                    if not self.symptomatic \
                    else deepcopy(df.EVENT_STRIP_TACHY_CRITERIA_SYMPTOMATIC)
            else:
                event_criteria = deepcopy(df.STRIP_TACHY_CRITERIA)

            for priority, criteria in event_criteria.items():
                if self.event['maxHr'] < criteria['heart_rate']:
                    continue

                if self._get_strip_by_heart_rate(criteria, priority, cal_type='max', hr_type='maxHr'):
                    break
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def process(
            self,
    ) -> Dict:
        
        try:
            match self.event['type']:
                case 'VT':
                    self._vt_priority()
                
                case 'PAUSE':
                    self._pause_priority()
                
                case 'AVB3':
                    self._avb3_priority()
                
                case 'AVB2':
                    self._avb2_priority()
                
                case 'AFIB':
                    self._afib_priority()
                
                case 'SVT':
                    self._svt_priority()
                
                case 'BRADY':
                    self._brady_priority()
                
                case 'TACHY':
                    self._tachy_priority()
                
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return self.event
