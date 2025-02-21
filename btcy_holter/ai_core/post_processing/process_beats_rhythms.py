from btcy_holter import *


class PostProcessingBeatsAndRhythms:
    VALID_RHYTHM_ID:    Final[int] = 0
    INVALID_RHYTHM_ID:  Final[int] = 1

    def __init__(
            self,
            is_process_event:   bool,
            data_channel:       sr.AIPredictionResult,
            record_config:      sr.RecordConfigurations
    ) -> None:
        try:
            self.is_process_event:      Final[bool]                     = is_process_event
            self.record_config:         Final[sr.RecordConfigurations]  = record_config
            self.criteria:              Final[Dict]                     = deepcopy(df.CRITERIA)

            self.data_channel:          sr.AIPredictionResult           = data_channel
            self.data_channel.lead_off                                  = np.zeros_like(self.data_channel.beat)
            
            self.rhythm_class:          Final[Dict] = deepcopy(cf.RHYTHMS_DATASTORE['classes'])
            self.rhythm_invert:         Final[Dict] = {i: k for k, i in self.rhythm_class.items()}

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    def _check_valid_fast_rhythm(
            self,
            symbols:        NDArray,
            criteria:       Dict,
    ) -> bool:

        is_valid = False
        try:
            ind = np.flatnonzero(np.isin(symbols, criteria['symbol_valid']))
            if len(ind) == 0:
                return is_valid
            
            max_consecutive_beats = max(list(map(len, df.get_group_from_index_event(ind))))
            if max_consecutive_beats >= criteria['max_consecutive_beat_count']:
                is_valid = True

        except (Exception,) as error:
            st.write_error_log(
                    error=f'{basename(self.record_config.record_path)} - {error}', 
                    class_name=self.__class__.__name__
            )

        return is_valid

    # @df.timeit
    def _check_condition_based_on_criteria(
            self,
            index:          NDArray,
            rhythm_type:    str
    ) -> bool:
        """
            Determine valid/invalid rhythm region based on Notification Criteria.
            REF: https://itrvn.sharepoint.com/:b:/s/CNGTYCPHNITRVN-AIteam/EbWDkAVeU59Ok_KLCghBMNwB-LKxmWnq7SvHxE_eT-eipw?e=eP8rvX
        """
        
        condition = False
        criteria_keys = list(self.criteria[rhythm_type].keys())
        
        # region duration
        duration = df.calculate_duration(
                beats=self.data_channel.beat,
                index=index,
                sampling_rate=self.data_channel.sampling_rate
        )
        if (
                'duration' in criteria_keys
                and duration >= self.criteria[rhythm_type]['duration']
        ):
            condition = True
        # endregion duration

        # region num beats
        if (
                'num_beat' in criteria_keys
                and condition
                and len(index) >= self.criteria[rhythm_type]['num_beat']
        ):
            condition = True
        # endregion num beats

        # region heart rate
        if (
                'heart_rate' in criteria_keys
                and condition
        ):
            
            
            if (
                    hr := ut.HeartRate(
                        beats=self.data_channel.beat[index],
                        symbols=self.data_channel.symbol[index],
                        sampling_rate=self.data_channel.sampling_rate).process()['avgHr']
            )  == df.DEFAULT_HR_VALUE:
                return False

            func = np.greater_equal if self.criteria[rhythm_type]['heart_rate'] > 60 else np.less_equal
            if func(hr, self.criteria[rhythm_type]['heart_rate']):
                condition = True
            else:
                condition = False
        # endregion heart rate
        
        # region For SVT/VT rhythm check HR
        if (
                all(x in criteria_keys for x in ['max_consecutive_beat_count', 'symbol_valid'])
                and condition
        ):
            condition = self._check_valid_fast_rhythm(
                symbols=self.data_channel.symbol[index],
                criteria=self.criteria[rhythm_type]
            )
        # endregion For SVT/VT rhythm check HR

        return condition

    def __b_expand_noise(
            self,
    ) -> None:
        """
            Process: Label two beats close to the artifact region as "Artifact."
        """
        try:
            if self.rhythm_class['OTHER'] not in self.data_channel.rhythm:
                return
            
            if np.unique(self.data_channel.rhythm).__len__() <= 1:
                return
        
            index = np.flatnonzero(self.data_channel.rhythm == self.rhythm_class['OTHER'])
            group = np.split(index, np.flatnonzero(np.diff(index) != 1) + 1)
            
            group = np.asarray([[x[0], x[-1]] for x in group])
            index = np.concatenate((group[:, 0] - 1, group[:, 1] + 1))

            index = index[np.flatnonzero(np.logical_and(
                    index >= 0,
                    index <= len(self.data_channel.rhythm) - 1
            ))]
            self.data_channel.rhythm[index] = self.rhythm_class['OTHER']

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    def __b_long_pause_miss_identification(
            self,
            ai:             sr.AIPredictionResult,
            beat_offset:    int = 10
    ) -> sr.AIPredictionResult:
        
        """
            Process: Remove Artifact events if the region contains pause valid.
        """
        
        try:
            if len(ai.beat) == 0:
                return ai
            
            other_index = np.flatnonzero(ai.rhythm == cf.RHYTHMS_DATASTORE['rhythm_class']['OTHER'])
            if len(other_index) == 0:
                return ai
            
            ind_mark = list()
            group_others_index = np.split(other_index, np.flatnonzero(np.abs(np.diff(other_index)) != 1) + 1)
            for idx in group_others_index:
                rrs = (np.diff(ai.beat[idx]) / self.data_channel.sampling_rate) * df.MILLISECOND
                long_rr_index = idx[np.flatnonzero(rrs >= cf.PAUSE_RR_THR)]
                if len(long_rr_index) == 0 or np.all(~np.in1d(ai.symbol[long_rr_index], df.VALID_BEAT_TYPE)):
                    continue
                
                # region Verify by Pause Detection
                index = (long_rr_index[:, None] + np.arange(-beat_offset, beat_offset + 1)).flatten()
                index = index[np.flatnonzero(np.logical_and(index > 0, index < len(ai.beat)))]
                group_index = np.split(index, np.flatnonzero(np.abs(np.diff(index)) != 1) + 1)
                
                high_quality_pause_events = False
                for i, j in enumerate(group_index):
                    if len(j) == 0:
                        continue
                    
                    pause_events, _ = al.PausedDetection(
                            data_structure=ai,
                            record_config=self.record_config
                    ).process()
                    if np.all(df.check_exists_event(pause_events[beat_offset: beat_offset + 2], df.HOLTER_PAUSE)):
                        high_quality_pause_events = True
                        break
                # endregion Verify by Pause Detection
                
                if high_quality_pause_events:
                    ai.rhythm[idx] = cf.RHYTHMS_DATASTORE['rhythm_class']['SINUS']
                    _ = np.flatnonzero(ai.symbol[idx] == df.HolterSymbols.MARKED.value)
                    ind_mark.extend(idx[_])
                
            if len(ind_mark) > 0:
                ai.beat             = np.delete(ai.beat, ind_mark)
                ai.beat_channel     = np.delete(ai.beat_channel, ind_mark)
                ai.rhythm           = np.delete(ai.rhythm, ind_mark)
                ai.symbol           = np.delete(ai.symbol, ind_mark)
                ai.lead_off         = np.delete(ai.lead_off, ind_mark)
            
        except Exception as error:
            st.get_error_exception(error=error, class_name=self.__class__.__name__)
        
        return ai

    # @df.timeit
    def __r_afib_and_svt(
            self,
    ) -> None:
        """
            Process: SVT rhythm stick AFib rhythm.
            Case:
                - B/c the priority of AFib is higher than the priority of SVT, SVT has a tendency to merge with AFib.
                - Rhythm SVT which sticks rhythm AFib and invalid (based on criteria) is converted to rhythm AFib.
                - Short event Sinus (approximately 6 seconds) between rhythm SVT and rhythm AFib is converted AFib.
                - There is not any rhythm AFib around rhythm SVT or SVT is longer than 4xAFib, rhythm SVT is maintained.
        """

        try:
            rhythm_check = ['AFIB', 'SVT']
            
            if not all(self.rhythm_class[x] in self.data_channel.rhythm for x in rhythm_check):
                return
            
            if np.unique(self.data_channel.rhythm).__len__() <= 1:
                return
            
            valid_index = self.__r_check_event(ignore=rhythm_check)['valid']
            split = np.flatnonzero(np.abs(np.diff(self.data_channel.rhythm)) != 0) + 1
            group = np.split(np.arange(len(self.data_channel.rhythm)), split)
            rhythm = np.array(list(map(
                    lambda r: r[0],
                    np.split(self.data_channel.rhythm, split)
            )))

            def check_condition(
                    i: int
            ) -> bool:

                dur = df.calculate_duration(
                        beats=self.data_channel.beat,
                        index=group[i],
                        sampling_rate=self.data_channel.sampling_rate
                )

                condition = ((rhythm[i] == self.rhythm_class['SINUS'] and dur < self.criteria['SINUS']['duration'])
                             or (rhythm[i] in [self.rhythm_class['AFIB'], self.rhythm_class['SVT']])
                             or (valid_index[group[i][0]] == self.INVALID_RHYTHM_ID))

                return condition

            previous_ind = 0
            for index in np.flatnonzero(rhythm == self.rhythm_class['SVT']):
                # AFIB + SINUS (< 6s) + SVT
                before_ind = index
                while previous_ind < before_ind <= len(rhythm) - 1:
                    if check_condition(before_ind - 1):
                        before_ind -= 1
                    else:
                        break

                # SVT + SINUS (< 6s) + AFIB
                after_ind = index
                while previous_ind <= after_ind < len(rhythm) - 1:
                    if check_condition(after_ind + 1):
                        after_ind += 1
                    else:
                        break

                if after_ind == index:
                    after_ind += 1

                total_index = [group[x] for x in range(before_ind, after_ind)]
                if len(total_index) > 0:
                    def _cal_duration(i: NDArray) -> float:
                        rr = (self.data_channel.beat[i[-1]] - self.data_channel.beat[i[0]])
                        rr /= self.data_channel.sampling_rate

                        return rr

                    values = np.array(rhythm[before_ind: after_ind])
                    values = list(map(
                        lambda x: sum(map(_cal_duration,
                                          map(lambda y: group[y],
                                              before_ind + np.flatnonzero(values == self.rhythm_class[x])))),
                        rhythm_check
                    ))
                    
                    index = np.hstack(total_index)
                    if len(index) > 0:
                        if values[1] >= values[0] * self.criteria['AFIB']['duration']:
                            self.data_channel.rhythm[index] = self.rhythm_class['SVT']
                        else:
                            self.data_channel.rhythm[index] = self.rhythm_class['AFIB']

                previous_ind = after_ind

        except (Exception,) as error:
            st.write_error_log(
                    error=f'{basename(self.record_config.record_path)} - {error}', 
                    class_name=self.__class__.__name__
            )

    # @df.timeit
    def __r_merge_vt(
            self,
    ) -> None:
        """
            Process: Short rhythm ((total beat less than 2 beats and not lead-off) or invalid rhythm) between two
                     valid rhythms VT is converted to rhythm VT.
        """
        try:
            if self.rhythm_class['VT'] not in self.data_channel.rhythm:
                return
            
            if np.unique(self.data_channel.rhythm).__len__() <= 1:
                return
            
            valid_rhythm = self.__r_check_event(ignore=['VT'])['valid']

            index = np.flatnonzero(self.data_channel.rhythm == self.rhythm_class['VT'])
            group_vt = np.split(index, np.flatnonzero(np.diff(index) != 1) + 1)
            if len(group_vt) < 2:
                return

            index_diff = np.setdiff1d(np.arange(len(self.data_channel.rhythm)), index)
            group = np.split(index_diff, np.flatnonzero(np.diff(index_diff) != 1) + 1)

            lens = list(map(
                lambda x: (len(x) <= 2) & (valid_rhythm[x[0]] == self.INVALID_RHYTHM_ID),
                group
            ))
            lens = np.flatnonzero(np.array(lens))
            if len(lens) == 0:
                return

            index = list(chain.from_iterable(map(
                lambda x, i: x if i in lens else list(),
                group,
                range(len(group)))
            ))
            if len(index) == 0:
                return
            
            index = np.hstack(index)
            index = index[np.flatnonzero(self.data_channel.lead_off[index] == 0)]
            if len(index) > 0:
                self.data_channel.rhythm[index] = self.rhythm_class['VT']

        except (Exception,) as error:
            st.write_error_log(
                    error=f'{basename(self.record_config.record_path)} - {error}', 
                    class_name=self.__class__.__name__
            )

    # @df.timeit
    def __r_extend_vt_svt_based_on_beats(
            self,
    ) -> None:
        """
            Process: Extend SVT/VT if exists S/V near to start/stop rhythm.
        """
        try:
            for rhythm_type, beat_type in [
                ('VT', 'V'),
                ('SVT', 'S')
            ]:
                index = np.flatnonzero(self.data_channel.rhythm == self.rhythm_class[rhythm_type])
                if len(index) == 0:
                    continue

                # region Merge
                group_event = np.split(index, np.flatnonzero(np.diff(index) != 1) + 1)
                group_event = list(filter(lambda x: len(x) > 0, group_event))
                if len(group_event) > 1:
                    group_event = np.array(list(map(lambda x: x[[0, -1]], group_event)))
                    group_check = np.column_stack((group_event[:-1, 1], group_event[1:, 0]))
                    merge_ind = np.array(list(filter(
                        lambda x: np.array_equal(np.unique(self.data_channel.symbol[x[0] + 1: x[-1] + 1]),
                                                 np.array([beat_type])),
                        group_check
                    )))

                    if len(merge_ind) > 0:
                        merge_ind = np.array(list(chain.from_iterable(map(
                            lambda x: np.arange(x[0], x[-1] + 1),
                            merge_ind
                        ))))
                        self.data_channel.rhythm[merge_ind] = self.rhythm_class[rhythm_type]
                        index = np.flatnonzero(self.data_channel.rhythm == self.rhythm_class[rhythm_type])
                # endregion Merge

                # region Extend
                index_beat_type = np.setdiff1d(np.arange(len(self.data_channel.rhythm)), index)
                index_beat_type = index_beat_type[np.flatnonzero(self.data_channel.symbol[index_beat_type] == beat_type)]
                group_beat_type = np.split(index_beat_type, np.flatnonzero(np.diff(index_beat_type) != 1) + 1)
                group_beat_type = list(filter(lambda x: len(x) > 0, group_beat_type))

                if len(group_beat_type) > 0:
                    group_beat_type = np.array(list(map(
                            lambda x: x[[0, -1]],
                            group_beat_type
                    )))
                    
                    new_index_events = list()
                    group_index = np.split(index, np.flatnonzero(np.diff(index) != 1) + 1)
                    for index in group_index:
                        if len(index) < self.criteria[rhythm_type]['num_beat']:
                            continue

                        dur = df.calculate_duration(
                                beats=self.data_channel.beat,
                                index=index,
                                sampling_rate=self.data_channel.sampling_rate
                        )
                        if dur < self.criteria[rhythm_type]['duration']:
                            continue

                        start_index = index[0]
                        if (
                                start_index > 0
                                and self.data_channel.symbol[start_index - 1] == beat_type
                        ):
                            check = np.flatnonzero(np.logical_and(
                                    group_beat_type[:, 0] <= start_index - 1,
                                    group_beat_type[:, 1] >= start_index - 1
                            ))
                            if len(check) > 0:
                                start_index = group_beat_type[check[0]][0]

                        stop_index = index[-1]
                        if (
                                stop_index < len(self.data_channel.rhythm) - 1
                                and self.data_channel.symbol[stop_index + 1] == beat_type
                        ):
                            check = np.flatnonzero(np.logical_and(
                                    group_beat_type[:, 0] <= stop_index + 1,
                                    group_beat_type[:, 1] >= stop_index + 1
                            ))
                            if len(check) > 0:
                                stop_index = group_beat_type[check[0]][-1]

                        new_index_events.append(np.arange(start_index, stop_index + 1, 1))

                    if len(new_index_events) > 0:
                        new_index_events = np.array(list(set(chain.from_iterable(new_index_events))))
                        self.data_channel.rhythm[new_index_events] = self.rhythm_class[rhythm_type]
                # endregion Extend

        except (Exception,) as error:
            st.write_error_log(
                    error=f'{basename(self.record_config.record_path)} - {error}', 
                    class_name=self.__class__.__name__
            )

    # @df.timeit
    def __r_split_vt_svt_based_on_beat_types(
            self,
    ) -> None:
        """
            Process: Scale SVT/VT region if exists N/Q/R at start/stop rhythm.
        """
        try:
            for rhythm_type, beat_type in [
                ('VT', 'V'),
                ('SVT', 'S')
            ]:
                index = np.flatnonzero(self.data_channel.rhythm == self.rhythm_class[rhythm_type])
                if len(index) == 0:
                    continue

                group_index = np.split(index, np.flatnonzero(np.abs(np.diff(index)) != 1) + 1)
                group_index = list(filter(lambda x: len(x) > 0, group_index))

                cons_beat = df.CRITERIA[rhythm_type]['max_consecutive_beat_count']
                cons_pattern = f'{beat_type}' + '{' + f'{cons_beat}' + '}' + f'{beat_type}*'
                for index in group_index:
                    if np.all(self.data_channel.symbol[index] == beat_type):
                        continue

                    str_symbols = ''.join(list(self.data_channel.symbol[index]))
                    valid_index = np.array(list(chain.from_iterable(map(
                        lambda x: range(x.start(0), x.end(0)),
                        re.finditer(cons_pattern, str_symbols))
                    )))
                    invalid_index = np.setdiff1d(np.arange(len(index)), valid_index)
                    if len(invalid_index) == 0:
                        continue

                    gr = np.split(invalid_index, np.flatnonzero(np.abs(np.diff(invalid_index)) != 1) + 1)
                    gr = list(filter(lambda x: len(x) > 0, gr))
                    if (
                            len(gr) > 0
                            and gr[0][0] == 0
                            and index[0] > 0
                    ):
                        self.data_channel.rhythm[index[gr[0]]] = self.data_channel.rhythm[index[0] - 1]
                        gr.pop(0)

                    if (
                            len(gr) > 0
                            and gr[-1][-1] == len(index) - 1
                            and index[-1] + 1 < len(self.data_channel.symbol)
                    ):
                        self.data_channel.rhythm[index[gr[-1]]] = self.data_channel.rhythm[index[-1] + 1]
                        gr.pop(-1)

                    invalid_index = np.array(list(chain.from_iterable(gr)))
                    if len(invalid_index) == 0:
                        continue
                    
                    index = index[invalid_index]
                    index = index[np.flatnonzero(np.logical_and(
                            index >= 0, 
                            index <= len(self.data_channel.rhythm) - 1
                    ))]
                    if len(index) > 0:
                        self.data_channel.rhythm[index] = self.rhythm_class['SINUS']

        except (Exception,) as error:
            st.write_error_log(
                    error=f'{basename(self.record_config.record_path)} - {error}', 
                    class_name=self.__class__.__name__
            )

    # @df.timeit
    def __r_check_event(
            self,
            **kwargs
    ) -> Dict:
        """
            Process: Determine valid/invalid rhythm region based on Notification Criteria.
        """

        ranking = list()
        valid_rhythm = np.zeros_like(self.data_channel.rhythm)
        try:
            list_rhythm_types = list(map(lambda x: self.rhythm_invert[x], np.unique(self.data_channel.rhythm)))
            for rhythm_type in list_rhythm_types:
                ranking.append(rhythm_type)

                index_rhythm = np.flatnonzero(self.data_channel.rhythm == self.rhythm_class[rhythm_type])
                if len(index_rhythm) == 0 or rhythm_type in kwargs.get('ignore', list()):
                    continue

                groups = np.split(index_rhythm, np.flatnonzero(np.abs(np.diff(index_rhythm)) != 1) + 1)
                invalid_index = list(chain.from_iterable(filter(
                    lambda y: len(y) > 0 and not self._check_condition_based_on_criteria(y, rhythm_type),
                    groups
                )))
                if len(invalid_index) > 0:
                    valid_rhythm[np.hstack(invalid_index)] = self.INVALID_RHYTHM_ID

        except (Exception,) as error:
            st.write_error_log(
                    error=f'{basename(self.record_config.record_path)} - {error}', 
                    class_name=self.__class__.__name__
            )

        return {
            'valid':    valid_rhythm,
            'rank':     ranking
        }

    # @df.timeit
    def __remark_invalid_region_to_target_rhythm(
            self,
            target_rhythm:  int,
            index:          NDArray
    ) -> None:
        """
            Process: Mark invalid region to target rhythm id.
        """

        try:
            if (
                    any(x in self.rhythm_invert[target_rhythm] for x in ['VT', 'AVB'])
                    and (hr := ut.HeartRate(
                                beats=self.data_channel.beat[index],
                                symbols=self.data_channel.symbol[index],
                                sampling_rate=self.data_channel.sampling_rate
                            ).process()['avgHr']) != df.DEFAULT_HR_VALUE
            ):
                keys_list = list(self.criteria[self.rhythm_invert[target_rhythm]].keys())
                
                hr_threshold = None
                if 'heart_rate' in keys_list:
                    hr_threshold = self.criteria[self.rhythm_invert[target_rhythm]]['heart_rate']

                func = np.greater_equal \
                    if 'VT' in self.rhythm_invert[target_rhythm] \
                    else np.less_equal
                
                if hr_threshold is not None and func(hr, hr_threshold):
                    self.data_channel.rhythm[index] = target_rhythm

                elif df.calculate_duration(self.data_channel.beat,
                                           index,
                                           self.data_channel.sampling_rate) >= self.criteria['SINUS']['duration']:
                    self.data_channel.rhythm[index] = self.rhythm_class['SINUS']

                else:
                    self.data_channel.rhythm[index] = target_rhythm

            else:
                self.data_channel.rhythm[index] = target_rhythm

        except (Exception,) as error:
            st.write_error_log(
                    error=f'{basename(self.record_config.record_path)} - {error}', 
                    class_name=self.__class__.__name__
            )

    def __r_event_invalid_by_hr_to_sinus(
            self,
            valid_rhythm:   NDArray,
            rank:           List
    ) -> [NDArray, NDArray, List]:
        """
            Process:
                - If SVT, VT, AVB1, AVB2, or AVB3 do not meet the heart rate criteria in the Notification Criteria and
                have a duration exceeding 6 seconds, then label the region as Sinus.
        """
        try:
            ind = np.flatnonzero(np.logical_and(
                self.data_channel.rhythm > self.rhythm_class['AFIB'],
                valid_rhythm == self.INVALID_RHYTHM_ID
            ))
            if len(ind) == 0:
                return valid_rhythm, rank
            
            group = np.split(ind, np.flatnonzero(np.abs(np.diff(ind)) != 1) + 1)
            index = np.array(list(map(lambda x: x[[0, -1]], group)))
            index = np.column_stack((index[:-1, 1], index[1:, 0]))

            range_index = list(map(lambda x: np.arange(x[0], x[-1] + 1), index))
            merge = list(chain.from_iterable(filter(
                lambda x: np.array_equal(np.unique(valid_rhythm[x[0]: x[-1] + 1]), [self.rhythm_class['SINUS']]),
                range_index
            )))

            total_index = np.concatenate((np.array(list(chain.from_iterable(group))), merge))
            total_index = np.array(sorted(total_index)).astype(int)

            group = np.split(total_index, np.flatnonzero(np.abs(np.diff(total_index)) != 1) + 1)
            sinus_threshold = self.criteria['SINUS']['duration'] * self.data_channel.sampling_rate
            total_index = list(chain.from_iterable(filter(
                lambda x: self.data_channel.beat[x[-1]] - self.data_channel.beat[x[0]] >= sinus_threshold,
                group
            )))
            if len(total_index) == 0:
                return valid_rhythm, rank
            
            valid_group = np.array(total_index).astype(int)
            self.data_channel.rhythm[valid_group] = self.rhythm_class['SINUS']
            valid_rhythm[valid_group] = self.VALID_RHYTHM_ID
            rank.append('SINUS')

        except (Exception,) as error:
            st.write_error_log(
                    error=f'{basename(self.record_config.record_path)} - {error}', 
                    class_name=self.__class__.__name__
            )

        return valid_rhythm, rank

    # @df.timeit
    def __r_process_rhythm(
            self,
            **kwargs
    ) -> None:

        try:
            valid_rhythm = kwargs['valid']
        except (Exception,):
            valid_rhythm = np.ones_like(self.data_channel.rhythm)

        try:
            rank = kwargs['rank']
        except (Exception,):
            rank = list()

        try:
            valid_rhythm, rank = self.__r_event_invalid_by_hr_to_sinus(
                valid_rhythm=valid_rhythm,
                rank=rank,
            )

            invalid_index = np.flatnonzero(valid_rhythm == self.INVALID_RHYTHM_ID)
            group_invalid_index = np.split(invalid_index, np.flatnonzero(np.abs(np.diff(invalid_index)) != 1) + 1)
            if np.array_equal(valid_rhythm, np.ones_like(valid_rhythm)):
                self.data_channel.rhythm[:] = self.rhythm_class['SINUS']

            elif len(group_invalid_index) > 0:
                for index in group_invalid_index:
                    if len(index) == 0:
                        continue

                    self.data_channel.rhythm[index] = self.rhythm_class['SINUS']
                    thr = len(self.data_channel.ecg_signal) // (4 * self.data_channel.sampling_rate)
                    if df.calculate_duration(self.data_channel.beat, index, self.data_channel.sampling_rate) >= thr:
                        self.data_channel.rhythm[index] = self.rhythm_class['SINUS']
                        continue

                    pres_rhythm = self.data_channel.rhythm[index[0] - 1] \
                        if index[0] - 1 >= 0 \
                        else -1
                    
                    post_rhythm = self.data_channel.rhythm[index[-1] + 1] \
                        if index[-1] + 1 <= len(self.data_channel.rhythm) - 1 \
                        else -1

                    if pres_rhythm < 0 <= post_rhythm:
                        self.__remark_invalid_region_to_target_rhythm(post_rhythm, index)

                    elif (pres_rhythm >= 0 > post_rhythm) or (0 < pres_rhythm == post_rhythm):
                        self.__remark_invalid_region_to_target_rhythm(pres_rhythm, index)

                    else:

                        pres_rhythm_rank = rank.index(self.rhythm_invert[pres_rhythm])
                        post_rhythm_rank = rank.index(self.rhythm_invert[post_rhythm])

                        if pres_rhythm_rank > post_rhythm_rank:
                            tg1 = post_rhythm
                            tg2 = pres_rhythm
                        else:
                            tg1 = pres_rhythm
                            tg2 = post_rhythm

                        spec_tg1 = any(x in self.rhythm_invert[tg1] for x in ['VT', 'AVB'])
                        spec_tg2 = any(x in self.rhythm_invert[tg2] for x in ['VT', 'AVB'])
                        if (spec_tg1 and not spec_tg2) or (not spec_tg1 and spec_tg2):
                            tg = tg1 if (spec_tg1 and not spec_tg2) else tg2
                            self.__remark_invalid_region_to_target_rhythm(tg, index)

                        elif spec_tg1 and spec_tg2:
                            self.data_channel.rhythm[index] = 0
                            
                            if (
                                    hr:= ut.HeartRate(
                                        beats=self.data_channel.beat[index],
                                        symbols=self.data_channel.symbol[index],
                                        sampling_rate=self.data_channel.sampling_rate
                                    ).process()['avgHr'] == df.DEFAULT_HR_VALUE
                            ):
                                self.data_channel.rhythm[index] = tg1

                            else:
                                for tg in [tg1, tg2]:
                                    hr_cre = self.criteria[self.rhythm_invert[tg]].get('heart_rate', None)
                                    func = np.greater_equal if 'VT' in self.rhythm_invert[tg] else np.less_equal
                                    
                                    if hr_cre is not None and func(hr, hr_cre):
                                        self.data_channel.rhythm[index] = tg
                                        break

                            if self.data_channel.rhythm[index][0] == 0:
                                self.data_channel.rhythm[index] = spec_tg1

                        else:
                            self.data_channel.rhythm[index] = tg1

        except (Exception,) as error:
            st.write_error_log(
                    error=f'{basename(self.record_config.record_path)} - {error}', 
                    class_name=self.__class__.__name__
            )

    def __b_convert_beat_s_in_afib(
            self,
    ) -> None:
        """
            Process: Converting beat types match to the rhythms
                - SVE in AFib => N
        """

        try:
            ind = np.flatnonzero(np.logical_and(
                self.data_channel.rhythm == self.rhythm_class['AFIB'],
                self.data_channel.symbol == df.HolterSymbols.SVE.value
            ))
            if len(ind) > 0:
                self.data_channel.symbol[ind] = df.HolterSymbols.N.value

        except (Exception,) as error:
            st.write_error_log(
                    error=f'{basename(self.record_config.record_path)} - {error}', 
                    class_name=self.__class__.__name__
            )

    # @df.timeit
    def __b_map_beat_rhythm(
            self,
    ) -> None:
        """
            Process: Converting beat types match to the rhythms
                - Symbol in Other => OTHER
        """

        try:
            def convert_beat_type(r):
                index = np.flatnonzero(self.data_channel.rhythm == self.rhythm_class[r])
                if len(index) > 0:
                    self.data_channel.symbol[index] = cf.SYMBOLS_IN_RHYTHMS[r]

            if not np.array_equal(
                    [df.HolterSymbols.MARKED.value] * df.LIMIT_BEAT_SAMPLE_IN_SIGNAL,
                    self.data_channel.symbol
            ):
                list(map(convert_beat_type, cf.SYMBOLS_IN_RHYTHMS.keys()))

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    # @df.timeit
    def process_rhythm_based_on_criteria(
            self
    ) -> None:
        try:
            self.__b_expand_noise()
            if np.all(self.data_channel.rhythm == self.rhythm_class['SINUS']):
                
                return
            
            if np.all(self.data_channel.rhythm == self.rhythm_class['OTHER']):
                self.__b_map_beat_rhythm()
                
                return
     
            self.__r_afib_and_svt()
            
            self.__r_merge_vt()
            
            self.__r_extend_vt_svt_based_on_beats()
            
            self.__r_process_rhythm(**self.__r_check_event())

            self.__r_split_vt_svt_based_on_beat_types()

            self.__b_convert_beat_s_in_afib()
            
            self.__b_map_beat_rhythm()

        except (Exception, ) as error:
            st.write_error_log(
                    error=f'{basename(self.record_config.record_path)} - {error}', 
                    class_name=self.__class__.__name__
            )

    # @df.timeit
    def process(
            self
    ) -> sr.AIPredictionResult:

        try:
            if (
                    self.data_channel.ecg_signal is not None
                    and len(self.data_channel.ecg_signal) > 0
            ):
                self.data_channel = al.RemoveBeats(
                    is_process_event=self.is_process_event,
                    data_structure=self.data_channel,
                ).process()

                if len(self.data_channel.beat) <= df.LIMIT_BEAT_SAMPLE_IN_SIGNAL:
                    self.data_channel.beat = df.initialize_two_beats_at_ecg_data(
                        len_ecg=len(self.data_channel.ecg_signal),
                        sampling_rate=self.data_channel.sampling_rate
                    )
                    self.data_channel.symbol        = np.array([df.HolterSymbols.MARKED.value] * len(self.data_channel.beat))
                    self.data_channel.rhythm        = np.ones_like(self.data_channel.beat) * self.rhythm_class['OTHER']
                    self.data_channel.lead_off      = np.zeros_like(self.data_channel.beat)
                    self.data_channel.beat_channel  = np.ones_like(self.data_channel.beat) * self.data_channel.channel

            self.process_rhythm_based_on_criteria()

        except (Exception, ) as error:
            st.write_error_log(
                    error=f'{basename(self.record_config.record_path)} - {error}', 
                    class_name=self.__class__.__name__
            )

        return self.data_channel
