from btcy_holter import *

from btcy_holter.calculate.strip_comment import (
    StripComment
)
from btcy_holter.calculate.event_priority import (
    EventPriority
)


class EcgEventSummary(
        pt.Summary
):
    PERCENT_STRIP:                  Final[int] = 0.8                        # %

    ONSET_EPOCH:                    Final[int] = 5 * df.MILLISECOND
    OFFSET_EPOCH:                   Final[int] = 1 * df.MILLISECOND
    STRIP_LEN:                      Final[int] = df.MIN_STRIP_LEN
    MAX_TRIP_LENGTH:                Final[int] = df.MAX_STRIP_LEN
    MAX_NUM_STRIPS_PER_EVENT:       Final[int] = df.MAX_NUM_STRIPS_PER_EVENT

    SELECT_PRIORITY_EVENTS:         Final[List[str]] = ['Emergent', 'Urgent']

    def __init__(
            self,
            save_file:          str,
            beat_df:            pl.DataFrame,
            record_config:      sr.RecordConfigurations,
            all_files:          List,
            **kwargs
    ) -> None:
        try:
            super(EcgEventSummary, self).__init__(
                    beat_df=beat_df,
                    record_config=record_config,
                    all_files=all_files
            )

            self.save_file:                 str = save_file

            self.list_event_ids:            List = list()
            self._event_summary:            List = list()
            
            self._step:                     int = kwargs.get('step', 2)

            self.beat_df: Final[pl.DataFrame] = (
                self.beat_df
                .select(
                    [
                        'EPOCH',
                        'CHANNEL',
                        'BEAT',
                        'BEAT_TYPE',
                        'EVENT',
                        'FILE_INDEX'
                    ]
                )
            )

            self._study_data, self._npy_col = df.generate_study_data(self.beat_df)
            self._study_data[:, self._npy_col.channel] = (
                df.get_channel_from_channel_column(self._study_data[:, self._npy_col.channel]))

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    def log(
            self,
            event_summary:      List,
            event_dataframe:    pl.DataFrame = None,
            num_of_col:         int = 5
    ) -> None:
        try:
            
            if event_summary is None or len(event_summary) == 0:
                return
            
            if event_dataframe is not None:
                st.LOGGING_SESSION.info(
                        f'{self.__class__.__name__}: '
                        f'- summary: {len(event_summary)} events '
                        f'- dataframe: {len(event_dataframe)} / {dict(Counter(list(event_dataframe["statusCode"])))} '
                )
                
            # region newEvents
            st.LOGGING_SESSION.info(f'{self._step}. newEvents [totalEvents: {len(event_summary)}]:' + ' {')
            total_events = dict(Counter(list(map(lambda y: y['type'], event_summary))))
            total_events = sorted(total_events.items(), key=lambda y: y[1], reverse=True)
            
            for x in range(0, len(total_events), num_of_col):
                tmp = [f'{total_events[x + i][0]:13}: {total_events[x + i][-1]:4}'
                       for i in range(num_of_col) if x + i < len(total_events)]
                tmp = '\t {}'.format(",\t".join(tmp))
                st.LOGGING_SESSION.info(tmp)
            st.LOGGING_SESSION.info('}')
            # endregion newEvents
            
            # region eventIncludedToReport
            event_report = list(filter(lambda y: y['isIncludedToReport'], event_summary))
            if len(event_report) > 0:
                st.LOGGING_SESSION.info(
                        f'{self._step}.1 isIncludedToReport [totalEvents: {len(event_report)}]:' + ' {')
                for event in event_report:
                    strip = event['strips'][0]
                    
                    win = (self.convert_timestamp_to_epoch_time(strip['stop']) -
                           self.convert_timestamp_to_epoch_time(strip['start']))
                    win = int(win / df.MILLISECOND)
                    
                    st.LOGGING_SESSION.info(
                            f'\t{event["type"]:17}: '
                            f'- [{win}s]'
                            f'- {strip["start"]:32} - {strip["stop"]:32} '
                            f'- avgHr: {strip["avgHr"]:3} bpm '
                            f'- countBeats: {event["countBeats"]:5} '
                            f'- noiseChannels: {strip["noiseChannels"]} '
                            f'- {strip.get("comment", "")}'
                    )
                st.LOGGING_SESSION.info('}')
            # endregion eventIncludedToReport
        
        except (Exception,) as error:
            cf.DEBUG_MODE and st.write_error_log(error, class_name=self.__class__.__name__)
            pass

    # @df.timeit
    def _get_group_events(
            self,
            hes_id:             int
    ) -> List:

        group = list()
        try:
            group = df.get_group_index_event(self.beat_df['EVENT'].to_numpy(), hes_id)
            segments_continuous = df.remove_channel_from_channel_column(self.beat_df['CHANNEL'].to_numpy())
            
            if len(group) > 0:
                group = list(chain.from_iterable(map(
                    lambda x: np.split(x, np.flatnonzero(np.abs(np.diff(segments_continuous[x]) != 0)) + 1),
                    group
                )))
                group = list(filter(lambda x: len(x) > 0, group))

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return group

    def _calculate_heart_rate(
            self,
            index: NDArray
    ) -> Dict:

        heart_rate = None
        try:
            heart_rate = ut.HeartRate(
                beats=self._study_data[index, self._npy_col.beat],
                symbols=self._study_data[index, self._npy_col.beat_type],
                sampling_rate=self.record_config.sampling_rate
            ).process()

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return heart_rate

    def get_noise_channels(
            self,
            epoch_start:    float | int,
            epoch_stop:     float | int,
            channel:        int
    ) -> List:
        
        noise_channels = list()
        try:
            noise_channels = df.get_noise_channels(
                    study_df=self.beat_df,
                    record_config=self.record_config,
                    all_files=self.all_files,
                    epoch_start=epoch_start,
                    epoch_stop=epoch_stop,
                    strip_channel=channel - 1
            )
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return noise_channels

    def generate_comment(
            self,
            event_type:     str = None,
            priority:       str = None,
            index:          NDArray = None
    ) -> str | None:
        
        comment = None
        try:
            if len(index) == 0:
                return comment

            comment = StripComment(
                    beat_df=(self._study_data[index], self._npy_col),
                    event_type=event_type,
                    record_config=self.record_config
            ).process()

            if priority in ['Emergent', 'Urgent']:
                comment = f'[{priority.capitalize()}] {comment}'
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
        return comment
    
    def get_priority_event(
            self,
            event:          Dict,
            index:          NDArray,
    ) -> Dict:
        
        try:
            if len(index) == 0:
                return event

            new_event = deepcopy(event)
            new_event['epoch']          = self._study_data[index, self._npy_col.epoch]
            new_event['beat']           = self._study_data[index, self._npy_col.beat]
            new_event['beat_type']      = self._study_data[index, self._npy_col.beat_type]

            new_event = EventPriority(
                event=new_event,
                sampling_rate=self.record_config.sampling_rate
            ).process()

            if new_event.get('priority', None) in self.SELECT_PRIORITY_EVENTS:
                event['priority']   = new_event['priority']
                event['startStrip'] = new_event['startStrip']
            pass
            
        except (Exception,) as error:
            st.write_error_log(f'{event.get("type")} - {error}', class_name=self.__class__.__name__)
        
        return event
    
    def get_beat_file(
            self,
            file_index: NDArray | pl.Series
    ) -> Dict:
        
        beat_files = dict()
        try:
            for file in np.unique(file_index):
                filename = df.get_filename(self.all_files[file]['path'])
                beat_path = join(self.record_config.record_path, 'airp/beat', filename + '.beat')
                beat_info_path = join(self.record_config.record_path, 'airp/beat', filename + '.json')
                if (
                        df.check_file_exists(beat_path)
                        and df.check_file_exists(beat_info_path)
                ):
                    beat_files[file] = {
                        "beatPath":         beat_path,
                        "beatInfoPath":     beat_info_path
                    }
                
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return beat_files

    def select_strip_by_beat_info(
            self,
            beat_data:      Dict,
            sample_range:   Tuple[float, float]
    ) -> Any:

        status = False
        try:
            start, stop = sample_range

            check_sv_events = list()
            for data in beat_data.values():
                index = np.flatnonzero(np.logical_and(data.beat >= start, data.beat <= stop))
                is_valid_beats = np.all(np.isin(data.beat_types[index], df.VALID_HES_BEAT_TYPE))
                is_sv_beats = np.any(np.isin(
                        data.beat_types[index],
                        [df.HolterBeatTypes.SVE.value, df.HolterBeatTypes.VE.value]
                ))
                check_sv_events.append(is_valid_beats and is_sv_beats)
                pass

            if len(check_sv_events) > 0:
                status = np.count_nonzero(check_sv_events) / len(check_sv_events) >= self.PERCENT_STRIP

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return status
    
    # @df.timeit
    def _get_baseline_event(
            self,
            offset_time:        float = 0.05  # second
    ) -> Dict:
        
        event: Dict = dict()
        try:
            
            # region Get event information
            index = np.flatnonzero(np.logical_and(
                ~df.check_artifact_function(self.beat_df['EVENT'].to_numpy()),
                self.beat_df['BEAT_TYPE'].is_in(df.VALID_HES_BEAT_TYPE).to_numpy()
            ))
            
            if len(index) > 0:
                j = None
                for x in np.split(index, np.flatnonzero(np.diff(index) != 1) + 1):
                    if self.calculate_duration(x) > self.STRIP_LEN:
                        j = x[0]
                        break

                if j is None:
                    index = 0
                else:
                    index = j
            else:
                index = 0
            
            start_study = self.convert_timestamp_to_epoch_time(self.all_files[0]['start'])
            start_strip = self._study_data[index, self._npy_col.epoch] - (offset_time * df.MILLISECOND)
            start_strip = max([start_study, start_strip])
            end_strip = start_strip + self.STRIP_LEN * df.MILLISECOND
            
            index = self.get_index_within_start_stop(
                    start_epoch=start_strip,
                    stop_epoch=end_strip
            )
            
            hr = self._calculate_heart_rate(index)
            channel = self._study_data[index[0], self._npy_col.channel] + 1
            
            noise_channels = self.get_noise_channels(
                    epoch_start=start_strip,
                    epoch_stop=end_strip,
                    channel=channel
            )

            comment = self.generate_comment(index=index, event_type=None)
            # endregion Get event information
            
            # region Summary
            event['id']:                        Any = df.generate_event_id(self.list_event_ids)
            event['startEpoch']:                Any = start_strip
            event['stopEpoch']:                 Any = end_strip
            event['start']:                     Any = self.convert_epoch_time_to_timestamp(start_strip)
            event['stop']:                      Any = self.convert_epoch_time_to_timestamp(end_strip)
            
            event['type']:                      Any = 'BASELINE'
            event['isIncludedToReport']:        Any = True
            event['maxHr']:                     Any = hr['maxHr']
            event['minHr']:                     Any = hr['minHr']
            event['avgHr']:                     Any = hr['avgHr']
            
            event['channel']:                   Any = channel
            event['countBeats']:                Any = len(index)
            event['duration']:                  Any = round((end_strip - start_strip) / df.MILLISECOND, 2)
            event['comment']:                   Any = comment
            
            strip = dict()
            strip['start']:                     Any = event['start']
            strip['stop']:                      Any = event['stop']
            strip['avgHr']:                     Any = event['avgHr']
            strip['channel']:                   Any = event['channel']
            strip['comment']:                   Any = event['comment']
            strip['noiseChannels']:             Any = noise_channels
            event['strips']:                    Any = [strip]
            
            self.list_event_ids.append(event['id'])
            # endregion Summary

        except (Exception,) as error:
            st.write_error_log(f'BASELINE - {error}', class_name=self.__class__.__name__)
        
        return event
    
    # @df.timeit
    def _get_single_event(
            self,
            hes_id: int
    ) -> Any:
        
        event = dict()
        try:
            index = df.get_index_event(self._study_data[:, self._npy_col.event], hes_id)
            if len(index) == 0:
                return

            # region Get beat files
            beat_file_index = np.unique(self._study_data[index, self._npy_col.file_index])
            beat_files = self.get_beat_file(beat_file_index)

            start_strip = 0
            for file_index, beat_file in beat_files.items():
                beat_data = ut.read_beat_file(
                        beat_file['beatPath'],
                        data_info=df.load_json_files(beat_file['beatInfoPath'])
                )
                if beat_data is None:
                    continue

                idx = index[np.flatnonzero(self._study_data[index, self._npy_col.file_index] == file_index)]

                offset = int(self.STRIP_LEN * self.record_config.sampling_rate)
                ranges = self._study_data[:, self._npy_col.beat][idx[:, None]] + [-offset, offset]

                is_contains_sv_events = False
                for i, (start, stop) in enumerate(ranges):
                    is_contains_sv_events = self.select_strip_by_beat_info(
                        beat_data=beat_data,
                        sample_range=(start, stop)
                    )
                    if is_contains_sv_events:
                        start_strip = idx[i]
                        break

                if is_contains_sv_events:
                    break
            
            if start_strip == 0:
                index = index[0]
            else:
                index = start_strip

            start_event = self._study_data[index, self._npy_col.epoch]
            start_strip = start_event - ((self.STRIP_LEN * df.MILLISECOND) / 2)
            stop_strip = start_strip + self.STRIP_LEN * df.MILLISECOND
            # endregion Get beat files

            # region Get event information
            channel = self._study_data[index, self._npy_col.channel] + 1
            
            ind = df.get_index_within_range(
                nums=self._study_data[:, self._npy_col.epoch],
                low=start_strip,
                high=stop_strip
            )
            comment = self.generate_comment(
                index=ind,
                event_type=df.HOLTER_ALL_EVENT_SUMMARIES_INVERT[hes_id]
            )
            
            noise_channels = self.get_noise_channels(
                    epoch_start=start_strip,
                    epoch_stop=stop_strip,
                    channel=channel
            )
            # endregion Get event information
            
            # region Summary
            event: Dict = dict()
            event['id']:                        Any = df.generate_event_id(self.list_event_ids)
            event['startEpoch']:                Any = start_event
            event['stopEpoch']:                 Any = start_event
            event['start']:                     Any = self.convert_epoch_time_to_timestamp(start_event)
            event['stop']:                      Any = self.convert_epoch_time_to_timestamp(start_event)
            
            event['type']:                      Any = df.HOLTER_ALL_EVENT_SUMMARIES_INVERT[hes_id]
            event['isIncludedToReport']:        Any = True
            event['maxHr']:                     Any = df.DEFAULT_HR_VALUE
            event['minHr']:                     Any = df.DEFAULT_HR_VALUE
            event['avgHr']:                     Any = df.DEFAULT_HR_VALUE
            
            event['channel']:                   Any = channel
            event['countBeats']:                Any = 1
            event['duration']:                  Any = round((start_event - start_event) / df.MILLISECOND, 2)
            event['comment']:                   Any = comment
            
            strip = dict()
            strip['start']:                     Any = self.convert_epoch_time_to_timestamp(start_strip)
            strip['stop']:                      Any = self.convert_epoch_time_to_timestamp(stop_strip)
            strip['avgHr']:                     Any = event['avgHr']
            strip['channel']:                   Any = event['channel']
            strip['comment']:                   Any = event['comment']
            strip['noiseChannels']:             Any = noise_channels
            event['strips']:                    Any = [strip]
            
            self.list_event_ids.append(event['id'])
            # endregion Summary

        except (Exception,) as error:
            st.write_error_log(f'{df.HOLTER_ALL_EVENT_SUMMARIES_INVERT[hes_id]} - {error}',
                               class_name=self.__class__.__name__)
        
        return event
    
    def _get_artifact_event(
            self,
            index: NDArray
    ) -> Dict:
        
        event = dict()
        try:
            # region Get event information
            start_event:    float = float(self._study_data[index[0], self._npy_col.epoch])
            stop_event:     float = float(self._study_data[index[-1], self._npy_col.epoch])
            
            comment = self.generate_comment(
                index=index,
                event_type='ARTIFACT'
            )
            # endregion Get event information
            
            # region Summary
            event['id']:                        Any = df.generate_event_id(self.list_event_ids)
            event['startEpoch']:                Any = start_event
            event['stopEpoch']:                 Any = stop_event
            event['start']:                     Any = self.convert_epoch_time_to_timestamp(start_event)
            event['stop']:                      Any = self.convert_epoch_time_to_timestamp(stop_event)
            
            event['type']:                      Any = 'ARTIFACT'
            event['isIncludedToReport']:        Any = False
            event['maxHr']:                     Any = df.DEFAULT_HR_VALUE
            event['minHr']:                     Any = df.DEFAULT_HR_VALUE
            event['avgHr']:                     Any = df.DEFAULT_HR_VALUE
            
            event['channel']:                   Any = self._study_data[index[0], self._npy_col.channel] + 1
            event['countBeats']:                Any = len(index)
            event['duration']:                  Any = round((stop_event - start_event) / df.MILLISECOND, 2)
            event['comment']:                   Any = comment
            
            self.list_event_ids.append(event['id'])
            # endregion Summary

        except (Exception,) as error:
            st.write_error_log(f'ARTIFACT - {error}', class_name=self.__class__.__name__)
        
        return event

    # @df.timeit
    def _get_pause_events(
            self,
            pause_beats: int = 2        # count
    ) -> List:
        
        events = list()
        try:
            index_pause = df.get_index_event(self._study_data[:, self._npy_col.event], df.HOLTER_PAUSE)
            index_pause = index_pause[:, None] + np.arange(pause_beats)
            for (start_index, stop_index) in tqdm(index_pause, desc='PAUSE'):
                event = dict()

                # region Get event information
                start_event:    float = float(self._study_data[start_index, self._npy_col.epoch])
                stop_event:     float = float(self._study_data[stop_index, self._npy_col.epoch])
                pass
                
                strip = dict()
                strip['type']       = 'PAUSE'
                strip['duration']   = round((stop_event - start_event) / df.MILLISECOND, 2)
                
                strip = self.get_priority_event(
                    strip,
                    index=np.arange(start_index, stop_index + 1)
                )
                if 'priority' in strip.keys():
                    event['priority']   = strip['priority']
                    event['startStrip'] = strip['startStrip']

                comment = self.generate_comment(
                    index=np.arange(start_index, stop_index + 1),
                    event_type='PAUSE',
                    priority=event.get('priority', None)
                )
                pass
                # endregion Get event information
                
                # region Summary
                event['id']:                        Any = df.generate_event_id(self.list_event_ids)
                event['startEpoch']:                Any = start_event
                event['stopEpoch']:                 Any = stop_event
                event['start']:                     Any = self.convert_epoch_time_to_timestamp(start_event)
                event['stop']:                      Any = self.convert_epoch_time_to_timestamp(stop_event)
                
                event['type']:                      Any = strip['type']
                event['isIncludedToReport']:        Any = False
                event['maxHr']:                     Any = df.DEFAULT_HR_VALUE
                event['minHr']:                     Any = df.DEFAULT_HR_VALUE
                event['avgHr']:                     Any = df.DEFAULT_HR_VALUE
                
                event['channel']:                   Any = self._study_data[start_index, self._npy_col.channel] + 1
                event['countBeats']:                Any = pause_beats
                event['duration']:                  Any = strip['duration']
                event['comment']:                   Any = comment

                events.append(event)
                self.list_event_ids.append(event['id'])
                # endregion Summary
        
        except (Exception,) as error:
            st.write_error_log(f'PAUSE - {error}', class_name=self.__class__.__name__)
        
        return events
    
    # @df.timeit
    def _get_events(
            self,
            event_type:         str,
            index:              NDArray,
    ) -> Dict:

        event = dict()
        try:
            # region Get event information
            start_event:    float = float(self._study_data[index[0], self._npy_col.epoch])
            stop_event:     float = float(self._study_data[index[-1], self._npy_col.epoch])
            
            hr = None
            if all(x not in event_type for x in df.LIST_TYPE_NOT_CALCULATE_HR):
                hr = self._calculate_heart_rate(index)
                
            if not df.is_beat_event(event_type):
                strip = dict()
                strip['type']               = event_type
                strip['countBeats']         = len(index)
                strip['duration']           = round((stop_event - start_event) / df.MILLISECOND, 2)
                strip['maxHr']:         Any = hr['maxHr'] if hr is not None else df.DEFAULT_HR_VALUE
                strip['minHr']:         Any = hr['minHr'] if hr is not None else df.DEFAULT_HR_VALUE
                strip['avgHr']:         Any = hr['avgHr'] if hr is not None else df.DEFAULT_HR_VALUE

                strip = self.get_priority_event(
                    strip,
                    index=index,
                )
                if 'priority' in strip.keys():
                    event['priority']   = strip['priority']
                    event['startStrip'] = strip['startStrip']
                pass

            comment = self.generate_comment(
                index=index,
                event_type=event_type,
                priority=event.get('priority', None)
            )
            # endregion Get event information

            # region Summary
            event['id']:                        Any = df.generate_event_id(self.list_event_ids)
            event['startEpoch']:                Any = start_event
            event['stopEpoch']:                 Any = stop_event
            event['start']:                     Any = self.convert_epoch_time_to_timestamp(start_event)
            event['stop']:                      Any = self.convert_epoch_time_to_timestamp(stop_event)
            
            event['type']:                      Any = event_type
            event['isIncludedToReport']:        Any = False

            event['maxHr']:                     Any = hr['maxHr'] if hr is not None else df.DEFAULT_HR_VALUE
            event['minHr']:                     Any = hr['minHr'] if hr is not None else df.DEFAULT_HR_VALUE
            event['avgHr']:                     Any = hr['avgHr'] if hr is not None else df.DEFAULT_HR_VALUE
            
            event['channel']:                   Any = self._study_data[index[0], self._npy_col.channel] + 1
            event['countBeats']:                Any = len(index)
            event['duration']:                  Any = round((stop_event - start_event) / df.MILLISECOND, 2)
            event['comment']:                   Any = comment
            # endregion Summary
            pass

        except (Exception,) as error:
            st.write_error_log(f'{event_type} - {error}', class_name=self.__class__.__name__)

        return event

    # @df.timeit
    def _get_all_events(
            self,
    ) -> None:

        try:
            for i, event_type in enumerate(['BASELINE'] + list(df.HOLTER_ALL_EVENT_SUMMARIES.keys())):
                hes_id = df.HOLTER_ALL_EVENT_SUMMARIES.get(event_type, df.DEFAULT_INVALID_VALUE)
                match event_type:
                    case _ if event_type in ['MIN_HR', 'MAX_HR', 'AVB1', 'SINUS']:
                        continue
                    
                    case 'BASELINE':
                        events = [self._get_baseline_event()]
                        
                    case _ if hes_id in [df.HOLTER_SINGLE_SVES, df.HOLTER_SINGLE_VES]:
                        events = [e] if (e := self._get_single_event(hes_id)) is not None else list()
                        pass
                    
                    case 'ARTIFACT':
                        events = list()
                        for group_index in self._get_group_events(hes_id):
                            (event := self._get_artifact_event(index=group_index)) and events.append(event)
                        pass
                        
                    case 'PAUSE':
                        events = e if (e := self._get_pause_events()) is not None else list()
                        pass
                    
                    case _:
                        events = list()
                        for index in self._get_group_events(hes_id):
                            (event := self._get_events(event_type=event_type, index=index)) and events.append(event)
                        pass
                
                len(events) > 0 and self._event_summary.extend(events)
                pass
            pass
            
            self._event_summary = list(filter(lambda x: x is not None, self._event_summary))

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _get_strip_included_to_report(
            self,
    ) -> None:
        
        try:
            strip_function = GetEventStrips(self)
            
            # region BEAT EVENTS
            for event_type in df.HOLTER_BEAT_EVENT.keys():
                if event_type in ['SINGLE_VE', 'SINGLE_SVE', 'VE_RUN', 'SVE_RUN']:
                    continue

                ((y := list(filter(lambda x: x['type'] == event_type, self._event_summary)))
                 and strip_function.select_sv_event_strip(y))
            # endregion BEAT EVENTS

            # region RHYTHM EVENTS
            for event_type, method in [
                ['AFIB',    strip_function.select_atrial_fibrillation_event_strip],
                ['PAUSE',   strip_function.select_pause_event_strip],
                ['TACHY',   strip_function.select_tachycardia_event_strip],
                ['BRADY',   strip_function.select_bradycardia_event_strip],
                ['AVB1',    strip_function.select_bradycardia_event_strip],
                ['AVB2',    strip_function.select_bradycardia_event_strip],
                ['AVB3',    strip_function.select_bradycardia_event_strip],
            ]:
                ((y := list(filter(lambda x: x['type'] == event_type, self._event_summary))) and method(y))
            # endregion RHYTHM EVENTS

            # region FAST RHYTHM
            pass
            for event_type, method in [
                [['SVT', 'SVE_RUN'],    strip_function.select_fast_hr_rhythm_event_strip],
                [['VT', 'VE_RUN'],      strip_function.select_fast_hr_rhythm_event_strip],
            ]:
                ((y := list(filter(lambda x: x['type'] in event_type, self._event_summary))) and method(y))
            # endregion FAST RHYTHM

            pass
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    # @df.timeit
    def _get_capture_strips(
            self
    ) -> None:

        try:
            event_filtered = list(filter(
                    lambda x: x['isIncludedToReport'],
                    self._event_summary
            ))
            event_remains = list(filter(
                    lambda x: not x['isIncludedToReport'],
                    self._event_summary
            ))

            capture_function = GetCaptureStrips(self)
            # region BEAT EVENTS
            for event_type in df.HOLTER_BEAT_EVENT.keys():
                if event_type in ['SINGLE_VE', 'SINGLE_SVE']:
                    continue

                ((y := list(filter(lambda x: x['type'] == event_type, event_filtered)))
                 and capture_function.capture_sv_event_strip(y))
                
            # endregion BEAT EVENTS
            
            # region RHYTHM EVENTS
            for event_type, method in [
                ['AFIB',    capture_function.capture_atrial_fibrillation_event_strip],
                ['PAUSE',   capture_function.capture_pause_event_strip],
                ['TACHY',   capture_function.capture_tachycardia_event_strip],
                ['BRADY',   capture_function.capture_bradycardia_event_strip],
                ['AVB2',    capture_function.capture_avb2_event_strip],
                ['AVB3',    capture_function.capture_avb3_event_strip],
            ]:
                ((y := list(filter(lambda x: x['type'] == event_type, event_filtered))) and method(y))
            # endregion RHYTHM EVENTS

            # region FAST RHYTHM
            for event_type, method in [
                [['SVT', 'SVE_RUN'],    capture_function.capture_fast_hr_rhythm_event_strip],
                [['VT', 'VE_RUN'],      capture_function.capture_fast_hr_rhythm_event_strip],
            ]:
                ((y := list(filter(lambda x: x['type'] in event_type, event_filtered))) and method(y))
            # endregion FAST RHYTHM
            
            self._event_summary = event_remains + event_filtered
            pass

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            
    def _save_data(
            self,
    ) -> None:
        
        try:
            pass
            _ = (
                (x := pl.DataFrame(self._event_summary))
                .with_columns(
                        [
                            pl.col('startEpoch')
                            .alias('start'),
                            
                            pl.col('stopEpoch')
                            .alias('stop')
                        ]
                )
                .select(
                        list(set(x.columns).intersection(set(df.HOLTER_EVENT_DATAFRAME.keys())))
                )
                .with_columns(
                        [
                            pl.lit(df.HolterEventStatusCode.VALID.value)
                            .alias('statusCode'),

                            pl.lit(df.HolterEventSource.AI.value)
                            .alias('source'),
                        ]
                )
                .write_parquet(df.get_path(self.save_file) + df.S3DataStructure.event_df_prefix)
            )
            pass
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
    @df.timeit
    def process(
            self,
            save_data:          bool = True,
            show_log:           bool = True,
    ) -> Any:

        try:
            if 'EVENT' not in self.beat_df.columns:
                st.get_error_exception(
                        'Event column is not available',
                        class_name=self.__class__.__name__
                )
                
            # region Events
            self._get_all_events()
            self._get_strip_included_to_report()
            self._get_capture_strips()
            # endregion Events

            save_data and self._save_data()
            show_log and self.log(self._event_summary)

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return self._event_summary


class GetEventStrips:
    def __init__(
            self,
            var: EcgEventSummary,
    ):
        self.var = var

    def _get_level_priority(
            self,
            priority: str = None
    ) -> int:

        level = len(self.var.SELECT_PRIORITY_EVENTS) + 1
        try:
            if priority is None:
                return level

            if priority in self.var.SELECT_PRIORITY_EVENTS:
                return self.var.SELECT_PRIORITY_EVENTS.index(priority)

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return level

    def select_sv_event_strip(
            self,
            events: List[Dict]
    ) -> None:

        try:
            if len(events) == 1:
                events[0]['isIncludedToReport'] = True
                return
            
            # region Get beat files
            check_sv_events = False
            for event in events:
                index = self.var.get_index_within_start_stop(
                        start_epoch=event['startEpoch'],
                        stop_epoch=event['stopEpoch']
                )
                if len(index) == 0:
                    continue
                
                file_indexes = self.var.beat_df['FILE_INDEX'][index].unique().to_numpy()
                beat_files = self.var.get_beat_file(file_indexes)
                for file_index, beat_file in beat_files.items():
                    beat_data = ut.read_beat_file(
                            beat_file['beatPath'],
                            data_info=df.load_json_files(beat_file['beatInfoPath'])
                    )
                    if beat_data is None:
                        continue

                    idx = index[np.flatnonzero(self.var.beat_df['FILE_INDEX'][index].to_numpy() == file_index)]

                    offset = int(self.var.STRIP_LEN * self.var.record_config.sampling_rate)
                    ranges = self.var.beat_df['BEAT'].to_numpy()[idx[:, None]] + [-offset, offset]

                    for i, (start, stop) in enumerate(ranges):
                        check_sv_events = self.var.select_strip_by_beat_info(
                            beat_data=beat_data,
                            sample_range=(start, stop)
                        )
                        if check_sv_events:
                            index = idx[i]
                            break

                    if check_sv_events:
                        break
                
                if check_sv_events:
                    event['isIncludedToReport'] = True
                    break

            if not check_sv_events:
                events[0]['isIncludedToReport'] = True
            # endregion Get beat files

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def select_atrial_fibrillation_event_strip(
            self,
            events: List[Dict]
    ) -> None:

        try:
            if len(events) == 0:
                return

            # The longest rhythm is the one with the longest duration
            events = sorted(events, key=lambda x: (-x['duration'], -x['startEpoch']))
            events[0]['isIncludedToReport'] = True
            events[0]['addition'] = 'longest'
            
            events = list(filter(lambda x: x['avgHr'] != df.DEFAULT_HR_VALUE, events))
            if len(events) == 0:
                return
            
            events = sorted(events, key=lambda x: (x['isIncludedToReport'], -x['avgHr'], x['startEpoch']))
            events[0]['isIncludedToReport'] = True
            events[0]['addition'] = 'fastest'
            
            events = sorted(events, key=lambda x: (x['isIncludedToReport'], x['avgHr'], x['startEpoch']))
            events[0]['isIncludedToReport'] = True
            events[0]['addition'] = 'slowest'
            pass

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def select_pause_event_strip(
            self,
            events: List[Dict]
    ) -> None:

        try:
            if len(events) == 0:
                return None

            # The longest rhythm is the one with the longest duration
            events = sorted(events, key=lambda event: [-event['duration']])
            events[0]['isIncludedToReport'] = True

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def select_tachycardia_event_strip(
            self,
            events: List[Dict]
    ) -> None:

        try:
            if len(events) == 0:
                return

            # The fastest rhythm is the one with the highest average heart rate
            events = sorted(events, key=lambda x: (-x['avgHr'], x['startEpoch']))
            events[0]['isIncludedToReport'] = True

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def select_bradycardia_event_strip(
            self,
            events: List[Dict]
    ) -> None:

        try:
            if len(events) == 0:
                return

            # The fastest rhythm is the one with the slowest average heart rate
            events = sorted(events, key=lambda x: (x['avgHr'], x['startEpoch']))
            events[0]['isIncludedToReport'] = True

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    def select_fast_hr_rhythm_event_strip(
            self,
            events: List[Dict]
    ) -> None:

        try:
            if len(events) == 0:
                return
            
            pass
            # The fastest rhythm is the one with the highest average heart rate
            events = sorted(events, key=lambda x: [-x['avgHr'], x['startEpoch']])
            events[0]['isIncludedToReport'] = True

            # The longest rhythm is the one with the highest average heart rate
            events = sorted(events, key=lambda x: [-x['countBeats'], x['startEpoch']])
            events[0]['isIncludedToReport'] = True
            pass

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)


class GetCaptureStrips:
    def __init__(
            self,
            var: Any
    ) -> None:
        self.var = var

    def _get_strip(
            self,
            event:          Dict,
            start_strip:    float,
            strip_win:      int = 1,
    ) -> Dict:
    
        strip = dict()
        try:

            stop_strip = start_strip + strip_win * self.var.STRIP_LEN * df.MILLISECOND

            index = df.get_index_within_range(
                self.var.beat_df['EPOCH'],
                low=start_strip,
                high=stop_strip
            )

            comment = self.var.generate_comment(
                index=index,
                event_type=event['type'],
                priority=event.get('priority', None)
            )
            
            strip['start']:                     Any = self.var.convert_epoch_time_to_timestamp(float(start_strip))
            strip['stop']:                      Any = self.var.convert_epoch_time_to_timestamp(stop_strip)
            strip['avgHr']:                     Any = event['avgHr']
            strip['channel']:                   Any = event['channel']
            strip['comment']:                   Any = comment
            
            noise_channels = self.var.get_noise_channels(
                    epoch_start=start_strip,
                    epoch_stop=stop_strip,
                    channel=strip['channel']
            )
            
            strip['noiseChannels']:             Any = noise_channels
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return strip

    def _get_default_strip(
            self,
            event: List[Dict],
    ) -> List[Dict]:
        try:
            list(map(
                lambda x: x.update({'strips': [self._get_strip(x, x['startEpoch'], 1)]}),
                event
            ))
            
        except (Exception,) as error:
            st.write_error_log(f'{event[0]["type"]} - {error}', class_name=self.__class__.__name__)
        
        return event

    def _capture_fastest_hr(
            self,
            event: Dict
    ) -> None:

        try:
            if 'startStrip' in event.keys():
                start_strip = event['startStrip']
            else:
                start_strip = event['startEpoch'] - self.var.OFFSET_EPOCH

            ind = df.get_index_within_range(
                nums=self.var.beat_df['EPOCH'],
                low=event['startEpoch'],
                high=event['stopEpoch']
            )
            if len(ind) > 0:
                heart_rate_values = ut.HeartRate(
                    beats=self.var.beat_df['BEAT'],
                    symbols=self.var.beat_df['BEAT_TYPE'],
                    sampling_rate=self.var.record_config.sampling_rate
                ).process_strips(index=ind)
                
                if len(heart_rate_values) <= 1:
                    start_strip = self.var.get_value('EPOCH', ind[0])
                else:
                    i = np.argmax(heart_rate_values[:, -1])
                    start_strip = self.var.get_value('EPOCH', heart_rate_values[i, 0])

                start_strip = start_strip - self.var.OFFSET_EPOCH

            event.update(
                {
                    'strips': [self._get_strip(event, start_strip, strip_win=1)]
                }
            )

        except (Exception,) as error:
            st.write_error_log(f'{event["type"]} - {error}', class_name=self.__class__.__name__)

    def _capture_slowest_hr(
            self,
            event: Dict
    ) -> None:

        try:
            if 'startStrip' in event.keys():
                start_strip = event['startStrip']
            else:
                start_strip = event['startEpoch'] - self.var.OFFSET_EPOCH

            ind = df.get_index_within_range(
                nums=self.var.beat_df['EPOCH'],
                low=event['startEpoch'],
                high=event['stopEpoch']
            )
            if len(ind) > 0:
                heart_rate_values = ut.HeartRate(
                    beats=self.var.beat_df['BEAT'],
                    symbols=self.var.beat_df['BEAT_TYPE'],
                    sampling_rate=self.var.record_config.sampling_rate
                ).process_strips(index=ind)
                
                if len(heart_rate_values) <= 1:
                    start_strip = self.var.get_value('EPOCH', ind[0])
                else:
                    i = np.argmin(heart_rate_values[:, -1])
                    start_strip = self.var.get_value('EPOCH', heart_rate_values[i, 0])
                
                start_strip = start_strip - self.var.OFFSET_EPOCH

            event.update({
                'strips': [self._get_strip(event, start_strip, strip_win=1)]
            })

        except (Exception,) as error:
            st.write_error_log(f'{event["type"]} - {error}', class_name=self.__class__.__name__)

    def capture_sv_event_strip(
            self,
            events: List[Dict]
    ) -> List[Dict]:

        events = self._get_default_strip(events)
        try:
            for event in events:
                if event['duration'] <= self.var.STRIP_LEN:
                    start_strip = event['startEpoch'] - (self.var.STRIP_LEN * df.MILLISECOND) // 2
                else:
                    start_strip = event['startEpoch'] - self.var.OFFSET_EPOCH
                event['strips'] = [self._get_strip(event, start_strip)]

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return events
    
    def capture_pause_event_strip(
            self,
            events:             List,
            # thr_long_pause:      float = 7.5 * df.MILLISECOND  # second
    ) -> Any:

        events = self._get_default_strip(events)
        try:
            for event in events:
                event_duration = event['duration'] * df.MILLISECOND

                # region Get start strip pause
                strip_win = 1
                strip_len_ms = self.var.STRIP_LEN * df.MILLISECOND
                if event_duration < strip_len_ms:
                    center_pause = event['startEpoch'] + event_duration / 2
                    start_strip = center_pause - strip_len_ms / 2

                elif strip_len_ms <= event_duration <= df.MAX_NUM_STRIPS_PER_EVENT * strip_len_ms:
                    strip_win = max(1, min(int(np.ceil(event_duration / strip_len_ms)), df.MAX_NUM_STRIPS_PER_EVENT))
                    start_strip = event['startEpoch'] - self.var.OFFSET_EPOCH
                    
                # elif strip_len_ms <= event_duration:
                #     strip_win = max(1, min(int(np.ceil(event_duration / strip_len_ms)), df.MAX_NUM_STRIPS_PER_EVENT))
                #     strip_win *= strip_len_ms / df.MILLISECOND
                #     start_strip = event['startEpoch'] - self.var.OFFSET_EPOCH

                else:
                    center_pause = event['startEpoch'] + event_duration / 2
                    start_strip = center_pause - strip_len_ms / 2

                event['strips'] = [self._get_strip(event, start_strip,  strip_win)]

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    def capture_atrial_fibrillation_event_strip(
            self,
            events: List[Dict]
    ) -> None:

        try:
            for event in events:
                addition = event.get('addition')

                match addition:
                    case 'longest':
                        start_strip = event['startEpoch'] - self.var.ONSET_EPOCH
                        event.update(
                            {
                                'strips': [self._get_strip(event, start_strip, strip_win=1)]
                            }
                        )
                        pass

                    case 'fastest':
                        self._capture_fastest_hr(event)

                    case 'slowest':
                        self._capture_slowest_hr(event)
                        pass

                    case _:
                        start_strip = event['startEpoch'] - self.var.OFFSET_EPOCH
                        event.update({
                            'strips': [self._get_strip(event, start_strip, strip_win=1)]
                        })
                        pass

                event.pop('addition', None)

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    def capture_tachycardia_event_strip(
            self,
            events: List[Dict]
    ) -> None:

        try:
            for event in events:
                self._capture_fastest_hr(event)
                pass

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
    def capture_bradycardia_event_strip(
            self,
            events: List[Dict]
    ) -> None:
        try:
            for event in events:
                self._capture_slowest_hr(event)

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
    def capture_avb2_event_strip(
            self,
            events: List[Dict]
    ) -> None:

        try:
            for event in events:
                self._capture_slowest_hr(event)
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
    def capture_avb3_event_strip(
            self,
            events: List[Dict]
    ) -> None:

        try:
            for event in events:
                self._capture_slowest_hr(event)

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
    def capture_fast_hr_rhythm_event_strip(
            self,
            events: List[Dict]
    ) -> None:

        events = self._get_default_strip(events)
        try:
            for event in events:
                if event['duration'] < self.var.STRIP_LEN:
                    strip_win = 1
                    start_strip = event['startEpoch'] - (self.var.STRIP_LEN * df.MILLISECOND) // 2
                else:
                    strip_win = min(int(np.ceil(event['duration'] // self.var.STRIP_LEN)),
                                    self.var.MAX_NUM_STRIPS_PER_EVENT)
                    start_strip = event['startEpoch'] - self.var.OFFSET_EPOCH
                    
                event['strips'] = [self._get_strip(event, start_strip, strip_win)]

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
