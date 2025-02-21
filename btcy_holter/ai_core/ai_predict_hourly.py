from btcy_holter import *
from btcy_holter.ai_core.core import *
from btcy_holter.ai_core.post_processing import *


class HourlyPrediction:
    MAIN_METHOD:                Final[str] = 'AI'
    # SUB_METHODS:                Final[List] = ['PAN-TOMPKINS']
    SUB_METHODS:                Final[List] = []
    BEAT_EXTENSION:             Final[str] = '.beat'
    
    CONFIG_DURATION_SEGMENTS:   Final[int]      = 1     # minutes
    LOWPASS_FILTER:             Final[float]    = 40.0  # Hz
    HIGHPASS_FILTER:            Final[float]    = 0.5   # Hz
    
    AMP_LOWPASS_FILTER:         Final[float]    = 30    # Hz
    AMP_HIGHPASS_FILTER:        Final[float]    = 2     # Hz
    
    def __init__(
            self,
            record_config:      sr.RecordConfigurations     = None,
            algorithm_config:   sr.AlgorithmConfigurations  = None,
            save_data:          bool                        = True,
            **kwargs
    ) -> None:
        try:
            # region Most important and common variables
            self.save_data:             Final[bool]                         = save_data
            self.record_config:         Final[sr.RecordConfigurations]      = record_config
            self.algorithm_config:      Final[sr.AlgorithmConfigurations]   = algorithm_config
            # endregion Most important and common variables
            
            # region Grouped by functionality
            self._save_data_path:        Final[str] = join(dirname(self.record_config.record_path), 'airp')
            self._save_beat_path:        Final[str] = join(self._save_data_path, 'beat')
            self._save_final_beat_path:  Final[str] = join(self._save_data_path, 'final-beat')
            self.__save_beat_data_path:    Final[str] = join(self._save_data_path, 'npy')
            self.__initial_saved_path()
            # endregion Grouped by functionality
            
            # region Optional or default variables
            self.noise_id:              Final[int] = cf.RHYTHMS_DATASTORE['classes']['OTHER']
            self.noise_criteria:        Final[int] = df.CRITERIA['OTHER']['duration']
            
            self.log_process_time:      Final[bool] = kwargs.get('log_process_time', False)
            self.log_step_time:         Final[bool] = kwargs.get('log_step_time', True)
            # endregion Optional or default variables
            
            # region Private variables
            self._min_rr_interval:      Final[float] = cf.MIN_RR_INTERVAL
            self._filename:             Final[str] = df.get_filename(self.record_config.record_path)
            
            self._time_tracking:        Dict = dict()
            self._square_wave_indexes:  List = list()
            
            self._data_channels:        Dict = dict()
            # endregion Private variables
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
    
    def __log_performance(
            self,
            log_performance_time:   Dict,
            title:                  str = ''
    ) -> None:
        if log_performance_time is None or not bool(log_performance_time):
            return
        
        keys = f'{title.lower()}Performance'
        is_new_log = keys not in self._time_tracking.keys()
        
        if is_new_log:
            log = dict()
        else:
            log = deepcopy(self._time_tracking[keys])
        
        if is_new_log:
            log['CPU'] = log_performance_time['cpu']
        else:
            log['CPU'] += log_performance_time['cpu']
        
        if is_new_log:
            log['GPU'] = log_performance_time['gpu']
        else:
            log['GPU'] += log_performance_time['gpu']
        
        if is_new_log:
            log['GPUStep'] = log_performance_time['gpu/step']
        else:
            log['GPUStep'] = (log['GPUStep'] + log_performance_time['gpu/step']) / 2
        
        if is_new_log:
            log['Step'] = log_performance_time['step']
        else:
            log['Step'] += log_performance_time['step']
        
        self._time_tracking[keys] = deepcopy(log)
    
    def __log(
            self,
            hourly_data:    Dict,
            time_start:     float = None,
            **kwargs
    ) -> None:
        
        try:
            try:
                total_file_index = kwargs['total_file_index']
            except (Exception,):
                total_file_index = None
            
            log_time_step = ''
            if self.log_step_time:
                for x in self._time_tracking.keys():
                    if 'performance' not in x.lower():
                        log_time_step += f'[{x}: {self._time_tracking[x]:.3f}] '
            log_time = f'[{df.get_time_process(time_start):2.3f}] {log_time_step[:-1]}'
            
            file_index = self.record_config.record_file_index
            if total_file_index is not None:
                tmp_str = '{' + f':{len(str(total_file_index))}' + '}/' + '{' + f':{len(str(total_file_index))}' + '}'
                file_index = tmp_str.format(self.record_config.record_file_index, total_file_index)

            total_channel = list(map(str, np.unique(hourly_data['hourlyData']['CHANNEL'])))
            channel_hourly = f'[CHN: {"-".join(total_channel):5}]'
            
            st.LOGGING_SESSION.info(
                    f'+ [FILE: {file_index}] [FS: {self.record_config.sampling_rate}Hz] {channel_hourly}'
                    f'->{log_time} - {basename(self._filename)} - {len(hourly_data["hourlyData"]["CHANNEL"]):4} beats.'
            )
        
        except (Exception,) as error:
            cf.DEBUG_MODE and st.write_error_log(error)
    
    # @df.timeit
    def __update_time_tracking(
            self,
            start:  float,
            key:    str
    ) -> None:
        try:
            if key not in self._time_tracking.keys():
                self._time_tracking[key] = df.get_time_process(start)
            else:
                self._time_tracking[key] += df.get_time_process(start)
        
        except (Exception,) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
    
    # @df.timeit
    def __initial_saved_path(
            self
    ) -> None:
        try:
            list(map(
                lambda x: os.makedirs(x, exist_ok=True),
                [self._save_data_path, self._save_beat_path, self._save_final_beat_path, self.__save_beat_data_path]
            ))

        except (Exception,) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
    
    # @df.timeit
    def __save_data_path(
            self,
            hourly_data: Dict,
    ) -> str | None:
        
        save_path = None
        try:
            if not bool(hourly_data):
                st.write_error_log('Hourly data is empty.', class_name=self.__class__.__name__)

            save_path = join(self.__save_beat_data_path, self._filename + '.parquet')
            if self.save_data:
                dataframe = pl.DataFrame(hourly_data)
                if dataframe.height == 0:
                    st.write_error_log('Hourly data is empty.', class_name=self.__class__.__name__)

                dataframe.write_parquet(save_path)
        
        except (Exception,) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
        
        return save_path
    
    # @df.timeit
    def __update_start_stop_record(
            self,
            len_record: int
    ) -> None:
        
        try:
            
            if all(isinstance(x, str) for x in [self.record_config.record_start_time,
                                                self.record_config.record_stop_time]):

                self.record_config.record_start_time = df.convert_timestamp_to_epoch_time(
                        timestamp=self.record_config.record_start_time,
                        timezone=self.record_config.timezone,
                        dtype=float,
                        ms=True
                )
                
                self.record_config.record_stop_time = df.convert_timestamp_to_epoch_time(
                        timestamp=self.record_config.record_stop_time,
                        timezone=self.record_config.timezone,
                        dtype=float,
                        ms=True
                )

            else:
                self.record_config.record_start_time = 0
                self.record_config.record_stop_time = (len_record / self.record_config.sampling_rate) * df.MILLISECOND
        
        except (Exception,) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
    
    # @df.timeit
    def __detect_square_wave_region(
            self,
            ecg_signal: NDArray
    ) -> None:
        
        try:
            if len(ecg_signal.T) <= 1:
                return
            
            subtract_channel = np.abs(ecg_signal[:, df.FIRST_SIGNAL_CHANNEL] - ecg_signal[:, 1])
            subtract_channel2 = np.abs(ecg_signal[:, df.FIRST_SIGNAL_CHANNEL] - ecg_signal[:, -1])
            inx_zeros = np.flatnonzero(np.logical_or(subtract_channel > 0.0001, subtract_channel2 > 0.0001))
            inx_zeros = np.concatenate((np.array([0]), inx_zeros, np.array([len(subtract_channel) - 1])))
            
            inx_seg_zeros = np.flatnonzero(np.diff(inx_zeros) >= self.record_config.sampling_rate)
            self._square_wave_indexes.extend(list(map(
                    lambda x: [inx_zeros[x], inx_zeros[x + 1]],
                    inx_seg_zeros
            )))
            
            self._square_wave_indexes = np.asarray(self._square_wave_indexes)
        
        except (Exception,) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
    
    # @df.timeit
    def __remove_beat_in_square_wave(
            self,
            predict: sr.AIPredictionResult
    ) -> sr.AIPredictionResult:

        try:
            if len(self._square_wave_indexes) == 0:
                return predict

            ind_square = np.hstack(list(map(
                    lambda x: np.arange(*x),
                    self._square_wave_indexes
            )))

            ind_delete = np.flatnonzero(np.in1d(predict.beat, ind_square))
            predict.beat = np.delete(predict.beat, ind_delete)
            predict.symbol = np.delete(predict.symbol, ind_delete)
        
        except (Exception,) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
        
        return predict
    
    # @df.timeit
    def _combine_noise_and_rhythm(
            self,
            noise:      NDArray,
            rhythms:    NDArray,
            sig_len:    int,
            other_id:   Dict
    ) -> NDArray:
        
        try:
            ind = np.flatnonzero(noise == cf.NOISE_DATASTORE['classes']['NOISE'])
            ind = ind[np.flatnonzero(ind < sig_len)]
            if len(ind) == 0:
                return rhythms
            
            group_noise = np.split(ind, np.flatnonzero(np.abs(np.diff(ind)) != 1) + 1)
            group_noise = list(map(lambda x: x[[0, -1]], df.filter_null_list_in_group(group_noise)))
            group_noise = (np.array(group_noise) / self.record_config.sampling_rate).astype(int)
            group_noise[:, 1] += 1
            
            index_noise = np.hstack(list(map(lambda x: np.arange(*x), group_noise)))
            index_noise = index_noise[index_noise < len(rhythms)]
            index_noise = np.sort(np.unique(index_noise)).astype(int)
            
            rhythms[index_noise] = other_id['OTHER']
        
        except (Exception,) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
        
        return rhythms
    
    # @df.timeit
    def _get_ecg_signals(
            self
    ) -> [NDArray | None]:
        
        ecg_signal = None
        try:
            if df.check_hea_file(self.record_config.record_path):
                ecg_signal = wf.rdrecord(df.get_path(self.record_config.record_path)).p_signal
                ecg_signal = np.nan_to_num(ecg_signal)
            
            else:
                ecg_signal = ut.get_data_from_dat(
                        file=self.record_config.record_path,
                        record_config=self.record_config
                )
            
            if ecg_signal is None or len(ecg_signal) == 0:
                st.write_error_log('Wrong data.', class_name=self.__class__.__name__)
            
            self.__detect_square_wave_region(ecg_signal)
            self.__update_start_stop_record(len_record=len(ecg_signal))
        
        except (Exception,) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
        
        return ecg_signal
    
    # @df.timeit
    def _combine_rhythm_and_beat(
            self,
            data:           sr.AIPredictionResult,
            count_beats:    int = df.LIMIT_BEAT_SAMPLE_IN_SIGNAL
    ) -> sr.AIPredictionResult:
        
        try:
            if len(data.beat) == 0:
                return data
            
            # region Create 2 beats no beat region
            group_beats = list()
            samples = list()
            
            ind_noise = np.flatnonzero(data.rhythm == self.noise_id)
            if len(ind_noise) > 0:
                group = np.split(ind_noise, np.flatnonzero(np.diff(ind_noise) != 1) + 1)
                group = list(filter(lambda x: len(x) >= self.noise_criteria, group))
                group = np.array(df.filter_null_list_in_group(list(map(
                    lambda x: x[[0, -1]] * self.record_config.sampling_rate,
                    group
                ))))
                
                cnt = np.array([len(df.get_indices_within_range(data.beat, x[0], x[-1])) for x in group])
                group_beats = group[np.flatnonzero(cnt < count_beats)]
                samples = list(map(
                        lambda x: df.get_indices_within_range(data.beat, start=x[0], stop=x[-1]),
                        df.filter_null_list_in_group(group)
                ))
                samples = list(filter(lambda x: len(x) > 0, samples))
            # endregion Create 2 beats no beat region
            
            # region Map beats
            rhythm_bk = deepcopy(data.rhythm)
            frame = (data.rhythm[:, None] + np.zeros(self.record_config.sampling_rate, dtype=int)).flatten()
            valid_index = np.flatnonzero(data.beat < len(frame))
            
            data.rhythm = frame[data.beat[valid_index]]
            if len(data.symbol) - len(data.rhythm) > 0:
                data.rhythm = np.concatenate((data.rhythm, [data.rhythm[-1]] * (len(data.symbol) - len(data.rhythm))))
            # endregion Map beats
            
            # region Create 2 beats no beat region
            offset = int(self._min_rr_interval * self.record_config.sampling_rate)
            data.beat = np.array(data.beat)
            for start, end in group_beats:
                ind = np.flatnonzero(data.beat <= start)
                samples_begin = 0 if len(ind) == 0 else data.beat[ind[-1]]
                samples_begin = max(0, int(samples_begin + offset))
                
                ind = np.flatnonzero(data.beat >= end)
                samples_end = len(data.rhythm) * self.record_config.sampling_rate \
                    if len(ind) == 0 else data.beat[ind[0]]
                samples_end = min(int(samples_end - offset), len(data.ecg_signal))
                
                data.beat = np.concatenate((data.beat, np.array([samples_begin, samples_end])))
                data.symbol = np.concatenate((data.symbol, [df.HolterSymbols.MARKED.value] * count_beats))
                data.rhythm = np.concatenate((data.rhythm, [self.noise_id] * count_beats))
                
                ind_sort = np.argsort(data.beat)
                data.beat   = data.beat[ind_sort]
                data.symbol = data.symbol[ind_sort]
                data.rhythm = data.rhythm[ind_sort]
            
            if len(samples) > 0:
                b_end = data.beat[0] / self.record_config.sampling_rate
                s_end = samples[0][-1]
                if (data.rhythm[0] == self.noise_id
                        and b_end >= self.noise_criteria // 3
                        and s_end >= self.noise_criteria):
                    
                    data.beat = np.concatenate((np.array([offset]), data.beat))
                    data.symbol = np.concatenate((np.array([df.HolterSymbols.MARKED.value]), data.symbol))
                    data.rhythm = np.concatenate((np.array([self.noise_id]), data.rhythm))
                
                b_end = len(rhythm_bk) - data.beat[-1] / self.record_config.sampling_rate
                s_end = len(rhythm_bk) - samples[-1][0]
                if (data.rhythm[-1] == self.noise_id
                        and b_end >= self.noise_criteria // 3
                        and s_end >= self.noise_criteria):
                    data.beat = np.concatenate((data.beat, [len(data.ecg_signal) - offset]))
                    data.symbol = np.concatenate((data.symbol, [df.HolterSymbols.MARKED.value]))
                    data.rhythm = np.concatenate((data.rhythm, [self.noise_id]))
            # endregion Create 2 beats no beat region
        
        except (Exception, ) as error:
            st.write_error_log(error=error, class_name=self.__class__.__name__)
        
        return data
    
    # @df.timeit
    def _merge_beats(
            self,
            data: sr.AIPredictionResult
    ) -> sr.AIPredictionResult:
        
        try:
            data = self._combine_rhythm_and_beat(data)
            
            ind_valid = np.flatnonzero(
                    np.logical_and(
                            data.beat > 0,
                            data.beat < len(data.ecg_signal)
                    )
            )
            data.beat = data.beat[ind_valid]
            data.symbol = data.symbol[ind_valid]
            data.rhythm = data.rhythm[ind_valid]
        
        except (Exception,) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
        
        return data
    
    # @df.timeit
    def _validate_data(
            self,
            data: sr.AIPredictionResult,
    ) -> sr.AIPredictionResult:
        try:
            if len(data.ecg_signal) == 0:
                return data

            # region Validate beat
            if len(data.beat) > 0:
                # 40 is the distance checked by T wave detection
                offset = df.THRESHOLD_T_WAVE_DISTANCE * self.record_config.sampling_rate
                check = np.logical_and(
                        data.beat >= offset,
                        data.beat <= len(data.ecg_signal) - offset
                )
                index = np.flatnonzero(~check)
                check_noise_beats = np.flatnonzero(
                        np.in1d(
                                data.symbol[index],
                                [df.HolterSymbols.OTHER.value, df.HolterSymbols.MARKED.value]
                        )
                )
                if len(check_noise_beats) > 0:
                    check[index[check_noise_beats]] = True
                
                ind = np.flatnonzero(check)
                data.beat = data.beat[ind]
                data.symbol = data.symbol[ind]
                data.rhythm = data.rhythm[ind]
                
                if len(data.beat) >= df.LIMIT_BEAT_SAMPLE_IN_SIGNAL and data.beat[0] == 0:
                    data.beat = data.beat[1:]
                    data.symbol = data.symbol[1:]
                    data.rhythm = data.rhythm[1:]
            # endregion Validate beat
            
            # region Create two beat if there is no beat
            if len(data.beat) < df.LIMIT_BEAT_SAMPLE_IN_SIGNAL:
                data.beat = df.initialize_two_beats_at_ecg_data(
                        len_ecg=len(data.ecg_signal),
                        sampling_rate=self.record_config.sampling_rate
                ).astype(int)
                data.symbol         = np.array([df.HolterSymbols.MARKED.value] * len(data.beat))
                data.rhythm         = np.zeros_like(data.beat).astype(int)
                data.beat_channel   = np.ones(len(data.beat), dtype=int)
            # endregion Create two beat if there is no beat
            
            # region Invalid
            else:
                ind = np.flatnonzero(np.logical_and(data.beat > 0, data.beat < len(data.ecg_signal)))
                data.beat = data.beat[ind]
                
                ind_sorted = np.argsort(data.beat)
                data.beat           = data.beat[ind_sorted].astype(int)
                data.symbol         = data.symbol[ind][ind_sorted]
                data.rhythm         = data.rhythm[ind][ind_sorted].astype(int)
                data.beat_channel   = data.beat_channel[data.beat].astype(int)
            # endregion Invalid
            pass
        
        except (Exception, ) as error:
            st.write_error_log(error=error, class_name=self.__class__.__name__)
        
        return data
    
    # @df.timeit
    def _tf_channel(
            self,
            ecg_signal: NDArray,
            **kwargs
    ) -> NDArray:
        
        aggregate_beat_channels = np.ones(len(ecg_signal[:, df.FIRST_SIGNAL_CHANNEL]), dtype=int)
        try:
            start = time.monotonic()
            
            # region Params
            try:
                valid_channel = kwargs['valid_channel']
            except (Exception,):
                valid_channel = True
            
            try:
                config_segment = kwargs['config_duration_segments']
            except (Exception,):
                config_segment = self.CONFIG_DURATION_SEGMENTS
            # endregion Params
            
            # region Process
            match self.record_config.channel:
                case _ if self.record_config.number_of_channels == df.MIN_NUMBER_OF_CHANNELS:
                    # For data with a channel
                    aggregate_beat_channels *= df.FIRST_SIGNAL_CHANNEL
                
                case _ if self.record_config.channel is not None:
                    # For data with multiple channels and channel
                    aggregate_beat_channels = aggregate_beat_channels * self.record_config.channel
                
                case _ if config_segment is not None:
                    # For data with multiple channels and hourly data
                    channel_func = ChannelDetection(
                            is_process_event=False,
                            log_process_time=self.log_process_time
                    )
                    aggregate_beat_channels = channel_func.prediction(
                            ecg_signal=ecg_signal,
                            ecg_signal_fs=self.record_config.sampling_rate,
                            valid_channel=valid_channel,
                            config_segment=config_segment * df.SECOND_IN_MINUTE
                    )
                    self.__log_performance(
                            log_performance_time=channel_func.log_performance_time,
                            title='channel'
                    )
                
                case _:
                    # For data with multiple channels
                    channel_func = ChannelDetection(
                            is_process_event=False,
                            log_process_time=self.log_process_time
                    )
                    channel_suggestion = channel_func.prediction(
                            ecg_signal=ecg_signal,
                            ecg_signal_fs=self.record_config.sampling_rate,
                            valid_channel=valid_channel
                    )
                    aggregate_beat_channels *= channel_suggestion
                    
                    self.__log_performance(
                            log_performance_time=channel_func.log_performance_time,
                            title='channel'
                    )
            # endregion Process
            
            self.__update_time_tracking(start=start, key='CHANNEL')
        
        except (Exception,) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
        
        return aggregate_beat_channels
    
    # @df.timeit
    def _tf_beat(
            self,
            ecg_signal: NDArray,
    ) -> [NDArray, NDArray]:
        
        beats = np.array([], dtype=int)
        symbols = np.array([], dtype=int)
        try:
            # region beat classification & detection
            start = time.monotonic()
            beat_prediction = BeatsDetectionAndClassification(
                    is_process_event=False,
                    log_process_time=self.log_process_time
            )
            beats, symbols = beat_prediction.prediction(
                    ecg_signal=ecg_signal,
                    ecg_signal_fs=self.record_config.sampling_rate
            )
            
            self.__update_time_tracking(start=start, key='BEAT')
            # endregion beat classification & detection
            
            # region HES algorithm
            if self.algorithm_config.run_hes_classification and len(beats) > 0:
                start = time.monotonic()
                beat_ies_function = BeatClassificationByIES()
                symbols = beat_ies_function.process(
                        ecg_signal=ecg_signal,
                        ecg_signal_fs=self.record_config.sampling_rate,
                        samples=beats,
                        symbols=deepcopy(symbols)
                )
                self.__update_time_tracking(start=start, key='BEAT-HES')
            # endregion HES algorithm
            
            self.__log_performance(
                    log_performance_time=beat_prediction.log_performance_time,
                    title='beat'
            )
        
        except (Exception,) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
        
        return beats, symbols
    
    # @df.timeit
    def tf_beat_pan_tompkins(
            self,
            ecg_signal: NDArray,
    ) -> [NDArray, NDArray]:
        
        beats = np.array([], dtype=int)
        symbols = np.array([], dtype=int)
        try:
            start = time.monotonic()
            beat_prediction = BeatDetectionClassificationPantompkins(
                    ecg_signal=ecg_signal,
                    ecg_signal_fs=self.record_config.sampling_rate
            )
            beats, symbols = beat_prediction.predict()
            
            ind = np.flatnonzero(beats > 0)
            if len(ind) > 0:
                beats = beats[ind]
                symbols = symbols[ind]
                
            self.__update_time_tracking(start=start, key='PTK')
        
        except (Exception,) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
        
        return beats, symbols
    
    # @df.timeit
    def _tf_rhythm(
            self,
            ecg_signal: NDArray,
    ) -> NDArray:
        
        rhythms = np.array([], dtype=int)
        try:
            start = time.monotonic()
            
            # region rhythm classification
            rhythms_func = RhythmClassification(
                    is_process_event=False,
                    log_process_time=self.log_process_time
            )
            
            rhythms = rhythms_func.prediction(
                    ecg_signal=ecg_signal,
                    ecg_signal_fs=self.record_config.sampling_rate
            )
            # endregion rhythm classification
            
            # region noise detection
            if self.algorithm_config.run_noise_detection:
                noise_prediction = NoiseDetection(
                        is_process_event=False,
                        log_process_time=self.log_process_time
                )
                
                noise = noise_prediction.prediction(
                        ecg_signal=ecg_signal,
                        ecg_signal_fs=self.record_config.sampling_rate,
                )
                
                rhythms = self._combine_noise_and_rhythm(
                        noise=noise,
                        rhythms=rhythms,
                        sig_len=len(ecg_signal),
                        other_id=rhythms_func.datastore['classes']
                )
            # endregion noise detection
            
            # region Remove AVB1
            ind = np.flatnonzero(rhythms == rhythms_func.datastore['classes']['AVB1'])
            if len(ind) > 0:
                rhythms[ind] = rhythms_func.datastore['classes']['SINUS']
            # endregion Remove AVB1
            
            self.__update_time_tracking(start=start, key='RHYTHM')
            self.__log_performance(log_performance_time=rhythms_func.log_performance_time, title='rhythm')
        
        except (Exception, ) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
        
        return rhythms
    
    # @df.timeit
    def _process_single_channel(
            self,
            predict: sr.AIPredictionResult,
    ) -> sr.AIPredictionResult:
        
        predict.beat = np.array([], dtype=int)
        predict.symbol = np.array([], dtype=int)
        predict.rhythm = np.array([], dtype=int)
        try:
            predict.beat, predict.symbol = self._tf_beat(
                    ecg_signal=predict.ecg_signal,
            )
            
            predict.rhythm = self._tf_rhythm(
                    ecg_signal=predict.ecg_signal,
            )
            
            predict = self.__remove_beat_in_square_wave(predict)
            predict = self._merge_beats(predict)
            predict = self._validate_data(predict)
        
        except (Exception,) as error:
            st.write_error_log(error=error, class_name=self.__class__.__name__)
        
        return predict
    
    # @df.timeit
    def _post_process_hes(
            self,
            data_structure: sr.AIPredictionResult
    ) -> sr.AINumPyResult:
        
        data = sr.AINumPyResult()
        try:
            data_check = deepcopy(data_structure)
            data_check.ecg_signal = None
            pass
            be_events = al.BEDetection(
                    data_structure=data_check,
                    record_config=self.record_config,
                    algorithm_config=self.algorithm_config,
                    is_hes_process=False,
            ).process()
            
            data.epoch = df.generate_epoch_from_samples(
                    samples=data_structure.beat,
                    start_time=self.record_config.record_start_time,
                    sampling_rate=self.record_config.sampling_rate
            )
            
            data.beat             = data_structure.beat
            data.beat_types       = df.convert_symbol_to_hes_beat(data_structure.symbol)
            data.events           = np.zeros_like(data.beat)
            data.beat_channels    = data_structure.beat_channel
            
            for rhythm, hes in df.RHYTHMS_TO_HES.items():
                index = np.flatnonzero(data_structure.rhythm == rhythm)
                if len(index) > 0:
                    data.events[index] = hes
            
            data.events |= be_events
            pass
        
        except (Exception,) as error:
            st.write_error_log(error=error, class_name=self.__class__.__name__)
        
        return data
    
    # @df.timeit
    def _post_process(
            self,
            predict: sr.AIPredictionResult
    ) -> sr.AINumPyResult:
        
        data = sr.AINumPyResult()
        try:
            # region Post-Processing Beats and Rhythms
            start = time.monotonic()
            post_process_data = PostProcessingBeatsAndRhythms(
                    is_process_event=False,
                    data_channel=predict,
                    record_config=self.record_config
            ).process()
            self.__update_time_tracking(start=start, key='PP')
            # endregion Post-Processing Beats and Rhythms
            
            # region Post Processing HES
            start = time.monotonic()
            data = self._post_process_hes(
                    data_structure=post_process_data
            )
            self.__update_time_tracking(start=start, key='R2H')
            # endregion Post Processing HES
            pass
        
        except (Exception, ) as error:
            st.write_error_log(error=error, class_name=self.__class__.__name__)
        
        return data
    
    # @df.timeit
    def tf_beats_and_rhythms(
            self,
            ecg_signals: NDArray
    ) -> Dict:
        
        predicts = dict()
        try:
            for channel in list(range(self.record_config.number_of_channels)):
                data = sr.AIPredictionResult()
                data.sampling_rate = self.record_config.sampling_rate
                data.channel = channel
                data.ecg_signal = ecg_signals[:, channel].copy()
                data.beat_channel = np.ones(len(data.ecg_signal), dtype=int) * channel
                
                data = self._process_single_channel(
                        predict=data
                )
                predicts[channel] = data
        
        except (Exception, ) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
        
        return predicts
    
    # @df.timeit
    def run_sub_method_and_save_beat_file(
            self,
            ecg_signals:    NDArray,
            main_methods:   Dict
    ) -> Any:
        
        record_info = dict()
        beat_path = join(self._save_beat_path, self._filename + self.BEAT_EXTENSION)
        
        try:
            beat_samples = list()
            beat_symbols = list()
            
            record_info['totalChannels'] = self.record_config.number_of_channels
            record_info[self.MAIN_METHOD] = list()
            for channel in list(range(self.record_config.number_of_channels)):
                beat_samples.extend(main_methods[channel].beat)
                symbols = df.convert_symbol_to_hes_beat(main_methods[channel].symbol)
                beat_symbols.extend(symbols)
                record_info[self.MAIN_METHOD].append(len(main_methods[channel].beat))
            
            for method in self.SUB_METHODS:
                record_info[method] = list()
                for channel in list(range(self.record_config.number_of_channels)):
                    match method:
                        case 'PAN-TOMPKINS':
                            beat_sample, beat_symbol = self.tf_beat_pan_tompkins(ecg_signals[:, channel])
                            beat_symbol = df.convert_symbol_to_hes_beat(beat_symbol)
                        
                        case _:
                            continue
                    
                    beat_samples.extend(beat_sample)
                    beat_symbols.extend(beat_symbol)
                    record_info[method].append(len(beat_sample))
            
            ut.write_beat_file(
                    beat_file=beat_path,
                    beat_types=np.array(beat_symbols),
                    beat_samples=np.array(beat_samples)
            )
            pass
            
            df.write_json_files(
                    data_path=join(self._save_beat_path, self._filename + '.json'),
                    data=record_info
            )
        
        except (Exception, ) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
        
        return record_info, beat_path
    
    # @df.timeit
    def process_final_data_channel(
            self,
            ecg_signals:    NDArray,
            data_channels:  Dict,
            **kwargs
    ) -> Any:
        
        hourly_channel = None
        pqst_data = np.array([], dtype=int)
        
        final_data = sr.AIPredictionResult()
        final_data.sampling_rate = self.record_config.sampling_rate
        
        try:
            rr_threshold: Final[float] = self._min_rr_interval * self.record_config.sampling_rate
            
            beat_channel = self._tf_channel(
                    ecg_signal=ecg_signals,
                    **kwargs
            )
            hourly_channel = df.find_most_frequency_occurring_values(beat_channel)
            
            group_index = np.split(np.arange(len(beat_channel)), np.flatnonzero(np.diff(beat_channel) != 0) + 1)
            for index in df.filter_null_list_in_group(group_index):
                channel = beat_channel[index[0]]
                
                ind = df.get_indices_within_range(
                        arr=data_channels[channel].beat,
                        start=index[0],
                        stop=index[-1]
                )
                if len(final_data.beat) > 0:
                    ind = ind[np.flatnonzero(data_channels[channel].beat[ind] >= final_data.beat[-1] + rr_threshold)]
                
                final_data.beat     = np.concatenate((final_data.beat, data_channels[channel].beat[ind]))
                final_data.rhythm   = np.concatenate((final_data.rhythm, data_channels[channel].rhythm[ind]))
                final_data.symbol   = np.concatenate((final_data.symbol, data_channels[channel].symbol[ind]))
                
                if self.algorithm_config.run_pqst_detection:
                    pqst = al.PQST(
                            data_structure=data_channels[channel],
                            return_data_format='array'
                    ).process()
                    pqst_data = df.append(pqst_data, pqst.astype(int)[ind])
                    
            final_data.beat          = final_data.beat.astype(int)
            final_data.rhythm        = final_data.rhythm.astype(int)
            final_data.beat_channel  = beat_channel[final_data.beat]
            final_data.channel       = hourly_channel

            ecg = ecg_signals[np.arange(len(ecg_signals))[:, None], beat_channel[:, None]].flatten()
            final_data.ecg_signal    = ecg
            
            raw_beats = deepcopy(final_data.beat)
            final_data = self._post_process(final_data)
            ind = np.flatnonzero(~np.in1d(raw_beats, final_data.beat))
            if len(ind) > 0:
                pqst_data = np.delete(pqst_data, ind, axis=0)
            pass
        
        except (Exception, ) as error:
            st.write_error_log(error=error, class_name=self.__class__.__name__)
        
        return hourly_channel, final_data, pqst_data
    
    # @df.timeit
    def save_parquet_file(
            self,
            final_data: sr.AINumPyResult,
            pqst_data:  NDArray,
    ) -> Dict:
        
        study_data_dict = dict()
        try:
            # region Hourly data
            ind_time_valid = df.get_index_within_range(
                    nums=final_data.epoch,
                    low=self.record_config.record_start_time,
                    high=self.record_config.record_stop_time
            )
            if len(ind_time_valid) == 0:
                st.get_error_exception('No valid data.', class_name=self.__class__.__name__)

            hourly_dict = {
                'EPOCH':        final_data.epoch[ind_time_valid],
                'CHANNEL':      final_data.beat_channels[ind_time_valid],
                'BEAT':         final_data.beat[ind_time_valid],
                'BEAT_TYPE':    final_data.beat_types[ind_time_valid],
                'EVENT':        final_data.events[ind_time_valid]
            }
            # endregion Hourly data

            # region PQST data
            schema = ['QT', 'QTC', 'ST_LEVEL', 'ST_SLOPE', 'P_ONSET', 'P_PEAK', 'P_OFFSET', 'P_AMPLITUDE',
                      'T_ONSET', 'T_PEAK', 'T_OFFSET', 'T_AMPLITUDE', 'QRS_ONSET', 'QRS_OFFSET']
            try:
                pqst_data = pqst_data[ind_time_valid]
            except (Exception,):
                pqst_data = np.zeros((len(ind_time_valid), len(schema)), dtype=int)

            for i, key in enumerate(schema):
                hourly_dict[key] = pqst_data[:, i]
            # endregion PQST data
            
            # region Numpy
            hourly_dict['FILE_INDEX'] = np.ones(len(ind_time_valid), dtype=int) * self.record_config.record_file_index
            study_data_dict['dataPath']     = self.__save_data_path(hourly_dict)
            study_data_dict['hourlyData']  = hourly_dict
            # endregion Numpy
        
        except (Exception,) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
        
        return study_data_dict
    
    # @df.timeit
    def process(
            self,
            **kwargs
    ) -> Dict:
        
        summaries = dict()
        try:
            start_total = time.monotonic()
            
            try:
                show_log = kwargs['show_log']
            except (Exception,):
                show_log = True
            
            # region Prediction
            ecg_signals = self._get_ecg_signals()
            
            data_channels = self.tf_beats_and_rhythms(
                    ecg_signals=ecg_signals
            )
            
            beat_info, beat_path = self.run_sub_method_and_save_beat_file(
                    ecg_signals=ecg_signals,
                    main_methods=data_channels
            )
            
            hourly_channel, final_data, pqst_data = self.process_final_data_channel(
                    ecg_signals=ecg_signals,
                    data_channels=data_channels,
                    **kwargs
            )
            # endregion Prediction
            
            # region Save + Response
            hourly_data_info = self.save_parquet_file(
                    final_data=final_data,
                    pqst_data=pqst_data
            )
            
            summaries = dict()
            summaries['id'] = self.record_config.record_id
            summaries['channel']            = hourly_channel + 1
            summaries['beatInfo']           = beat_info
            summaries['beatPath']           = beat_path
            summaries['dataPath']           = hourly_data_info['dataPath']
            summaries['hourlyData']         = hourly_data_info['hourlyData']
            # endregion Save + Response
            
            show_log and self.__log(hourly_data_info, start_total, **kwargs)
        
        except (Exception,) as error:
            st.write_error_log(f'{self._filename}-{error}', class_name=self.__class__.__name__)
        
        return summaries


# @df.timeit
def run_hourly(
        record_config:      sr.RecordConfigurations     = None,
        algorithm_config:   sr.AlgorithmConfigurations  = None,
        **kwargs
) -> Dict | None:
    
    """
       Executes the ECG signal processing pipeline with optional configurations.

       This function orchestrates the ECG signal processing by initializing the
       AIPrediction class with the provided configurations and executing the
       processing flow based on the `run_full_flow` flag. It supports handling
       of both full processing and specific beats and rhythms analysis.

       Parameters:
       - record_config (sr.RecordConfigurations): Configuration for the ECG record,
         including path, channel settings, and other metadata.
       - algorithm_config (sr.AlgorithmConfigurations): Configuration for the
         algorithm to be used in processing, including settings for artifact
         detection, PQST analysis, etc.
       - run_full_flow (bool): Flag to determine whether to run the full processing
         pipeline or only beats and rhythms analysis. Defaults to True.
       - **kwargs: Additional keyword arguments that can be passed to the processing
         functions for further customization.

       Returns:
       - Dict | None: The results of the processing as a dictionary, or None if
         the processing could not be completed due to errors or invalid configurations.

       Raises:
       - Exception: Captures and logs any exceptions that occur during the processing,
         including file not found errors, invalid configurations, and others.
    """
    
    results = None
    try:
        if df.check_all_variables_values(record_config):
            return
        
        if not isinstance(record_config.record_path, str):
            return
        
        if not df.check_file_exists(record_config.record_path):
            return
        
        if not any(df.get_extension_file(record_config.record_path) == x for x in ['.hea', '.dat']):
            return
        
        if algorithm_config is None or not df.check_all_variables_values(algorithm_config):
            func = sr.AlgorithmConfigurations()
            algorithm_config = func.get_default_configurations()
        
        ai_funcs = HourlyPrediction(
                record_config=record_config,
                algorithm_config=algorithm_config,
                log_process_time=kwargs.get('log_process_time', False),
                log_step_time=kwargs.get('log_step_time', True),
                save_data=kwargs.get('save_data', True),
        )
        results = ai_funcs.process(**kwargs)
    
    except (Exception,) as error:
        st.write_error_log(f"{record_config.record_path} - {error}")
    
    return results


def run_hourly_data_dict_dataframe(
        hourly_data_dict: Dict,
) -> Dict | None:
    results = None
    try:
        if bool(hourly_data_dict) == 0:
            return results

        _ = (
            pl.DataFrame(hourly_data_dict['hourlyData'])
            .sort(
                [
                    'EPOCH',
                    'CHANNEL',
                    'BEAT',
                    'BEAT_TYPE',
                    'EVENT',
                    'QT',
                    'QTC',
                    'ST_LEVEL',
                    'ST_SLOPE',
                    'P_ONSET',
                    'P_PEAK',
                    'P_OFFSET',
                    'P_AMPLITUDE',
                    'T_ONSET',
                    'T_PEAK',
                    'T_OFFSET',
                    'T_AMPLITUDE',
                    'QRS_ONSET',
                    'QRS_OFFSET',
                    'FILE_INDEX'
                ]
            )
            .write_parquet(hourly_data_dict['dataPath'])
        )
        results = dict()
        results['dataPath'] = hourly_data_dict['dataPath']

    except (Exception,) as error:
        st.write_error_log(error=error)

    return results
