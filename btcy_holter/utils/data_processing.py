from btcy_holter import *


# Size	    Signed Values	                                                Unsigned Values
# 1-byte	-128 to 127	                                                    0 to 255
# 2-byte	-32,768 to 32,767	                                            0 to 65,535
# 3-byte	-8,388,608 to 8,388,607	                                        0 to 16,777,215
# 4-byte	-2,147,483,648 to 2,147,483,647	                                0 to 4,294,967,295
# 5-byte	-549,755,813,888 to 549,755,813,887	                            0 to 1,099,511,627,775
# 6-byte	-140,737,488,355,328 to 140,737,488,355,327	                    0 to 281,474,976,710,655
# 7-byte	-36,028,797,018,963,968 to 36,028,797,018,963,967	            0 to 72,057,594,037,927,935
# 8-byte	-9,223,372,036,854,775,808 to 9,223,372,036,854,775,807	        0 to 18,446,744,073,709,551,615


SAMPLE_DTYPE:               Final[Any] = int


BEAT_BYTE_FORMAT:           Final[Dict] = {
    'beat_type':    1,
    'beat_sample':  3,
}


FINAL_BEAT_BYTE_FORMAT:     Final[Dict] = {
    'version_1': {
        'beat_type':        1,
        'channel':          1,
        'beat_sample':      3,
        'rr_heatmap':       2,
    },
    'version_2': {
        'beat_type':        1,
        'channel':          1,
        'beat_sample':      3,
        'rr_heatmap':       2,
        'pvc_morphology':   1,
    }
}


def get_data_from_dat(
        file:           str,
        record_config:  sr.RecordConfigurations,
) -> NDArray | None:
    
    ecg_signals = None
    try:
        if df.check_all_variables_values(record_config):
            return ecg_signals
            
        raw_signals = np.fromfile(file, dtype=np.int16)
        raw_signals = raw_signals / record_config.gain
    
        length = len(raw_signals) // record_config.number_of_channels
        
        ecg_signals = raw_signals[: length * record_config.number_of_channels]
        ecg_signals = ecg_signals.reshape((-1, record_config.number_of_channels), order='C')
        
    except (Exception, ) as error:
        st.get_error_exception(error)
        
    return ecg_signals


def sync_data_type(
        data: Any
) -> int:
    
    return SAMPLE_DTYPE(data)


def get_data_format_type_by_filename(
        filename: str
) -> str:
    data_format_type = 'version_1'
    try:
        filename = df.get_filename(filename)
        filename = filename.split('-')[-1]
        if filename in FINAL_BEAT_BYTE_FORMAT.keys():
            data_format_type = filename
            
    except (Exception, ) as error:
        st.write_error_log(error)
    
    return data_format_type


def write_beat_file(
        beat_file:      str,
        beat_types:     NDArray | List,
        beat_samples:   NDArray | List,
        extension:      str = 'beat',
) -> bool:
    
    status = False
    try:

        beat_types = df.convert_to_array(beat_types)
        beat_samples = df.convert_to_array(beat_samples)
        
        if any(len(x) == 0 for x in [beat_samples, beat_types]):
            return status
        
        if len(beat_samples) != len(beat_types):
            return status
        
        if np.any(np.isin(beat_types, list(df.HES_TO_SYMBOL.values()))):
            beat_types = df.convert_symbol_to_hes_beat(beat_types)
        
        beat_samples = np.array(beat_samples)
        beat_types = np.array(beat_types)
        
        beat_file = join(dirname(beat_file), df.get_filename(beat_file) + f'.{extension}')
        with open(beat_file, 'wb') as file:
            for t, s in zip(beat_types, beat_samples):
                # 1 byte for beat type
                file.write(sync_data_type(t).to_bytes(BEAT_BYTE_FORMAT['beat_type']))
                
                # 3 bytes for beat samples
                file.write(sync_data_type(s).to_bytes(BEAT_BYTE_FORMAT['beat_sample'], byteorder='big'))
        
        file.close()
        status = True
        
    except (Exception,) as error:
        status = False
        st.write_error_log(error=f'{basename(beat_file)} - {error}')
    
    return status
    
    
def write_final_beat_file(
        beat_file:          str,
        beat_types:         NDArray | List | pl.Series,
        beat_samples:       NDArray | List | pl.Series,
        beat_channels:      NDArray | List | pl.Series,
        rr_heat_maps:       NDArray | List | pl.Series,
        pvc_morphology:     NDArray | List | pl.Series  = None,
        extension:          str                         = 'beat'
) -> str:
    
    try:
        beat_types      = df.convert_to_array(beat_types)
        beat_samples    = df.convert_to_array(beat_samples)
        beat_channels   = df.convert_to_array(beat_channels)
        rr_heat_maps    = df.convert_to_array(rr_heat_maps)
        
        data_format_type = 'version_1'
        if pvc_morphology is not None:
            pvc_morphology = df.convert_to_array(pvc_morphology)
            data_format_type = 'version_2'
        
        if any(len(x) == 0 for x in [beat_samples, beat_types, beat_channels, rr_heat_maps]):
            return beat_file
        
        if len(beat_samples) != len(beat_types) != len(beat_channels) != len(rr_heat_maps):
            return beat_file
        
        if np.any(np.isin(beat_types, list(df.HES_TO_SYMBOL.values()))):
            beat_types = df.convert_symbol_to_hes_beat(beat_types)
        
        data_format = FINAL_BEAT_BYTE_FORMAT[data_format_type]
        beat_file = join(dirname(beat_file), df.get_filename(beat_file) + f'-{data_format_type}.{extension}')
        with open(beat_file, 'wb') as file:
            for i, (c, t, s, rr) in enumerate(zip(beat_channels, beat_types, beat_samples, rr_heat_maps)):
                # 1 byte for channel
                file.write(sync_data_type(c).to_bytes(data_format['channel']))
                
                # 1 byte for beat type
                file.write(sync_data_type(t).to_bytes(data_format['beat_type']))

                # 3 bytes for beat samples
                file.write(sync_data_type(s).to_bytes(data_format['beat_sample'], byteorder='big'))
                
                # 2 bytes for heatmap cell ids
                file.write(sync_data_type(rr).to_bytes(data_format['rr_heatmap'], byteorder='big'))
                
                # 1 byte for pvc morphology
                if pvc_morphology is not None:
                    file.write(sync_data_type(pvc_morphology[i]).to_bytes(data_format['pvc_morphology']))
                pass
            
        file.close()
        status = True
        pass
    
    except (Exception,) as error:
        st.write_error_log(error=f'{basename(beat_file)} - {error}')
    
    return beat_file


def read_beat_file(
        input_file: str,
        data_info:  Dict = None
) -> Dict:
    
    data_channels = dict()
    try:
        if not df.check_file_exists(input_file):
            return data_channels
        
        if data_info is None:
            return data_channels
        
        # region Load data
        beat_samples = list()
        beats_types = list()
        with open(input_file, 'rb') as file:
            while True:
                # Read 1 byte for beat type
                type_bytes = file.read(BEAT_BYTE_FORMAT['beat_type'])
                if not type_bytes:
                    break
                
                beats_types.append(int.from_bytes(type_bytes))
                
                # Read 3 bytes for beat samples
                beat_samples.append(int.from_bytes(file.read(BEAT_BYTE_FORMAT['beat_sample']), byteorder='big'))
        
        beat_samples = np.array(beat_samples)
        beats_types = np.array(beats_types)
        # endregion Load data

        # region Summaries data
        start_sample = 0
        for method, data in data_info.items():
            if method == 'totalChannels':
                continue
                
            for channel, data_length in enumerate(data):
                key = f'{method}_{channel}'
                data = sr.AINumPyResult()
                data.beat = beat_samples[start_sample: start_sample + data_length]
                data.beat_types = beats_types[start_sample: start_sample + data_length]
                data_channels[key] = deepcopy(data)
                start_sample += data_length
        pass
        # endregion Summaries data
        
    except (Exception,) as error:
        st.write_error_log(error=f'{basename(input_file)} - {error}')
    
    return data_channels


def read_final_beat_file(
        input_file: str
) -> sr.AINumPyResult:
    
    data = sr.AINumPyResult()
    try:
        if not df.check_file_exists(input_file):
            return data
        
        data_format_type = get_data_format_type_by_filename(input_file)
        data_format = FINAL_BEAT_BYTE_FORMAT[data_format_type]
        
        with open(input_file, 'rb') as file:
            while True:
                # Read 1 byte for channel
                channel_bytes = file.read(data_format['channel'])
                if not channel_bytes:
                    break
                
                data.beat_channels.append(int.from_bytes(channel_bytes))
                
                # Read 1 byte for beat type
                data.beat_types.append(int.from_bytes(file.read(data_format['beat_type'])))

                # Read 3 bytes for beat samples
                data.beat.append(int.from_bytes(file.read(data_format['beat_sample']), byteorder='big'))
                
                # Read 2 bytes for heatmap cell ids
                data.rr_cell_ids.append(int.from_bytes(file.read(data_format['rr_heatmap']), byteorder='big'))
                
                if 'pvc_morphology' in data_format.keys():
                    # Read 1 byte for pvc morphology
                    data.pvc_morphology.append(int.from_bytes(file.read(data_format['pvc_morphology'])))
        
        if len(data.beat) > 0:
            data.beat = np.array(data.beat)
        
        if len(data.beat_types) > 0:
            data.beat_types = np.array(data.beat_types)
        
        if len(data.beat_channels) > 0:
            data.beat_channels  = np.array(data.beat_channels)
        
        if len(data.rr_cell_ids) > 0:
            data.rr_cell_ids = np.array(data.rr_cell_ids)
        
        if len(data.pvc_morphology) > 0:
            data.pvc_morphology = np.array(data.pvc_morphology)
        
    except (Exception,) as error:
        st.write_error_log(error=f'{basename(input_file)} - {error}')
    
    return data
