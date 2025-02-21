from dateutil import parser
from dateutil import tz as date_util_tz

from btcy_holter import *


# # -------------------------------------------------
# region DEFINE: COLOR
CHANNEL_COLORS: Final[Dict] = {
    0:          'silver',
    1:          'green',
    2:          'yellow'
}

BEAT_COLORS: Final[Dict] = {
    'N':        'white',
    'S':        'orange',
    'A':        'orange',
    'V':        'blue',
    '|':        'purple',
    'Others':   'purple',
    'Q':        'purple',
    'R':        'cyan',
    'M':        'olive'
}

BEAT_COLORS_EC: Final[Dict] = {
    'NOTABEAT': 'grey',
    'N':        'black',
    'S':        'orange',
    'A':        'orange',
    'V':        'blue',
    '|':        'purple',
    'Others':   'purple',
    'Q':        'purple',
    'R':        'white',
    'M':        'white'
}
# endregion DEFINE: COLOR


# # -------------------------------------------------
# region DEFINE: THRESHOLD
THRESHOLD_T_WAVE_DISTANCE:                          Final[float]    = 0.16              # seconds
THRESHOLD_MIN_ECG_SIGNAL_LENGTH:                    Final[int]      = 10                # seconds

THRESHOLD_MIN_PAUSE_RR_INTERVALS:                   Final[float]    = 2450              # ms
THRESHOLD_MAX_PAUSE_RR_INTERVALS:                   Final[float]    = 25000             # ms

LIMIT_BEAT_TO_CALCULATE_HRV:                        Final[int]      = 2000000

# endregion DEFINE: THRESHOLD


# # -------------------------------------------------
# region DEFINE: VARIABLES
DATA_VERSION:                                       Final[int]      = 2
DEFAULT_HR_VALUE:                                   Final[int]      = -1                 # bpm
DEFAULT_INVALID_VALUE:                              Final[int]      = -1
DEFAULT_COUNT_VALUE:                                Final[int]      = 0

LIMIT_BEAT_CALCULATE_HR:                            Final[int]      = 2                  # beats

EPOCH_TIME_START_SERVICE:                           Final[int]      = 1577811600
LIMIT_BEAT_SAMPLE_IN_SIGNAL:                        Final[int]      = 2

HES_CONTINUOUS:                                     Final[int]      = 0x0010

## region Heart rate
HR_MIN_THR:                                         Final[int]      = 25                 # bpm
HR_MAX_THR:                                         Final[int]      = 250                # bpm

LIST_TYPE_NOT_CALCULATE_HR:                         Final[List]     = ['SINGLE', 'BIGEMINY', 'TRIGEMINAL', 'QUADRIGEMINY', 'COUPLET']
## endregion Heart rate

MIN_NUMBER_OF_CHANNELS:                             Final[int]      = 1
MAX_NUMBER_OF_CHANNELS:                             Final[int]      = 12
FIRST_SIGNAL_CHANNEL:                               Final[int]      = 0


MILLISECOND:                                        Final[int]      = 1000               # ms
VOLT_TO_MV:                                         Final[int]      = 1000               # mV

HOUR_IN_DAY:                                        Final[int]      = 24
SECOND_IN_MINUTE:                                   Final[int]      = 60
MINUTE_IN_HOUR:                                     Final[int]      = 60

MILLISECOND_IN_MINUTE:                              Final[int]      = MILLISECOND * SECOND_IN_MINUTE
MILLISECOND_IN_HOUR:                                Final[int]      = MILLISECOND * SECOND_IN_MINUTE * MINUTE_IN_HOUR

MINUTE_IN_DAY:                                      Final[int]      = HOUR_IN_DAY * MINUTE_IN_HOUR
SECOND_IN_HOUR:                                     Final[int]      = SECOND_IN_MINUTE * MINUTE_IN_HOUR
SECOND_IN_DAY:                                      Final[int]      = SECOND_IN_MINUTE * MINUTE_IN_DAY

MIN_STRIP_LEN:                                      Final[int]      = 10                     # second
MAX_STRIP_LEN:                                      Final[int]      = 30                     # second
MAX_NUM_STRIPS_PER_EVENT:                           Final[int]      = int(MAX_STRIP_LEN / MIN_STRIP_LEN)

HOLTER_EVENT_DATAFRAME:                             Final[Dict]      = {
    'id':                                           pl.String,
    'start':                                        pl.UInt64,
    'stop':                                         pl.UInt64,
    'type':                                         pl.Utf8,
    'isIncludedToReport':                           pl.Boolean,
    'duration':                                     pl.Float32,
    'maxHr':                                        pl.UInt8,
    'minHr':                                        pl.UInt8,
    'avgHr':                                        pl.UInt8,
    'countBeats':                                   pl.UInt32,
    'statusCode':                                   pl.Int8,
    'source':                                       pl.String
}

HOLTER_PVC_DATAFRAME:                             Final[Dict]      = {
    'ID':                                           pl.Int8,
    'CENTER_VECTOR':                                pl.Array,
    'IS_INCLUDED_TO_REPORT':                        pl.Boolean,
    'UNCLASSIFIED':                                 pl.Boolean
}
# endregion DEFINE: VARIABLES


# # -------------------------------------------------
# region DEFINE: HES ID
class HolterFilenames(
        Enum
):
    ALL_HOURLY_DATA:                                Final[str] = 'allHourlyData.json'
    HOLTER_REPORT:                                  Final[str] = 'holterReport.json'
    

class HolterEventStatusCode(
        Enum
):
    INVALID:                                        Final[int] = 0
    VALID:                                          Final[int] = 1

    
class HolterEventSource(
        Enum
):
    AI:                                             Final[str] = 'AI'
    USER:                                           Final[str] = 'USER'


class HolterSymbols(
        Enum
):
    N:                                              Final[str] = 'N'
    SVE:                                            Final[str] = 'S'
    VE:                                             Final[str] = 'V'
    OTHER:                                          Final[str] = '|'
    IVCD:                                           Final[str] = 'R'
    MARKED:                                         Final[str] = 'M'
    
    
class HolterBeatTypes(
        Enum
):
    N:                                              Final[int] = 1
    VE:                                             Final[int] = 60
    SVE:                                            Final[int] = 70
    OTHER:                                          Final[int] = 80
    IVCD:                                           Final[int] = 90
    MARKED:                                         Final[int] = 100
    

HOLTER_SINUS_RHYTHM:                                Final[int] = 0
HOLTER_SINGLE_VES:                                  Final[int] = 1 << 0  # 1
HOLTER_VES_RUN:                                     Final[int] = 1 << 1  # 2
HOLTER_VES_COUPLET:                                 Final[int] = 1 << 2  # 4
HOLTER_VES_BIGEMINY:                                Final[int] = 1 << 3  # 8
HOLTER_VES_TRIGEMINAL:                              Final[int] = 1 << 4  # 16
HOLTER_VES_QUADRIGEMINY:                            Final[int] = 1 << 28  # 268435456

HOLTER_SINGLE_SVES:                                 Final[int] = 1 << 5  # 32
HOLTER_SVES_RUN:                                    Final[int] = 1 << 6  # 64
HOLTER_SVES_COUPLET:                                Final[int] = 1 << 7  # 128
HOLTER_SVES_BIGEMINY:                               Final[int] = 1 << 8  # 256
HOLTER_SVES_TRIGEMINAL:                             Final[int] = 1 << 9  # 512
HOLTER_SVES_QUADRIGEMINY:                           Final[int] = 1 << 26  # 67108864

HOLTER_BRADY:                                       Final[int] = 1 << 10  # 1024
HOLTER_TACHY:                                       Final[int] = 1 << 11  # 2048
HOLTER_MAX_HR:                                      Final[int] = 1 << 12  # 4096        # Disable
HOLTER_MIN_HR:                                      Final[int] = 1 << 13  # 8192        # Disable
HOLTER_LONG_RR:                                     Final[int] = 1 << 14  # 16384       # Disable
HOLTER_PAUSE:                                       Final[int] = 1 << 15  # 32768
HOLTER_AFIB:                                        Final[int] = 1 << 16  # 65536
HOLTER_ARTIFACT:                                    Final[int] = 1 << 17  # 131072
HOLTER_USER_EVENTS:                                 Final[int] = 1 << 18  # 262144      # Disable
HOLTER_SINUS_ARRHYTHMIA:                            Final[int] = 1 << 19  # 524288      # Disable
HOLTER_AV_BLOCK_3:                                  Final[int] = 1 << 20  # 1048576
HOLTER_AV_BLOCK_2:                                  Final[int] = 1 << 21  # 2097152
HOLTER_AV_BLOCK_1:                                  Final[int] = 1 << 22  # 4194304
HOLTER_SVT:                                         Final[int] = 1 << 23  # 8388608
HOLTER_VT:                                          Final[int] = 1 << 24  # 16777216

# endregion DEFINE: HES ID


# # -------------------------------------------------
# region DEFINE: STRIP INCLUDE IN REPORT CRITERIA
__root_path = dirname(abspath(__file__))
__rule_path = f'{__root_path}/ai_core/rule/'
with open(__rule_path + 'criteria_rhythms.json', 'r') as file:
    CRITERIA = json.load(file)
file.close()

with open(__rule_path + 'criteria_strip_hourly.json', 'r') as file:
    HOURLY_CRITERIA = json.load(file)
file.close()

with open(__rule_path + 'criteria_strip_event.json', 'r') as file:
    EVENT_CRITERIA = json.load(file)
file.close()

STRIP_AFIB_CRITERIA:                                    Final[Dict] = HOURLY_CRITERIA.get('AFIB', dict())
STRIP_AVB2_CRITERIA:                                    Final[Dict] = HOURLY_CRITERIA.get('AVB2', dict())
STRIP_AVB3_CRITERIA:                                    Final[Dict] = HOURLY_CRITERIA.get('AVB3', dict())
STRIP_SVT_CRITERIA:                                     Final[Dict] = HOURLY_CRITERIA.get('SVT', dict())
STRIP_VT_CRITERIA:                                      Final[Dict] = HOURLY_CRITERIA.get('VT', dict())
STRIP_BRADY_CRITERIA:                                   Final[Dict] = HOURLY_CRITERIA.get('BRADY', dict())
STRIP_TACHY_CRITERIA:                                   Final[Dict] = HOURLY_CRITERIA.get('TACHY', dict())
STRIP_PAUSE_CRITERIA:                                   Final[Dict] = HOURLY_CRITERIA.get('PAUSE', dict())

EVENT_STRIP_AFIB_CRITERIA:                              Final[Dict] = EVENT_CRITERIA.get('AFIB', dict())
EVENT_STRIP_AVB2_CRITERIA:                              Final[Dict] = EVENT_CRITERIA.get('AVB2', dict())
EVENT_STRIP_AVB3_CRITERIA:                              Final[Dict] = EVENT_CRITERIA.get('AVB3', dict())

EVENT_STRIP_SVT_CRITERIA:                               Final[Dict] = EVENT_CRITERIA.get('SVT', dict())
EVENT_STRIP_SVT_CRITERIA_SYMPTOMATIC:                   Final[Dict] = EVENT_CRITERIA.get('SVT_SYMPTOMATIC', dict())

EVENT_STRIP_VT_CRITERIA:                                Final[Dict] = EVENT_CRITERIA.get('VT', dict())
EVENT_STRIP_VT_CRITERIA_SYMPTOMATIC:                    Final[Dict] = EVENT_CRITERIA.get('VT_SYMPTOMATIC', dict())

EVENT_STRIP_BRADY_CRITERIA:                             Final[Dict] = EVENT_CRITERIA.get('BRADY', dict())

EVENT_STRIP_TACHY_CRITERIA:                             Final[Dict] = EVENT_CRITERIA.get('TACHY', dict())
EVENT_STRIP_TACHY_CRITERIA_SYMPTOMATIC:                 Final[Dict] = EVENT_CRITERIA.get('TACHY_SYMPTOMATIC', dict())

EVENT_STRIP_PAUSE_CRITERIA:                             Final[Dict] = EVENT_CRITERIA.get('PAUSE', dict())
# endregion DEFINE: STRIP INCLUDE IN REPORT CRITERIA


# # -------------------------------------------------
# region STRUCTURE: NUMPY COLUMNS

STUDY_DATA_DICT: Final[Dict] = {
    1: [
        'EPOCH',
        'CHANNEL',
        'BEAT',
        'BEAT_TYPE',
        'EVENT',
        'QTC',
        'ST_LEVEL',
        'ST_SLOPE',
        'QT',
        'T_AMPLITUDE',
        'P_ONSET',
        'QRS_ONSET',
        'QRS_OFFSET',
        'FILE_INDEX',
        'RR_HEATMAP',
        'RR_HEATMAP_REVIEWED',
        'AI_BEAT_TYPE',
        'AI_EVENT',
        'AI_USER_EVENTS',
        'AI_CHECK_EVENTS',
        'EVENT_OFFSET_TIME',
        'CH0_BEAT',
        'CH0_EVENT',
        'CH1_BEAT',
        'CH1_EVENT',
        'CH2_BEAT',
        'CH2_EVENT'
    ],
    2: [
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
        'FILE_INDEX',
        'RR_HEATMAP',
        'RR_HEATMAP_REVIEWED',
        'PVC_TEMPLATE',
        'PVC_TEMPLATE_CORRCOEF',
        'PVC_TEMPLATE_REVIEWED',
    ]
}


class StudyDataColumnStructure:
    def __init__(
            self,
            key: Any
    ):
        try:
            if isinstance(key, list):
                self._study_data = key
                
            elif key in STUDY_DATA_DICT.keys():
                self._get_study_data_column_based_on_version(key)
                
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
    def _get_study_data_column_based_on_version(
            self,
            data_version: str = DATA_VERSION
    ) -> Any:
        self._study_data = STUDY_DATA_DICT[data_version]
            
    def get_col_index(
            self,
            column_name: str,
    ) -> int:
        return self._study_data.index(column_name) if column_name in self._study_data else None
    
    def __call__(
            self,
            *args,
            **kwargs
    ):
        try:
            self.epoch:                 int = self.get_col_index('EPOCH')
            self.channel:               int = self.get_col_index('CHANNEL')
            self.beat:                  int = self.get_col_index('BEAT')
            self.beat_type:             int = self.get_col_index('BEAT_TYPE')
            self.file_index:            int = self.get_col_index('FILE_INDEX')
            self.rr_heatmap:            int = self.get_col_index('RR_HEATMAP')
            self.rr_heatmap_reviewed:   int = self.get_col_index('RR_HEATMAP_REVIEWED')
            
            self.qt:                    int = self.get_col_index('QT')
            self.qtc:                   int = self.get_col_index('QTC')
            self.st_level:              int = self.get_col_index('ST_LEVEL')
            self.st_slope:              int = self.get_col_index('ST_SLOPE')
            
            self.qrs_onset:             int = self.get_col_index('QRS_ONSET')
            self.qrs_offset:            int = self.get_col_index('QRS_OFFSET')

            # v2
            self.t_peak:                int = self.get_col_index('T_PEAK')
            self.t_amp:                 int = self.get_col_index('T_AMPLITUDE')
            self.t_onset:               int = self.get_col_index('T_ONSET')
            self.t_offset:              int = self.get_col_index('T_OFFSET')
            
            self.p_peak:                int = self.get_col_index('P_PEAK')
            self.p_amp:                 int = self.get_col_index('P_AMPLITUDE')
            self.p_onset:               int = self.get_col_index('P_ONSET')
            self.p_offset:              int = self.get_col_index('P_OFFSET')
            
            self.pvc_template:          int = self.get_col_index('PVC_TEMPLATE')
            self.pvc_template_corrcoef: int = self.get_col_index('PVC_TEMPLATE_CORRCOEF')
            self.pvc_template_reviewed: int = self.get_col_index('PVC_TEMPLATE_REVIEWED')
            
            # v1
            self.event:                 int = self.get_col_index('EVENT')
            self.ch0_beat:              int = self.get_col_index('CH0_BEAT')
            self.ch0_event:             int = self.get_col_index('CH0_EVENT')
            self.ch1_beat:              int = self.get_col_index('CH1_BEAT')
            self.ch1_event:             int = self.get_col_index('CH1_EVENT')
            self.ch2_beat:              int = self.get_col_index('CH2_BEAT')
            self.ch2_event:             int = self.get_col_index('CH2_EVENT')
            
            self.ai_beat_type:          int = self.get_col_index('AI_BEAT_TYPE')
            self.ai_event:              int = self.get_col_index('AI_EVENT')
            self.ai_user_events:        int = self.get_col_index('AI_USER_EVENTS')
            self.ai_check_events:       int = self.get_col_index('AI_CHECK_EVENTS')
            self.event_offset_time:     int = self.get_col_index('EVENT_OFFSET_TIME')
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return self
    

def get_study_data_columns(
        key: Any
) -> Any:
    col = StudyDataColumnStructure(
            key=key
    )
    
    return col()
# endregion STRUCTURE: NUMPY COLUMNS


# # -------------------------------------------------
# region FUNCTION: PROCESS NUMPY
def is_beat_event(
        event_type: str
) -> bool:
    return any(x in ['VE', 'SVE'] for x in event_type.split('_'))

    
def check_hes_event(
        hes_event_status: [NDArray, int],
        hes_id:           int | str
) -> bool | NDArray:
    if isinstance(hes_id, str):
        hes_id = HOLTER_ALL_EVENT_SUMMARIES[hes_id]
        
    return hes_event_status & hes_id == hes_id


def get_index_event(
        hes_event_status:   NDArray,
        hes_id:             int | str
) -> NDArray:
    if isinstance(hes_id, str):
        hes_id = HOLTER_ALL_EVENT_SUMMARIES[hes_id]
        
    return np.flatnonzero(check_hes_event(hes_event_status, hes_id))


def filter_none_values(
        x: List | NDArray
) -> List:
    return list(filter(partial(is_not, None), x))


def filter_null_list_in_group(
        group: List | NDArray
) -> List:
    return list(filter(lambda x: len(x) > 0, group))


def get_group_from_index_event(
        index:      NDArray,
        value:      int = 1,
        is_region:  bool = False
) -> NDArray | List:

    group_index = np.split(index, np.flatnonzero(np.diff(index) != value) + 1)
    if is_region:
        group_index = np.array(list(map(
                lambda x: x[[0, -1]],
                group_index
        )))

    group_index = filter_null_list_in_group(group_index)

    return group_index


def get_group_index_event(
        hes_event_status:   NDArray,
        hes_id:             int | str
) -> NDArray | List:

    index = get_index_event(hes_event_status, hes_id)
    group_index = get_group_from_index_event(index)

    return group_index


def check_exists_event(
        hes_event_status:   [NDArray, int],
        hes_id:             int
) -> bool:
    if isinstance(hes_event_status, int):
        return check_hes_event(hes_event_status, hes_id)
    
    return any(check_hes_event(hes_event_status, hes_id))


def get_ind_hr_valid(
        x: NDArray
) -> NDArray:
    valid_index = np.flatnonzero(np.logical_and(
            x >= HR_MIN_THR,
            x <= HR_MAX_THR
    ))
    
    return valid_index


def check_artifact_function(
        x: NDArray | int
) -> NDArray:
    return x & HOLTER_ARTIFACT == HOLTER_ARTIFACT


def get_index_artifact_in_region(
        x: NDArray | int
) -> NDArray:
    return np.flatnonzero(check_artifact_function(x))


def get_indices_within_range(
        arr:    NDArray,
        start:  int | float,
        stop:   int | float
) -> NDArray:
    return np.flatnonzero(np.logical_and(arr >= start, arr <= stop))


def get_channel_from_channel_column(
        channel_value: [int, NDArray]
):
    return deepcopy(channel_value % HES_CONTINUOUS)


def remove_channel_from_channel_column(
        channel: NDArray
):
    return deepcopy((channel & ~(0 | 1 | 2)))


def convert_hes_beat_to_symbol(
        hes_beat_status: NDArray | List,
        is_holter_label: bool = False
) -> NDArray:

    rs = np.array(list(map(
        lambda hes: HES_TO_SYMBOL[hes],
        hes_beat_status
    )))
    
    if not is_holter_label:
        other_index = np.flatnonzero(rs == HolterSymbols.OTHER.value)
        rs[other_index] = HolterSymbols.OTHER.value

    return rs


def convert_symbol_to_hes_beat(
        symbol: NDArray | List
) -> NDArray:

    rs = np.array(list(map(
        lambda sym: SYMBOL_TO_HES.get(sym, HolterBeatTypes.OTHER.value),
        symbol
    )))

    return rs


def clear_event_type(
        hes_event_status:   NDArray,
        hes_id:             int
) -> NDArray:
    hes_event_status &= ~hes_id

    return hes_event_status


# @timeit
def get_continuous_file_index(
        study_file_index:               NDArray,
        time_files:                     NDArray
) -> NDArray | List:
    
    file_index = np.unique(study_file_index)
    group_file = np.split(file_index, np.flatnonzero(np.diff(file_index) != 0) + 1)
    try:
        check_continue_function: Any = lambda x: (time_files[x][0] == time_files[x - 1][1])
        check_status = list(map(check_continue_function, np.arange(1, len(time_files))))
        if len(check_status) > 0:
            group_file = np.split(file_index, np.flatnonzero(~np.asarray(check_status)) + 1)
    
    except (Exception, ) as error:
        st.write_error_log(error)
    
    return group_file


def get_noise_channels(
        study_df:           pl.DataFrame,
        record_config:      sr.RecordConfigurations,
        all_files:          List[Dict],
        epoch_start:        float | int,
        epoch_stop:         float | int,
        strip_channel:      int,    # 0 / 1 / 2
        percentile:         float = 0.2
) -> List:
    
    noise_channels = list()
    try:
        ind = get_index_within_range(
                nums=study_df['EPOCH'].to_numpy(),
                low=epoch_start,
                high=epoch_stop,
        )
        
        if len(ind) == 0:
            return noise_channels
        
        file_index: int = study_df['FILE_INDEX'][int(ind[0])]
        
        beat_file = join(
                record_config.record_path,
                'airp/beat',
                get_filename(all_files[file_index]['path']) + '.beat'
        )
        if 'beatInfo' not in all_files[file_index].keys():
            return noise_channels
        
        from btcy_holter import ut
        beat_data = ut.read_beat_file(
                beat_file,
                data_info=all_files[file_index]['beatInfo']
        )
        if beat_data is None:
            return noise_channels
        
        start_hourly = convert_timestamp_to_epoch_time(all_files[file_index]['start'], record_config.timezone, ms=True)
        start_sample = ((epoch_start - start_hourly) / MILLISECOND) * record_config.sampling_rate
        stop_sample = ((epoch_stop - start_hourly) / MILLISECOND) * record_config.sampling_rate
        
        for key, data in beat_data.items():
            index = np.flatnonzero(
                    np.logical_and(
                            data.beat >= start_sample,
                            data.beat <= stop_sample
                    )
            )
            channel = int(key.split('_')[-1])
            if len(index) == 0:
                noise_channels.append(channel)
                
            elif np.count_nonzero(data.beat_types[index] == HolterSymbols.OTHER.value) / len(index) > percentile:
                noise_channels.append(channel)
        
        noise_channels = list(set(noise_channels))
        strip_channel in noise_channels and noise_channels.remove(strip_channel)
        
        noise_channels = list(map(lambda x: x + 1, noise_channels))
    
    except (Exception,) as error:
        st.write_error_log(error)
    
    return noise_channels  # 1 / 2 / 3


def get_index_within_multiple_ranges(
        nums:               NDArray | pl.DataFrame,
        low:                NDArray | pl.DataFrame,
        high:               NDArray | pl.DataFrame,
        is_filter_index:    bool = True
) -> NDArray:
    
    index = np.array([])
    try:
        if isinstance(nums, pl.DataFrame):
            nums = nums.to_numpy()
        
        if isinstance(low, pl.DataFrame):
            low = low.to_numpy()
        
        if isinstance(high, pl.DataFrame):
            high = high.to_numpy()
            
        if len(low) != len(high):
            return index
        
        start_index = np.searchsorted(nums, low, side='left')
        end_index = np.searchsorted(nums, high, side='right') - 1
        
        index = np.array([start_index, end_index]).T
        if is_filter_index:
            ind = np.flatnonzero(start_index < end_index)
            if len(ind) > 0:
                index = index[ind]
        
    except (Exception,) as error:
        st.write_error_log(error)
    
    return index


def get_group_index_within_multiple_ranges(
        nums:   NDArray | pl.DataFrame,
        low:    NDArray | pl.DataFrame,
        high:   NDArray | pl.DataFrame,
        is_filter_index: bool = True
) -> List:
    
    group_index = list()
    try:
        if isinstance(nums, pl.DataFrame):
            nums = nums.to_numpy()
        
        if isinstance(low, pl.DataFrame):
            low = low.to_numpy()
        
        if isinstance(high, pl.DataFrame):
            high = high.to_numpy()
            
        if len(low) != len(high):
            return group_index
        
        index = get_index_within_multiple_ranges(nums, low, high, is_filter_index)
        group_index = list(map(
            lambda x: np.arange(x[0], x[1] + 1),
            index
        ))
        
    except (Exception,) as error:
        st.write_error_log(error)
    
    return group_index


def get_flattened_index_within_multiple_ranges(
        nums:   NDArray | pl.DataFrame,
        low:    NDArray | pl.DataFrame,
        high:   NDArray | pl.DataFrame
) -> NDArray:
    
    index = np.array([])
    try:
        if isinstance(nums, pl.DataFrame):
            nums = nums.to_numpy()
        
        if isinstance(low, pl.DataFrame):
            low = low.to_numpy()
        
        if isinstance(high, pl.DataFrame):
            high = high.to_numpy()
        
        if len(low) != len(high):
            return index
        
        group_index = get_group_index_within_multiple_ranges(nums, low, high)
        index = np.array(list(sorted(set(chain.from_iterable(group_index)))))
        
    except (Exception,) as error:
        st.write_error_log(error)
    
    return index


def get_index_within_range(
        nums:   NDArray | pl.Series,
        low:    int | float,
        high:   int | float
) -> NDArray:
    
    index = np.array([])
    try:
        if low > high:
            return index
        
        if isinstance(nums, pl.Series):
            nums = nums.to_numpy()
        
        begin = np.searchsorted(nums, low, side='left')
        end = np.searchsorted(nums, high, side='right') - 1
        index = np.arange(begin, end + 1)

    except (Exception,) as error:
        st.write_error_log(error)
    
    return index

# endregion FUNCTION: PROCESS NUMPY


# # -------------------------------------------------
# region FUNCTION: PROCESS POLARS

def pl_load_dataframe_from_parquet_file(
        study_data_path: str
) -> pl.DataFrame:
    return pl.read_parquet(study_data_path)


def pl_get_index_events(
        dataframe:      pl.DataFrame,
        event_type:     str | List,
        epochs:         NDArray | pl.Series
) -> NDArray:
    
    index = np.array([], dtype=int)
    try:
        if dataframe.height == 0:
            return index
        
        if isinstance(epochs, pl.Series):
            epochs = epochs.to_numpy()
        
        if isinstance(event_type, str):
            event_type = [event_type]
            
        df_events = (
            dataframe
            .filter(
                    pl.col('type').is_in(event_type)
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
        if df_events.height > 0:
            index = get_flattened_index_within_multiple_ranges(
                    nums=epochs,
                    low=df_events['start'].to_numpy(),
                    high=df_events['stop'].to_numpy()
            )

    except (Exception,) as error:
        st.get_error_exception(error)
    
    return index


def pl_get_group_index_events(
        dataframe:      pl.DataFrame,
        event_type:     str | List,
        epochs:         NDArray
) -> List:
    
    group = list()
    try:
        if dataframe.height == 0:
            return group

        if isinstance(event_type, str):
            event_type = [event_type]
            
        df_events = (
            dataframe
            .filter(
                    pl.col('type').is_in(event_type)
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
        if df_events.height == 0:
            return group
        
        group = get_group_index_within_multiple_ranges(
                nums=epochs,
                low=df_events['start'].to_numpy(),
                high=df_events['stop'].to_numpy()
        )
        
    except (Exception,) as error:
        st.write_error_log(error)
    
    return group


def pl_get_valid_events(
        dataframe: pl.DataFrame
) -> pl.DataFrame:
    
    try:
        if dataframe.height == 0:
            return dataframe
        
        dataframe = (
            dataframe
            .filter(
                    pl.col('statusCode') == HolterEventStatusCode.VALID.value
            )
        )
        
    except (Exception, ) as error:
        st.write_error_log(error)
    
    return dataframe


def pl_get_index_in_event_region(
        epoch_time:     NDArray | pl.Series,
        dataframe:      pl.DataFrame
) -> NDArray:

    index = np.array([], dtype=int)
    try:
        if dataframe.height == 0:
            return index
        
        if not all(x in dataframe.columns for x in ['start', 'stop']):
            return index
    
        if isinstance(epoch_time, pl.Series):
            epoch_time = epoch_time.to_numpy()
            
        dataframe = (
            dataframe
            .filter(
                    pl.col('start').is_not_null()
                    & pl.col('stop').is_not_null()
            )
        )
        if dataframe.height == 0:
            return index
        
        range_index = get_index_within_range(
            nums=epoch_time,
            low=dataframe['start'].min(),
            high=dataframe['stop'].max()
        )
        
        if len(range_index) > 0:
            # Find the insertion points for low and high
            index = get_flattened_index_within_multiple_ranges(
                nums=epoch_time,
                low=dataframe['start'].to_numpy(),
                high=dataframe['stop'].to_numpy()
            )
            pass

    except (Exception,) as error:
        st.write_error_log(error)

    return index


def pl_get_index_based_on_event_type(
        epoch_time:     NDArray,
        dataframe:      pl.DataFrame,
        event_type:     str | float
) -> NDArray:

    index = np.array([], dtype=int)
    try:
        if dataframe.height == 0:
            return index
        
        event_df = (
            pl_get_valid_events(dataframe)
            .select(
                    [
                        'start',
                        'stop',
                        'type'
                    ]
            )
            .filter(
                    pl.col('type') == event_type
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
        if event_df.height == 0:
            return index

        index = pl_get_index_in_event_region(
            epoch_time=epoch_time,
            dataframe=event_df
        )
        pass
        
    except (Exception,) as error:
        st.write_error_log(error)

    return index
    
    
def pl_get_event_based_on_event_dataframe(
        epoch_time: NDArray,
        dataframe:  pl.DataFrame
) -> NDArray:
    
    events = np.zeros_like(epoch_time)
    try:
        dataframe = pl_get_valid_events(dataframe)
        if dataframe.height == 0:
            return events
        
        for event_type in dataframe['type'].unique():
            if event_type not in HOLTER_ALL_EVENT_SUMMARIES.keys():
                continue
            
            time_ranges = dataframe.filter(pl.col('type') == event_type)
            if len(time_ranges) == 0:
                continue
                
            if event_type == 'PAUSE':
                user_events_df = (
                    time_ranges
                    .filter(
                            pl.col('source') == HolterEventSource.USER.value
                    )
                )
                
                if user_events_df.height > 0:
                    index = pl_get_index_based_on_event_type(
                            epoch_time=epoch_time,
                            dataframe=user_events_df,
                            event_type=event_type
                    )
                    if len(index) > 0:
                        events[index] |= HOLTER_ALL_EVENT_SUMMARIES[event_type]
                
                ai_events_df = (
                    time_ranges
                    .filter(
                            ~(pl.col('source') == HolterEventSource.USER.value)
                    )
                )
                index = pl_get_index_based_on_event_type(
                        epoch_time=epoch_time,
                        dataframe=ai_events_df,
                        event_type=event_type
                )
                if len(index) > 0:
                    group = get_group_from_index_event(index)
                    for index in group:
                        if len(index) == 2:
                            index = index[:-1]
                        events[index] |= HOLTER_ALL_EVENT_SUMMARIES[event_type]
            
            else:
                index = pl_get_index_based_on_event_type(
                        epoch_time=epoch_time,
                        dataframe=time_ranges,
                        event_type=event_type
                )
                if len(index) > 0:
                    events[index] |= HOLTER_ALL_EVENT_SUMMARIES[event_type]
    
    except (Exception,) as error:
        st.write_error_log(error)
    
    return events

# endregion FUNCTION: PROCESS DATAFRAME


# # -------------------------------------------------
# region FUNCTION: PERFORMANCE ANALYSIS
def timeit(
        method
):
    @functools.wraps(method)
    def timed(*args, **kwargs):
        method_name = method.__name__
        if '.' in method.__qualname__:
            method_name = method.__qualname__

        try:
            file_info = f' [{basename(kwargs["s3_file"])}] '
        except (Exception,):
            file_info = ' '

        start_time = time.monotonic()
        result = method(*args, **kwargs)
        elapsed_time = time.monotonic() - start_time
        if (
                not cf.LOGGING_ONLY_PROCESS_FUNC
                or (cf.LOGGING_ONLY_PROCESS_FUNC and method_name in ['process'])
        ):
            st.LOGGING_SESSION.timeit(f'@timeit [{method_name}]{file_info}executed in {round(elapsed_time, 4)}s')
        return result

    return timed


def get_time_process(
        start_time_process: float
) -> float:
    
    return round(time.monotonic() - start_time_process, 4)


def get_update_time_process(
        time_process:       Dict,
        start_time_process: float,
        key: str
) -> Dict:
    if key in time_process.keys():
        time_process[key] += get_time_process(start_time_process)
    else:
        time_process[key] = get_time_process(start_time_process)
        
    return time_process


def format_time_with_timezone(
        datetime_aware: str | datetime,
        timezone:       str,
) -> Any:
    utc_int = int(timezone.replace('UTC', ''))
    datetime_aware_tz_offset = datetime_aware.astimezone(
        tz=date_util_tz.tzoffset(None, utc_int * SECOND_IN_HOUR)
    )

    return datetime_aware_tz_offset


def convert_timestamp_to_epoch_time(
        timestamp:      Any,
        timezone:       str,
        dtype:          Any = int,
        ms:             bool = False
) -> int:
    if dtype == int:
        dtype: Any = lambda x: int(round(x))
    
    input_time = parser.parse(timestamp.replace('Z', '+00:00'))
    tz_time = format_time_with_timezone(input_time, timezone)
    if not isinstance(tz_time, datetime):
        tz_time = parser.parse(tz_time)
        
    epoch_time = tz_time.timestamp()
    if ms:
        epoch_time *= MILLISECOND
    
    epoch_time = dtype(epoch_time)
    
    return epoch_time


def convert_epoch_time_to_timestamp(
        epoch_time:         int | float,
        timezone:           str,
        is_iso_format:      bool = True,
        number_of_digits:   int = 13  # seconds have 10 digits, whereas milliseconds have 13 digits
) -> str | datetime:
    
    if len(str(int(epoch_time))) == number_of_digits:
        epoch_time = epoch_time / MILLISECOND
    
    fmt_time = datetime.utcfromtimestamp(float(epoch_time))
    fmt_time = fmt_time.replace(tzinfo=date_util_tz.tzutc())
    tz_time = format_time_with_timezone(fmt_time, timezone)
    
    if is_iso_format:
        tz_time = tz_time.isoformat()

    return tz_time

# endregion FUNCTION: PERFORMANCE ANALYSIS


# # -------------------------------------------------
# region FUNCTION: UTILITIES
class NumpyEncoder(
        json.JSONEncoder
):
    def default(
            self,
            obj
    ):
        
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex64, np.complex128)):
            return {'real': obj.real, 'image': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, np.void):
            return None

        return json.JSONEncoder.default(self, obj)


def check_all_variables_values(
        structure: Any
) -> bool:
    return all(value is not None for value in structure.__dict__.values())
    
    
def convert_to_array(
        x: NDArray | List | pl.Series
) -> NDArray:

    if isinstance(x, list):
        x = np.array(x)
    elif isinstance(x, pl.Series):
        x = x.to_numpy()

    return x


def append(
        x: NDArray | List,
        y: NDArray | List
) -> NDArray | List:
    
    if len(x) == 0:
        x = y
    elif isinstance(x, list):
        x = x.append(y)
    else:
        x = np.concatenate((x, y))
        
    return x


def find_most_frequency_occurring_values(
        arr:            NDArray | List,
        ignore_values:  NDArray | List = None
) -> int | None:
    value = None
    try:
        if len(arr) == 0:
            return value

        arr = convert_to_array(arr)
        if ignore_values is not None:
            ignore_values = convert_to_array(ignore_values)
            if len(ignore_values) > 0:
                arr = arr[~np.in1d(arr, ignore_values)]

        if len(arr) == 0:
            return value

        if not np.any(arr < 0):
            counts = np.bincount(arr)
            value = np.argmax(counts)
        else:
            unique_values, counts = np.unique(arr, return_counts=True)
            max_index = np.argmax(counts)
            value = unique_values[max_index]

    except (Exception,) as error:
        st.get_error_exception(error)

    return value


def is_channel_valid(
    channel: Any
) -> bool:
    return (isinstance(channel, int)
            and MIN_NUMBER_OF_CHANNELS <= channel <= MAX_NUMBER_OF_CHANNELS)


def get_filename(
        file_path: str
) -> str:
    return Path(file_path).stem


def get_path(
        file_path: str
) -> str:
    return os.path.splitext(file_path)[0]


def load_json_files(
        data_path: str
) -> Any:
    data = dict()
    try:
        
        if check_file_exists(data_path):
            with open(data_path, 'r') as json_file:
                data = json.load(json_file)
                
    except (Exception,) as error:
        st.write_error_log(error)
    
    return data


def dumps_dict(
        data: Dict
) -> Any:
    return json.dumps(data, indent=4, cls=NumpyEncoder, ensure_ascii=False)


def write_json_files(
        data_path:  str,
        data:       Dict | List
) -> None:
    try:
        data_path = join(dirname(data_path), basename(data_path))
        os.makedirs(dirname(data_path), exist_ok=True)
        with open(data_path, 'w') as json_file:
            json_file.write(dumps_dict(data))
        json_file.close()
        
    except (Exception,) as error:
        st.write_error_log(error)
    
    
# @timeit
def load_npy_file(
        npy_path: str
) -> NDArray:
    return np.load(npy_path, mmap_mode='r')


def load_all_files(
        all_files: str | List | None
) -> List[Dict]:
    
    if isinstance(all_files, str) and check_file_exists(all_files):
        all_files = load_json_files(all_files)['files']

    if isinstance(all_files, list):
        all_files = sorted(all_files, key=lambda x: x['start'])

    return all_files


def generate_study_data(
        dataframe: pl.DataFrame
) -> Any:
    
    study_data = dataframe.to_numpy()
    columns = get_study_data_columns(list(dataframe.columns))
    
    return study_data, columns


def calculate_duration(
        beats:          NDArray,
        index:          NDArray = None,
        sampling_rate:  int = None
) -> float:

    duration = 0
    try:
        if sampling_rate is None:
            st.get_error_exception('Sampling rate is not defined.')
        
        if index is None:
            index = np.arange(len(beats))
            
        interval = np.diff(beats[index])
        interval = interval[interval > 0]
        duration = np.sum(interval) / sampling_rate

    except (Exception,) as error:
        st.write_error_log(error)

    return duration


def invert_dictionary(
        inputs: Dict
) -> Dict:
    return {i: k for k, i in inputs.items()}


def generate_id_with_current_epoch_time() -> int: return int(round(time.time() * MILLISECOND))


def round_burden(
        burden: float
) -> float | int:
    burden = round(burden * 100, 2)
    if burden == 100:
        burden = int(burden)

    return burden


def check_file_exists(
        file_path: str
) -> bool:
    return exists(file_path) and isfile(file_path)


def get_extension_file(
        file_path: str
) -> str:
    file_extension = Path(file_path).suffix
    return file_extension


def check_hea_file(
        file_path: str
) -> bool:
    return get_extension_file(file_path) == '.hea'


def check_dat_file(
        file_path: str
) -> bool:
    return get_extension_file(file_path) == '.dat'


def initialize_two_beats_at_ecg_data(
        len_ecg:        int,
        sampling_rate:  int
) -> NDArray:

    beats = np.array([])
    try:
        if len_ecg < THRESHOLD_MIN_ECG_SIGNAL_LENGTH * sampling_rate:
            buffer_samples = len_ecg // THRESHOLD_MIN_ECG_SIGNAL_LENGTH
            beats = np.array([int(buffer_samples * 1), int(buffer_samples * (THRESHOLD_MIN_ECG_SIGNAL_LENGTH - 1))])
        else:
            beats = np.array([sampling_rate, len_ecg - sampling_rate])

    except (Exception,) as error:
        st.write_error_log(error)

    return beats


def resample_beat_samples(
        samples:                NDArray,
        sampling_rate:         int,
        target_sampling_rate:  int
) -> NDArray:
    
    resample_samples = deepcopy(samples)
    try:
        resample_samples = samples * target_sampling_rate / sampling_rate
        resample_samples = np.round(resample_samples).astype(int)
    
    except (Exception, ) as error:
        st.write_error_log(error)
    
    return resample_samples


def generate_epoch_from_samples(
        samples:        NDArray,
        start_time:     int,
        sampling_rate:  int
) -> NDArray:
    
    beat_times = (samples / sampling_rate) * MILLISECOND
    epoch = np.round(start_time + beat_times).astype(int)
    
    return epoch


def generate_event_id(
        event_id_existed:   NDArray | List = None,
) -> str:
    
    if event_id_existed is None:
        event_id_existed = np.array([])
        
    while True:
        event_id = str(ObjectId())
        if event_id not in event_id_existed:
            break
    
    return event_id


def get_record_configurations_by_meta_data(
        meta_data: Dict
) -> sr.RecordConfigurations:
    
    record_config = sr.RecordConfigurations()
    record_config = record_config.get_default_configurations()
    try:
        try:
            record_config.timezone = meta_data['timezone']
        except (Exception,):
            pass
        
        try:
            record_config.gain = meta_data['gain']
        except (Exception,):
            pass
        
        try:
            record_config.number_of_channels = meta_data['channels']
        except (Exception,):
            pass
        
        try:
            record_config.sampling_rate = meta_data['samplingFrequency']
        except (Exception,):
            pass
        
        try:
            record_config.tachy_threshold = meta_data['tachycardiaThreshold']
        except (Exception,):
            pass
        
        try:
            record_config.brady_threshold = meta_data['bradycardiaThreshold']
        except (Exception,):
            pass
        
        try:
            record_config.pause_threshold = int(meta_data['pauseThreshold'])
        except (Exception,):
            pass
        
        if 'studyStart' in meta_data.keys():
            record_config.record_start_time = meta_data['studyStart']
            
        elif 'start' in meta_data.keys():
            record_config.record_start_time = meta_data['start']
        
        if 'studyStop' in meta_data.keys():
            record_config.record_stop_time = meta_data['studyStop']
            
        elif 'stop' in meta_data.keys():
            record_config.record_stop_time  = meta_data['stop']
    
    except (Exception,) as error:
        st.write_error_log(error)
    
    return record_config


def get_record_configurations_by_header_file(
        record_path: str
) -> sr.RecordConfigurations:
    record_config = sr.RecordConfigurations()
    record_config = record_config.get_default_configurations()
    try:
        record = wf.rdheader(get_path(record_path))
        record_config.record_path           = record_path
        record_config.sampling_rate         = record.fs
        record_config.number_of_channels    = record.n_sig
        record_config.gain                  = record.adc_gain[0]
        pass
        
    except (Exception,) as error:
        st.write_error_log(error)
    
    return record_config

# endregion FUNCTION: UTILITIES


# # -------------------------------------------------
# region FUNCTION: CONVERT CLASSES ID to HES
RHYTHMS_TO_HES: Final[Dict] = {
    cf.RHYTHMS_DATASTORE['classes']['AFIB']:           HOLTER_AFIB,
    cf.RHYTHMS_DATASTORE['classes']['OTHER']:          HOLTER_ARTIFACT,
    cf.RHYTHMS_DATASTORE['classes']['VT']:             HOLTER_VT,
    cf.RHYTHMS_DATASTORE['classes']['SVT']:            HOLTER_SVT,
    cf.RHYTHMS_DATASTORE['classes']['AVB1']:           HOLTER_AV_BLOCK_1,
    cf.RHYTHMS_DATASTORE['classes']['AVB2']:           HOLTER_AV_BLOCK_2,
    cf.RHYTHMS_DATASTORE['classes']['AVB3']:           HOLTER_AV_BLOCK_3,
}

HES_TO_SYMBOL:              Final[Dict] = {
    HolterBeatTypes.N.value:                            HolterSymbols.N.value,
    HolterBeatTypes.VE.value:                           HolterSymbols.VE.value,
    HolterBeatTypes.SVE.value:                          HolterSymbols.SVE.value,
    HolterBeatTypes.IVCD.value:                         HolterSymbols.IVCD.value,
    HolterBeatTypes.OTHER.value:                        HolterSymbols.OTHER.value,
    HolterBeatTypes.MARKED.value:                       HolterSymbols.MARKED.value,
}

SYMBOL_TO_HES:              Final[Dict] = {i: k for k, i in HES_TO_SYMBOL.items()}


VALID_BEAT_TYPE:            Final[List] = [
    HolterSymbols.N.value,
    HolterSymbols.VE.value,
    HolterSymbols.SVE.value,
    HolterSymbols.IVCD.value
]

INVALID_BEAT_TYPE:          Final[List] = [HolterSymbols.OTHER.value, HolterSymbols.MARKED.value]

VALID_HES_BEAT_TYPE:        Final[List] = list(map(lambda x: SYMBOL_TO_HES[x], VALID_BEAT_TYPE))
INVALID_HES_BEAT_TYPE:      Final[List] = list(map(lambda x: SYMBOL_TO_HES[x], INVALID_BEAT_TYPE))

TB_VALID_BEAT_DETECT:       Final[list] = [HolterSymbols.N.value, HolterSymbols.IVCD.value]
TB_VALID_HES_BEAT_TYPE:     Final[list] = list(map(lambda x: SYMBOL_TO_HES[x], TB_VALID_BEAT_DETECT))

# endregion CONVERT CLASSES ID to HES


# # -------------------------------------------------
# region PHYSIONET
_PHYSIONET_ANN: Final[Dict] = {
    'N': ["N", "e", "j", "n", "F", "f", "/"],
    'S': ["A", "a", "S", "J"],
    'V': ["V", "E", "!", "r"],
    '|': ['f', 'Q', '|'],
    'R': ["L", "R"]
}


def _convert_ann_type(
    sym: str
) -> str | None:
    for lb, types in _PHYSIONET_ANN.items():
        if sym in types:
            return lb


def convert_physionet_symbol_format(
        symbols: NDArray | List
) -> NDArray | List:
    try:
        symbols = np.array(list(map(_convert_ann_type, symbols)))

    except (Exception,) as error:
        st.write_error_log(error)

    return symbols


def convert_physionet_symbol_format_to_hes(
        symbols: NDArray | List
) -> NDArray | List:
    try:
        symbols = np.array(list(map(lambda x: SYMBOL_TO_HES[_convert_ann_type(x)], symbols)))

    except (Exception,) as error:
        st.write_error_log(error)

    return symbols
# endregion PHYSIONET


# # -------------------------------------------------
# region DICTIONARY: UTILITIES

HOLTER_BEAT_EVENT:                  Final[Dict] = {
    'SINGLE_VE':        HOLTER_SINGLE_VES,
    'VE_RUN':           HOLTER_VES_RUN,
    'VE_COUPLET':       HOLTER_VES_COUPLET,
    'VE_BIGEMINY':      HOLTER_VES_BIGEMINY,
    'VE_TRIGEMINAL':    HOLTER_VES_TRIGEMINAL,
    'VE_QUADRIGEMINY':  HOLTER_VES_QUADRIGEMINY,

    'SINGLE_SVE':       HOLTER_SINGLE_SVES,
    'SVE_RUN':          HOLTER_SVES_RUN,
    'SVE_COUPLET':      HOLTER_SVES_COUPLET,
    'SVE_BIGEMINY':     HOLTER_SVES_BIGEMINY,
    'SVE_TRIGEMINAL':   HOLTER_SVES_TRIGEMINAL,
    'SVE_QUADRIGEMINY': HOLTER_SVES_QUADRIGEMINY
}


HOLTER_ALL_EVENT_TITLES:            Final[Dict] = {
    # BEAT EVENTS
    'SINGLE_VE':                'VE',
    'VE_RUN':                   'VE Run',
    'VE_COUPLET':               'VE Couplet',
    'VE_BIGEMINY':              'VE Bigeminy',
    'VE_TRIGEMINAL':            'VE Trigeminy',
    'VE_QUADRIGEMINY':          'VE Quadrigeminy',
    
    'SINGLE_SVE':               'SVE',
    'SVE_RUN':                  'SVE Run',
    'SVE_COUPLET':              'SVE Couplet',
    'SVE_BIGEMINY':             'SVE Bigeminy',
    'SVE_TRIGEMINAL':           'SVE Trigeminy',
    'SVE_QUADRIGEMINY':         'SVE Quadrigeminy',
    
    # RHYTHM EVENTS
    'SINUS':                    'Sinus Rhythm',
    'SINUS_TACHY':              'Sinus Tachycardia',
    'SINUS_BRADY':              'Sinus Bradycardia',
    'ARTIFACT':                 'Artifact',
    
    'TACHY':                    'Tachycardia',
    'BRADY':                    'Bradycardia',
    'PAUSE':                    'Pause',
    'MAX_HR':                   'Max HR',
    'MIN_HR':                   'Min HR',
    'LONG_RR':                  'Long RR',
    'SINUS_ARRHYTHMIA':         'Sinus Arrhythmia',
    
    'AVB1':                     '1st degree AV Block',
    'AVB2':                     '2nd degree AV Block',
    'AVB3':                     '3rd degree AV Block',
    'VT':                       'Ventricular Tachycardia',
    'SVT':                      'Supraventricular Tachycardia',
    'AFIB':                     'Atrial Fibrillation/Flutter',
    'VFF':                      'Ventricular Fibrillation/Flutter'
}


HOLTER_ALL_EVENT_SUMMARIES:         Final[Dict] = {
    # BEAT EVENTS
    'SINGLE_VE':                HOLTER_SINGLE_VES,
    'VE_RUN':                   HOLTER_VES_RUN,
    'VE_COUPLET':               HOLTER_VES_COUPLET,
    'VE_BIGEMINY':              HOLTER_VES_BIGEMINY,
    'VE_TRIGEMINAL':            HOLTER_VES_TRIGEMINAL,
    'VE_QUADRIGEMINY':          HOLTER_VES_QUADRIGEMINY,
    
    'SINGLE_SVE':               HOLTER_SINGLE_SVES,
    'SVE_RUN':                  HOLTER_SVES_RUN,
    'SVE_COUPLET':              HOLTER_SVES_COUPLET,
    'SVE_BIGEMINY':             HOLTER_SVES_BIGEMINY,
    'SVE_TRIGEMINAL':           HOLTER_SVES_TRIGEMINAL,
    'SVE_QUADRIGEMINY':         HOLTER_SVES_QUADRIGEMINY,
    
    # RHYTHM EVENTS
    'SINUS':                    HOLTER_SINUS_RHYTHM,
    'BRADY':                    HOLTER_BRADY,
    'TACHY':                    HOLTER_TACHY,
    'PAUSE':                    HOLTER_PAUSE,
    'MAX_HR':                   HOLTER_MAX_HR,
    'MIN_HR':                   HOLTER_MIN_HR,
    'ARTIFACT':                 HOLTER_ARTIFACT,
    'AVB1':                     HOLTER_AV_BLOCK_1,
    'AVB2':                     HOLTER_AV_BLOCK_2,
    'AVB3':                     HOLTER_AV_BLOCK_3,
    'VT':                       HOLTER_VT,
    'SVT':                      HOLTER_SVT,
    'AFIB':                     HOLTER_AFIB,
}


HOLTER_ALL_EVENT_SUMMARIES_INVERT = invert_dictionary(HOLTER_ALL_EVENT_SUMMARIES)

# endregion DICTIONARY: UTILITIES


# # -------------------------------------------------
# region DIRECTORY: S3 PATH
class S3DataStructure:
    beat_file_prefix:       Final[str] = 'airp/beat'
    hourly_file_prefix:     Final[str] = 'airp/npy'
    
    beat_final_prefix:      Final[str] = 'final-beat'
    
    study_data_prefix:      Final[str] = 'npy'
    hourly_data_prefix:     Final[str] = 'data'
    
    beat_df_prefix:         Final[str] = '.parquet'
    event_df_prefix:        Final[str] = '-events.parquet'
    pvc_df_prefix:          Final[str] = '-pvc_morphology.parquet'
    
    def __call__(
            self,
            message: Dict
    ) -> Any:
        try:
            if not all(x in message.keys() for x in ['studyId', 'profileId']):
                return
            
            self.beat_file_dir:     str = join(message['studyId'], self.beat_file_prefix)
            self.hourly_file_dir:   str = join(message['studyId'], self.hourly_file_prefix)
            
            self.beat_final_dir:    str = join(message['studyId'], self.beat_final_prefix, str(message['profileId']))
            
            self.study_data_dir:    str = join(message['studyId'], self.study_data_prefix)
            self.hourly_data_dir:   str = join(message['studyId'], self.hourly_data_prefix)
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            
        return self
    
# endregion DIRECTORY: S3 PATH
