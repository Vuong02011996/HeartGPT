import os

import structs as sr
from typing import Final, Any
from dotenv import load_dotenv


# # -------------------------------------------------
# region ...
load_dotenv()


def get_param(
        param_name:     str,
        data_type:      type,
        default_value:  Any = None
) -> Any:
    
    if param_name in os.environ:
        return data_type(os.environ[param_name])

    return default_value


def get_boolean_params(
        field:      str,
        default:    bool = False
) -> bool:
    status = default
    if field in os.environ.keys():
        status = os.environ.get(field, 'true').lower() in ('true', '1', 't')
        
    return status
# endregion ...


# # -------------------------------------------------
# region ENV: AWS CONFIGURATION
DEBUG_MODE:                Final[bool] = get_boolean_params('DEBUG_MODE')

# Config to disable logging
ENABLE_LOGGING:            Final[bool] = get_boolean_params('ENABLE_LOGGING', default=True)

# Config only log time of process function
LOGGING_ONLY_PROCESS_FUNC: Final[bool] = get_boolean_params('LOGGING_ONLY_PROCESS_FUNCTION')

USE_IRSA:                  Final[bool] = get_boolean_params('USE_IRSA', default=False)

AWS_REGION:                Final[str] = os.environ.get('AWS_REGION')
AWS_BUCKET:                Final[str] = os.environ.get('AWS_BUCKET')
AWS_PROFILE:               Final[str] = os.environ.get('AWS_PROFILE')
# endregion AWS CONFIGURATION


# # -------------------------------------------------
# region ENV: WORKING SPACE
GLOBAL_STUDY_ID:            Any = None
GLOBAL_MESSAGE_ID:          Any = None

TIMEZONE:                   Final[str] = get_param('TIMEZONE', str, 'UTC-0')
RELEASE:                    Final[str] = get_param('RELEASE', str, '...')

RECORD_VOLUME:              Final[str] = os.environ['RECORD_VOLUME']
RESULT_VOLUME:              Final[str] = os.environ['RESULT_VOLUME']
# endregion ENV: WORKING SPACE


# # -------------------------------------------------
# region ENV: SYS
NUM_CORES:                  Final[int] = get_param('NUM_CORES', int, os.cpu_count())

CHANNEL:                    Final[int] = get_param('CHANNEL', int, 1)
GAIN:                       Final[float] = get_param('GAIN', float, 655.35875)
NUMBER_OF_CHANNELS:         Final[int] = get_param('NUMBER_OF_CHANNELS', int, 3)

SAMPLING_RATE:              Final[int] = get_param('SAMPLING_RATE', int, 250)               # Hz
HES_SAMPLING_RATE:          Final[int] = get_param('HES_SAMPLING_RATE', int, 200)           # Hz
MAX_SAMPLE_PROCESS:         Final[int] = get_param('MAX_TIME_PROCESS', float, 16) * 60      # minutes

TACHY_THR:                  Final[int] = get_param('TACHY_THR', int, 120)                   # bpm
BRADY_THR:                  Final[int] = get_param('BRADY_THR', int, 43)                    # bpm

PAUSE_RR_THR:               Final[float] = get_param('PAUSE_RR_THR', int, 2450)             # milliseconds
LONG_RR_THR:                Final[float] = get_param('LONG_RR_THR', float, 1.9)             # seconds
MIN_RR_INTERVAL:            Final[float] = get_param('MIN_RR_INTERVAL', float, 0.2)         # seconds
# endregion ENV: SYS


# # -------------------------------------------------
# region TFSERVER
DEFAULT_TF_SERVER_NAME:     Final[str] = os.environ.get('TF_SERVER_NAME')
DEFAULT_TF_SERVER_PORT:     Final[str] = os.environ.get('TF_SERVER_PORT')
DEFAULT_TF_MAX_BATCH_SIZE:  Final[str] = os.environ.get('TF_MAX_BATCH_SIZE', 1024)
DEFINE_TF_WAITING_TIME:     Final[int] = get_param('TF_WAITING_TIME', int, 10)              # seconds

DEFAULT_TF_SERVER_ID:       Final[str] = f'{DEFAULT_TF_SERVER_NAME}:{DEFAULT_TF_SERVER_PORT}'

TF_CHANNEL_MODEL: Final[sr.TFServerModelStructure] = sr.TFServerModelStructure(
        model_name=get_param('MODEL_NAME', str, 'channels'),
        signature_name=get_param('SIGNATURE', str, 'channels'),
        input_name=get_param('INPUT', str, 'segment'),
        output_name=get_param('OUTPUT', str, 'prediction'),
)

TF_QRS_MODEL: Final[sr.TFServerModelStructure] = sr.TFServerModelStructure(
        model_name=get_param('MODEL_NAME', str, 'qrs'),
        signature_name=get_param('SIGNATURE', str, 'predict_signal'),
        input_name=get_param('INPUT', str, 'signal'),
        output_name=get_param('OUTPUT', str, 'classes'),
)

TF_BEAT_MODEL: Final[sr.TFServerModelStructure] = sr.TFServerModelStructure(
        model_name=get_param('MODEL_NAME', str, 'beats'),
        signature_name=get_param('SIGNATURE', str, 'beats'),
        input_name=get_param('INPUT', str, 'segment'),
        output_name=get_param('OUTPUT', str, 'prediction'),
)

TF_RHYTHM_MODEL: Final[sr.TFServerModelStructure] = sr.TFServerModelStructure(
        model_name=get_param('MODEL_NAME', str, 'rhythms'),
        signature_name=get_param('SIGNATURE', str, 'rhythms'),
        input_name=get_param('INPUT', str, 'segment'),
        output_name=get_param('OUTPUT', str, 'prediction'),
)

TF_URGENT_MODEL: Final[sr.TFServerModelStructure] = sr.TFServerModelStructure(
        model_name=get_param('MODEL_NAME', str, 'urgent'),
        signature_name=get_param('SIGNATURE', str, 'urgent'),
        input_name=get_param('INPUT', str, 'segment'),
        output_name=get_param('OUTPUT', str, 'prediction'),
)

TF_NOISE_MODEL: Final[sr.TFServerModelStructure] = sr.TFServerModelStructure(
        model_name=get_param('MODEL_NAME', str, 'noise'),
        signature_name=get_param('SIGNATURE', str, 'noise'),
        input_name=get_param('INPUT', str, 'segment'),
        output_name=get_param('OUTPUT', str, 'prediction'),
)

# endregion ENV: TFSERVER


# # -------------------------------------------------
# region ENV: AI MODEL
STANDARD_EVENT_LENGTH:      Final[int] = 60                 # seconds

URGENT_DATASTORE: Final[dict] = {
    'feature_len':              2500,                   # samples
    'sampling_rate':            250,                    # Hz
    'classes': {
            'Non-Urgent':       0,
            'Urgent':           1
    }
}

NOISE_DATASTORE: Final[dict] = {
    'feature_len':              2500,                   # samples
    'sampling_rate':            250,                    # Hz
    'classes': {
            'SINUS':            0,
            'NOISE':            1
    },
    
    'bwr':                      False,
    'norm':                     [False, True],          # [input, output]
    'highpass':                 [True, True],           # [input, output]
    
    'NUM_OVERLAP':              5,                      # second
    'NUM_NORM':                 0.5,                    # seconds
    'BANDPASS_FILTER':          [0.5, 30.0],            # Hz
    'ABNORMAL_THR_MSE':         [0, 50],
    'ABNORMAL_THR_STD':         [0, 150],
}

QRS_DATASTORE: Final[dict] = {
    'beat_offset':              [0.2, 0.3],             # second
    'sampling_rate':            250,                    # Hz
    'classes': {
            'NO_QRS':           0,
            'QRS':              1
    },
    'bwr':                      True,
    'norm':                     True,
    'NUM_NORM':                 0.6,                    # seconds
    'BANDPASS_FILTER':          [0.5, 30.0],             # Hz
    'OFFSET_FRAME_BEAT':        [-6, -3, 0, 3, 6]       # samples
}

CHANNELS_DATASTORE: Final[dict] = {
    'feature_len':              2560,                   # samples
    'sampling_rate':            256,                    # Hz
    'classes': {
        'CH1':                  0,
        'CH2':                  1,
        'CH3':                  2,
        'LEADOFF':              3
    },
    'bwr':                      False,
    'norm':                     False,
    'NUM_NORM':                 0.6,                    # seconds
    'BANDPASS_FILTER':          [1.0, 30.0],             # Hz
    'R_PEAK_AMPLITUDE':         False,
}

RHYTHMS_DATASTORE: Final[dict] = {
    'feature_len':              2560,                   # samples
    'sampling_rate':            256,                    # Hz
    'classes': {
        'OTHER':                0,
        'SINUS':                1,
        'AFIB':                 2,
        'SVT':                  3,
        'VT':                   4,
        'AVB1':                 5,
        'AVB2':                 6,
        'AVB3':                 7
    },
    'bwr':                      False,
    'norm':                     False,
    'NUM_NORM':                 0.6,                    # seconds
    'NUM_OVERLAP':              60,  # seconds
    'BANDPASS_FILTER':          [1.0, 30.0],            # Hz
    'R_PEAK_AMPLITUDE':         [-5.0, 5.0],            # mV
}

BEATS_DATASTORE: Final[dict] = {
    'feature_len':              15360,                  # samples
    'sampling_rate':            256,                    # Hz
    'classes': {
        'NOTABEAT':             0,
        '|':                    1,
        'N':                    2,
        'S':                    3,
        'V':                    4,
        'R':                    5,
        'Q':                    6
    },
    'bwr':                      False,
    'norm':                     False,
    'NUM_BLOCK':                480,                    # samples
    'NUM_NORM':                 0.6,                    # seconds
    'NUM_OVERLAP':              60,                     # seconds
    'BANDPASS_FILTER':          [1, 30],                # Hz
    'R_PEAK_AMPLITUDE':         [-5, 5],                # mV
    'OFFSET_FRAME_BEAT':        [0, 3, 6, 9, 11],       # samples
    'NUM_BEAT_BEFORE':          2,
    'NUM_BEAT_AFTER':           1
}

SYMBOLS_IN_RHYTHMS: Final[dict] = {
    'OTHER': '|',
}
# endregion ENV: AI MODEL
