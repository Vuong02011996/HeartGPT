import os
from ssl import CHANNEL_BINDING_TYPES

HEARTBEAT_TYPE = {
    "_": 0,
    "N": 1,
    "V": 2,
    "S": 3,
    "|": 0,
}

EVENT_TYPE = {
    "SINUS": 0,
    "SVE": 1,
    "VE": 2,
    "SVT": 3,
    "VT": 4,
    "AF": 5,
    "OTHER": 6,
}

ECG_TYPE = {
    "ECG": 0,
    "ARTIFACT": 1,
}

NUM_CLASS = max([HEARTBEAT_TYPE[k] for k in HEARTBEAT_TYPE.keys()]) + 1
NAME_CLASS = [k for k in HEARTBEAT_TYPE.keys()][:NUM_CLASS]
SAMPLING_RATE = 128
MAX_FILE_LEN_PROCESS = (39 * 60 * SAMPLING_RATE) # 128G RAM
FEATURE_LEN = SAMPLING_RATE * 10
OFFSET_LABEL = int(0.035 * SAMPLING_RATE)   # Normal QRS width is 70-100 ms
NUM_CHANNEL = 1
vocab_size = 1001
scale_number = vocab_size - 1
                                            # (a duration of 110 ms is sometimes observed in healthy subjects)
GROUP_LABEL_LEN = int(0.125 * SAMPLING_RATE)
MIN_RR_LEN = (0.2 * SAMPLING_RATE) # 400 Bpm
CHANNEL_DEFAULT = 0
USED_NORM = True
ROOT_DISK = "media"
# DISK_NAME = "UltraDataset"
DISK_NAME = "MegaDataset"

# PHYSIONET_DATA = "/{}/{}/{}/ECG/physionet.org/files/{}/".format(ROOT_DISK, os.getlogin(), DISK_NAME, "xyz")
PHYSIONET_DATA = "/{}/{}/{}/ECG/physionet.org/files/{}/1.0.0".format(ROOT_DISK, os.getlogin(), DISK_NAME, "xyz")
OUT_PHYSIONET_DATA = "/{}/{}/{}/REBAR/physionet/afib".format(ROOT_DISK, os.getlogin(), DISK_NAME)
OUT_PHYSIONET_DATA_PROCESS = "/{}/{}/{}/REBAR/ecg_norm_{}_data_physionet".format(ROOT_DISK, os.getlogin(), DISK_NAME, int(USED_NORM))