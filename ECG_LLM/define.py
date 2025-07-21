
path_model = "/media/server2/MegaDataset/DataTraining/Models/"
model_path = path_model + "Model_beat_classify_study_data_n_embd_64_n_head_8_n_layer_8_block_size_1280_token_512.pth"

data_path = "/media/server2/MegaDataset/DataTraining/NumpyData_tokenization_512"
PHYSIONET_DATA = "/media/server2/MegaDataset/PhysionetData/TEST_BXB"


HEARTBEAT_TYPE = {
    "_": 0,
    "N": 1,
    "V": 2,
    "S": 3,
    "|": 0,
    "R": 1,
}

NUM_CLASS = max([HEARTBEAT_TYPE[k] for k in HEARTBEAT_TYPE.keys()]) + 1
NAME_CLASS = [k for k in HEARTBEAT_TYPE.keys()][:NUM_CLASS]

# NUM_EVENT_CLASS = max([EVENT_TYPE[k] for k in EVENT_TYPE.keys()]) + 1
# NAME_EVENT_CLASS = [k for k in EVENT_TYPE.keys()][:NUM_EVENT_CLASS]

SAMPLING_RATE = 128
MAX_FILE_LEN_PROCESS = (60 * 60 * SAMPLING_RATE) # 128G RAM
FEATURE_LEN = SAMPLING_RATE * 10
OFFSET_LABEL = int(0.035 * SAMPLING_RATE)   # Normal QRS width is 70-100 ms
NUM_CHANNEL = 1
                                            # (a duration of 110 ms is sometimes observed in healthy subjects)
GROUP_LABEL_LEN = int(0.125 * SAMPLING_RATE)
MIN_RR_LEN = (0.2 * SAMPLING_RATE) # 400 Bpm
CHANNEL_DEFAULT = 0
USED_NORM = True