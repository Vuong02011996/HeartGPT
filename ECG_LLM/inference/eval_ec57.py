import os
from glob import glob
import matplotlib
import numpy as np
import torch
import wfdb as wf
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
from wfdb.processing import resample_singlechan
from EC57.ec57_script import ec57_eval, del_result
# from REBAR.data.config_dataset import (SAMPLING_RATE,
#                                        NUM_CLASS,
#                                        MIN_RR_LEN,
#                                        HEARTBEAT_TYPE,
#                                        FEATURE_LEN,
#                                        PHYSIONET_DATA,
#                                        CHANNEL_DEFAULT,
#                                        ROOT_DISK,
#                                        USED_NORM,
#                                        DISK_NAME)
from ECG_LLM.define import (SAMPLING_RATE,
                           NUM_CLASS,
                           MIN_RR_LEN,
                           HEARTBEAT_TYPE,
                           FEATURE_LEN,
                           PHYSIONET_DATA,
                           CHANNEL_DEFAULT,
                           # ROOT_DISK,
                           USED_NORM,
                           # DISK_NAME
                            )

from ECG_LLM.dataset_processing.reprocessing import butter_bandpass_filter, beat_annotations, norm
# from REBAR.downstream.downstream_config import DownStream_ExpConfig
# from REBAR.downstream.downstream_nets import ResNetLSTM
# from REBAR.downstream.utils import (get_peaks)
# from REBAR.models.REBAR.REBAR_CrossAttn.REBAR_CrossAttn import REBAR_CrossAttn_Config
# from experiments.configs.rebar_expconfigs import REBAR_ExpConfig
# from utils.utils import import_model, init_dl_program
# from rebar_beat_classification_60s import load_model_from_path
import time
from ECG_LLM.define import model_path
from ECG_LLM.Hearbeat_without_pre_training_k_fold import HeartGPTModel
from ECG_LLM.utils import get_peaks

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = HeartGPTModel()
model.load_state_dict(torch.load(model_path))
model.eval()
m = model.to(device)


def delete_all_file():
    from pathlib import Path

    folder = Path(PHYSIONET_DATA)

    # Find all .beat files recursively
    beat_files = list(folder.rglob("*.beat"))

    # Delete them
    for file in beat_files:
        file.unlink()
        print(f"Deleted: {file}")
if __name__ == "__main__":
    delete_all_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 50
    # beat_classify = torch.jit.load('/media/server2/MegaDataset/Backup/Vuong/Reference_Project/ECG-LM/Deploy/Models/encoder_beat_classify.pt',
    #                                map_location=device)

    # encode_model, downstream_model = load_model_from_path(
    #     '/media/khanh/UltraDataset/REBAR/ecg_norm_1_data_strip60_250528/out/CrossAttn_MaskExtended_128_ReceptiveField_3_EmbedDim_320/ts2vec_output_dims_320_hidden_dims_64_depth_11_alpha_0.5_candidateset_size_30/Beat5_ResNetDilatedLSTM_num_layers_1_hidden_dim_320_dilation_7_relu_active_0_dropout_0.0_combine_loss_alpha_0.7/')

    DS3 = {102, 104, 107, 217}
    # DS4 = {207}
    MAX_SAMPLE_PROCESS = 60 * 60

    DB_TESTING = [['mitdb', 'atr', 'atr'], ['ahadb', 'atr', 'atr'], ['nstdb', 'atr', 'atr'],
                  # ['edb', 'atr', 'atr']
                  ]
    for db in DB_TESTING:
        # if db[0] != "edb":
        #     continue

        # all_files = glob(PHYSIONET_DATA.replace('xyz', db[0]) + '/*.atr')
        all_files = glob(PHYSIONET_DATA + f'/{db[0]}/*.atr')
        if db[0] == "mitdb":
            all_file_eval = [f[:-4] for f in all_files if
                             len(os.path.basename(f)[:-4]) == 3 and int(os.path.basename(f)[:-4]) not in DS3]
        else:
            all_file_eval = [f[:-4] for f in all_files]

        beat_inv = {i: k for i, k in enumerate(HEARTBEAT_TYPE.keys())}
        beat_ind = {k: i for i, k in enumerate(HEARTBEAT_TYPE.keys())}
        for inx in tqdm(range(len(all_file_eval))):
            start_time = time.time()
            file_name = all_file_eval[inx]
            header = wf.rdheader(file_name)
            # if os.path.basename(file_name) != '100':
            #     continue

            study_len = header.sig_len
            samp_to = 0
            samp_from = 0
            fs_origin = header.fs
            total_peak = []
            total_symbol = []
            while samp_to < header.sig_len:
                if fs_origin >= study_len:
                    break

                samp_len = min(int(MAX_SAMPLE_PROCESS * fs_origin), study_len)
                samp_to = samp_from + samp_len
                record = wf.rdsamp(file_name, sampfrom=samp_from, sampto=samp_to, channels=[CHANNEL_DEFAULT])
                buf_record = np.nan_to_num(record[0][:, 0])
                ann = wf.rdann(file_name, extension='atr')
                if fs_origin != SAMPLING_RATE:
                    buf_ecg, ann = resample_singlechan(buf_record,
                                                       ann,
                                                       ann.fs,
                                                       SAMPLING_RATE)

                else:
                    buf_ecg = buf_record.copy()

                true_beats, true_types = beat_annotations(ann)
                _buf_ecg = butter_bandpass_filter(buf_ecg, 0.5, 40.0, SAMPLING_RATE)
                if USED_NORM:
                    buf_ecg = norm(_buf_ecg, int(0.5 * SAMPLING_RATE))
                else:
                    buf_ecg = _buf_ecg.copy()

                buf_ecg_size = len(buf_ecg)
                index_sample = np.arange(FEATURE_LEN)[None, :] + \
                               np.arange(0, buf_ecg_size - FEATURE_LEN, FEATURE_LEN)[:, None]

                frame_ecg = buf_ecg[index_sample]

                scale_number = 1000
                # Min-max scaling across each row (frame)
                data_min = np.min(frame_ecg, axis=1, keepdims=True)
                data_max = np.max(frame_ecg, axis=1, keepdims=True)
                scaled = (frame_ecg - data_min) / (data_max - data_min + 1e-8)  # avoid division by zero

                # Scale to [0, 1000] and round
                result = np.round(scale_number * scaled, 0)
                frame_ecg = result

                batch_index = 0
                samp_len = len(index_sample)
                pred_beats = []
                pred_types = []
                while batch_index < len(index_sample):
                    _data = frame_ecg[batch_index: min(batch_index + batch_size, samp_len)].copy()
                    if len(_data) == 0:
                        break

                    with torch.no_grad():
                        # x = encode_model.encode_run(np.expand_dims(_data, axis=-1)).to(device)
                        # logits = downstream_model.classify(x)
                        # _data = torch.from_numpy(_data).to(device).float()
                        _data = torch.from_numpy(_data).to(device).long()
                        # _data = torch.tensor(_data, dtype=torch.long, device=device)
                        # logits = beat_classify(_data.unsqueeze(-1))
                        logits = m.get_logits(_data)

                        group_beat_candidate = np.argmax(logits.cpu().detach().numpy(), axis=-1)
                    # _ecg_signal = _data.flatten().cpu().numpy()
                    beats, symbols = get_peaks(ecg_signal=_data.flatten().cpu().numpy(),
                                               group_beat_candidate=group_beat_candidate.flatten(),
                                               reprocess=False)
                    beats += index_sample[batch_index][0]
                    if len(pred_beats) == 0:
                        pred_beats = beats
                        pred_types = symbols
                    else:
                        pred_beats = np.concatenate((pred_beats, beats))
                        pred_types = np.concatenate((pred_types, symbols))

                    batch_index += _data.shape[0]

                # transfer data to the original frequency
                pred_beats = (pred_beats * fs_origin) // SAMPLING_RATE

                if len(total_peak) == 0 and len(pred_beats) > 0:
                    total_peak = pred_beats + samp_from
                    total_symbol = pred_types
                elif len(pred_beats) > 0:
                    tmp_peaks = pred_beats + samp_from
                    for inx in reversed(range(0, len(total_peak) - 1)):
                        if ((MIN_RR_LEN * fs_origin) // SAMPLING_RATE) < tmp_peaks[0] - total_peak[inx]:
                            inx += 1
                            break

                    total_peak = np.concatenate((total_peak[:inx], tmp_peaks[0:]), axis=0)
                    total_symbol = np.concatenate((total_symbol[:inx], pred_types[0:]), axis=0)

                study_len -= (samp_to - samp_from)
                samp_from = samp_to - fs_origin

            chk = list(np.where(np.diff(total_peak) < ((MIN_RR_LEN * fs_origin) // SAMPLING_RATE)))
            if len(chk[0]) > 0:
                total_peak = np.delete(total_peak, chk[0])
                total_symbol = np.delete(total_symbol, chk[0])
            print(f"Time cost {file_name}: ", time.time() - start_time)
            # try:
            dir_test = os.path.dirname(file_name)
            curr_dir = os.getcwd()
            os.chdir(dir_test + '/')
            annotation2 = wf.Annotation(record_name=os.path.basename(file_name),
                                        extension='beat',
                                        sample=np.asarray(total_peak),
                                        symbol=np.asarray(total_symbol),
                                        fs=fs_origin)
            annotation2.wrann(write_fs=True)
            os.chdir(curr_dir)
            # except Exception as err:
            #     print("file {} get err {}".format(file_name, err))
        downstream_model_ec57_dir = os.path.join('/media/server2/MegaDataset/PhysionetData/TEST_BXB/OUT/')
        os.makedirs(downstream_model_ec57_dir, exist_ok=True)
        output_ec57_directory_by_db = os.path.join(downstream_model_ec57_dir,
                                                   db[0],
                                                   "1.0.0"
                                                   )
        os.makedirs(output_ec57_directory_by_db, exist_ok=True)
        ec57_eval(dir_db=db[0],
                  output_ec57_directory=downstream_model_ec57_dir + "/",
                  # physionet_directory="/{}/{}/{}/ECG/physionet.org/files/".format(ROOT_DISK, os.getlogin(), DISK_NAME),
                  physionet_directory=PHYSIONET_DATA + "/",
                  beat_ext_db=db[1],
                  event_ext_db=None,
                  beat_ext_ai='beat',
                  event_ext_ai=None)
        del_result(dir_db=db[0],
                   # physionet_directory="/{}/{}/{}/ECG/physionet.org/files/".format(ROOT_DISK, os.getlogin(), DISK_NAME),
                   physionet_directory=PHYSIONET_DATA + "/",
                   output_ec57_directory=downstream_model_ec57_dir + "/")