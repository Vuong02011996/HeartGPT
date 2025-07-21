import os
import random
import re
from glob import glob
import shutil
import wfdb as wf
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
from wfdb.processing import resample_singlechan, resample_sig
import numpy as np
import torch
from ECG_LLM.process_data_training.config_dataset import EVENT_TYPE, SAMPLING_RATE, USED_NORM, FEATURE_LEN, MIN_RR_LEN, \
    OFFSET_LABEL, HEARTBEAT_TYPE, scale_number
from ECG_LLM.process_data_training.helper import ecg_to_tokens, plot_ecg_with_annotations
from ECG_LLM.process_data_training.reprocessing import beat_annotations, butter_bandpass_filter, norm

from ECG_LLM.define import model_path
from ECG_LLM.Hearbeat_without_pre_training_k_fold import HeartGPTModel
from ECG_LLM.utils import get_peaks

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = HeartGPTModel()
model.load_state_dict(torch.load(model_path))
model.eval()
m_infer = model.to(device)


def get_peak_symbols():
    all_events = glob(study_path + '*/*/*.hea')
    error = 0
    # Open a file to log error paths
    # with open(error_log_file, 'w') as f:
    #     f.write("Error Event Paths:\n")
    #     f.write("=" * 50 + "\n\n")
    for event_path in tqdm(all_events):
        # if "670caae9907d979628088a52" not in event_path:
        #     continue
        file_name = event_path[:-4]
        header = wf.rdheader(file_name)
        fs_origin = header.fs
        channel = min(1, len(header.sig_name) - 1)
        startSample = 0
        stopSample = header.sig_len
        com = header.comments
        event_type = EVENT_TYPE["OTHER"]
        for m in com:
            if "channel:" in m:
                channel = int(m[-2:])
            elif "startSample:" in m:
                startSample = (int(re.findall(r'\d+', m)[0]) * SAMPLING_RATE) // fs_origin
            elif "stopSample:" in m:
                stopSample = (int(re.findall(r'\d+', m)[0]) * SAMPLING_RATE) // fs_origin
            elif "eventType" in m:
                # lock.acquire()
                # if str(m).replace("eventType: ", "") not in event_type_dict.keys():
                #     event_type_dict[str(m).replace("eventType: ", "")] = 1
                # else:
                #     event_type_dict[str(m).replace("eventType: ", "")] += 1
                #
                # lock.release()
                if "SVT" in m:
                    event_type = EVENT_TYPE["SVT"]
                elif "VT" in m:
                    event_type = EVENT_TYPE["VT"]
                elif "SVE_" in m or "_SVE" in m:
                    event_type = EVENT_TYPE["SVE"]
                elif "VE_" in m or "_VE" in m:
                    event_type = EVENT_TYPE["VE"]
                elif "AFIB" in m:
                    event_type = EVENT_TYPE["AF"]
                elif "TACHY" in m or "BRADY" in m or "PAUSE" in m or "AVB" in m:
                    event_type = EVENT_TYPE["SINUS"]

        ann = wf.rdann(file_name, extension='atr')
        record = wf.rdsamp(file_name, channels=[channel])
        buf_record = np.nan_to_num(record[0][:, 0])

        if fs_origin != SAMPLING_RATE:
            buf_ecg, ann = resample_singlechan(buf_record,
                                               ann,
                                               ann.fs,
                                               SAMPLING_RATE)

        else:
            buf_ecg = buf_record.copy()

        beats, types = beat_annotations(ann)

        buf_ecg = butter_bandpass_filter(buf_ecg, 0.5, 40, SAMPLING_RATE)
        if USED_NORM:
            buf_ecg = norm(buf_ecg, int(0.5 * SAMPLING_RATE))

        buf_ecg = ecg_to_tokens(buf_ecg)

        buf_ecg = buf_ecg[:(len(buf_ecg) // FEATURE_LEN) * FEATURE_LEN]
        buf_ecg_size = len(buf_ecg)
        index_label = np.arange(buf_ecg_size)
        buf_label = np.zeros(buf_ecg_size, np.int64)
        chk = list(np.where(np.diff(beats) < MIN_RR_LEN))
        if len(chk[0]) > 0:
            beats = np.delete(beats, chk[0])
            types = np.delete(types, chk[0])

        for beat, btype in zip(beats, types):
            # buf_label += (np.astype(((index_label - OFFSET_LABEL < beat) == (beat <= index_label + OFFSET_LABEL)), np.int64) * (HEARTBEAT_TYPE[btype]))
            buf_label += (
                    ((index_label - OFFSET_LABEL < beat) & (beat <= index_label + OFFSET_LABEL))
                    .astype(np.int64) * HEARTBEAT_TYPE[btype]
            )

        if ((stopSample - startSample) < (10 * SAMPLING_RATE) or
                stopSample > buf_ecg_size or
                (stopSample - startSample) > (20 * SAMPLING_RATE)):
            frame_ecg = []
            frame_label = []
            # process_frame_label = []
            # process_event_label = []
        else:
            if (stopSample - startSample) % FEATURE_LEN != 0 and (
                    stopSample - startSample) // FEATURE_LEN == 1:
                stopSample = startSample + FEATURE_LEN
            elif (stopSample - startSample) % FEATURE_LEN != 0 and (
                    stopSample - startSample) // FEATURE_LEN == 2:
                stopSample = startSample + FEATURE_LEN * 2

            # Check beats in start/stop sample have the same with beat from model.
            # index_correct = np.where(
            #     (beats > startSample) & (beats < stopSample))[0]
            # beats = beats[index_correct]

            frame_ecg = buf_ecg[startSample: stopSample]
            frame_label = buf_label[startSample: stopSample]

            # frame_ecg = np.round((scale_number) * minmax_scale(frame_ecg), 0)
            frame_ecg = np.reshape(frame_ecg, (-1, FEATURE_LEN))
            # frame_ecg = np.expand_dims(frame_ecg, axis=0)
            # frame_ecg = np.expand_dims(frame_ecg, axis=-1)
            with torch.no_grad():
                # x = encode_model.encode_run(np.expand_dims(_data, axis=-1)).to(device)
                # logits = downstream_model.classify(x)
                # _data = torch.from_numpy(_data).to(device).float()
                _data = torch.from_numpy(frame_ecg).to(device).long()
                # _data = torch.tensor(_data, dtype=torch.long, device=device)
                # logits = beat_classify(_data.unsqueeze(-1))
                logits = m_infer.get_logits(_data)

                group_beat_candidate = np.argmax(logits.cpu().detach().numpy(), axis=-1)
            # _ecg_signal = _data.flatten().cpu().numpy()
            beats_predict, symbols_predict = get_peaks(ecg_signal=_data.flatten().cpu().numpy(),
                                       group_beat_candidate=group_beat_candidate.flatten(),
                                       reprocess=False)

            # Plot the ECG signal and the predicted beats and symbols
            # plot_ecg_with_annotations(ecg_signal, beats_predict, symbols_predict)
            # transfer data to the original frequency
            beats_predict = (beats_predict * fs_origin) // SAMPLING_RATE

            chk = list(np.where(np.diff(beats_predict) < ((MIN_RR_LEN * fs_origin) // SAMPLING_RATE)))
            if len(chk[0]) > 0:
                beats_predict = np.delete(beats_predict, chk[0])
                symbols_predict = np.delete(symbols_predict, chk[0])

            if len(beats_predict) != 0:
                wf.wrann(record_name=os.path.basename(file_name),
                         extension="new",
                         sample=beats_predict,
                         symbol=symbols_predict,
                         write_dir=os.path.dirname(file_name))





if __name__ == '__main__':
    error_log_file = '/media/server2/MegaDataset/DataTraining/Log_File_Error/error_event_paths_data_training.txt'
    study_path = "/media/server2/MegaDataset/DataTraining/LabelCorrect/"
    # study_path = "/media/server2/MegaDataset/DataTraining/strip-include-to-report-10_11_12-2024/"
    get_peak_symbols()
    """
    '/media/server2/MegaDataset/DataTraining/LabelCorrect/373490/671a3569f734811533417e4c/event-strip-captured-2024-10-11-06-23-38-utc-04'
    """