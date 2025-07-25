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

from ECG_LLM.detail_model.plot_attention_wei import plot_attention_lines_horizontal
from ECG_LLM.process_data_training.config_dataset import EVENT_TYPE, SAMPLING_RATE, USED_NORM, FEATURE_LEN, MIN_RR_LEN, \
    OFFSET_LABEL, HEARTBEAT_TYPE, scale_number
from ECG_LLM.process_data_training.helper import ecg_to_tokens, _bbox_color
from ECG_LLM.process_data_training.reprocessing import beat_annotations, butter_bandpass_filter, norm

from ECG_LLM.define import model_path
from ECG_LLM.HearGPTModel_with_save_wei_attention import HeartGPTModel
from ECG_LLM.utils import get_peaks
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'

weights_matrices = []

# Pass the shared list to the model
model = HeartGPTModel(weights_matrices=weights_matrices)
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

    list_event_id_error = []
    with open(error_log_file, 'r') as file:
        for line in file:
            # parts = line.strip().split(os.sep)
            # if len(parts) > 2:  # Ensure the path has enough parts
            #     list_event_id_error.append(parts)  # Extract event_id from the path
            list_event_id_error.append(line.strip())

    for event_path in tqdm(all_events):
        """
        65beb4ddc69788a1bdb71b0b: batch = 2 True
        65c7bdf9ac9cbe99b56a1f2d: V True
        65c7bdf9ac9cbe99b56a1f2c: S True
        65c7bdf7ac9cbe99b56a1ed7: N True brady
        65beb4ddc69788a1bdb71b16: Error N reg S, Miss Q.
        65f5ada850ac063879d8e167: V reg S
        """
        if "65c7bdf9ac9cbe99b56a1f2d" not in event_path: # 65beb4ddc69788a1bdb71b0b, 65beb4ddc69788a1bdb71b16
            continue

        # if event_path in list_event_id_error:
        #     path_save = "/media/server2/MegaDataset/DataTraining/image_wei_attention/Error/"
        # else:
        #     # path_save = "/media/server2/MegaDataset/DataTraining/image_wei_attention/True/"
        #     continue

        event_id = event_path.split("/")[-2]
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

            frame_ecg = buf_ecg[startSample: stopSample]
            frame_label = buf_label[ startSample: stopSample]

            # frame_ecg = np.round((scale_number) * minmax_scale(frame_ecg), 0)
            frame_ecg = np.reshape(frame_ecg, (-1, FEATURE_LEN))
            with torch.no_grad():
                # x = encode_model.encode_run(np.expand_dims(_data, axis=-1)).to(device)
                # logits = downstream_model.classify(x)
                # _data = torch.from_numpy(_data).to(device).float()
                _data = torch.from_numpy(frame_ecg).to(device).long()
                # _data = torch.tensor(_data, dtype=torch.long, device=device)
                # logits = beat_classify(_data.unsqueeze(-1))
                logits = m_infer.get_logits(_data)

                group_beat_candidate = np.argmax(logits.cpu().detach().numpy(), axis=-1)
                a = 0
            # _ecg_signal = _data.flatten().cpu().numpy()
            beats_predict, symbols_predict = get_peaks(ecg_signal=_data.flatten().cpu().numpy(),
                                       group_beat_candidate=group_beat_candidate.flatten(),
                                       reprocess=False)

            chk = list(np.where(np.diff(beats_predict) < MIN_RR_LEN ))
            if len(chk[0]) > 0:
                beats_predict = np.delete(beats_predict, chk[0])
                symbols_predict = np.delete(symbols_predict, chk[0])

            mask = (beats >= startSample) & (beats <= stopSample)
            indices = np.where(mask)[0]

            labels_gt = types[indices]
            beats_gt = beats[indices] - startSample


            start_beat = max(beats_gt[0] - 9, 0)
            stop_beat = min(beats_gt[-1] + 9, beats_predict[-1])
            mask = (beats_predict >= start_beat) & (beats_predict <= stop_beat)
            indices = np.where(mask)[0]
            beats_predict = beats_predict[indices]
            symbols_predict = symbols_predict[indices]
            # Compare
            offset = int(0.075 * SAMPLING_RATE)
            # if isinstance(ref_beats, list):
            #     ref_beats = np.array(ref_beats)

            beat_dumps = (beats_gt[:, None] + np.arange(-offset, offset)).flatten()
            beat_dumps[beat_dumps < 0] = 0

            index = np.flatnonzero(~np.in1d(beats_predict, beat_dumps))

            fig, axs = plt.subplots(5, 1, figsize=(20, 12))

            # Plot the ECG signal
            ecg_tokens_signal = frame_ecg.flatten()
            axs[0].plot(ecg_tokens_signal, label="ECG Signal")
            axs[0].set_title("ECG Signal")
            axs[0].set_xlabel("Sample Index")
            axs[0].set_ylabel("Amplitude")
            axs[0].legend()
            [
                axs[0].annotate(
                    x,
                    xy=(beats_gt[j], max(ecg_tokens_signal)),
                    xycoords='data',
                    textcoords='data',
                    bbox=_bbox_color(x)
                )
                for j, x in enumerate(labels_gt)
            ]
            [
                axs[0].annotate(
                    x,
                    xy=(beats_predict[j], min(ecg_tokens_signal)),
                    xycoords='data',
                    textcoords='data',
                    bbox=_bbox_color(x)
                )
                for j, x in enumerate(symbols_predict)
            ]


            axs[1].plot(frame_label, label="Ground Label", color="red", linestyle="--")
            axs[1].plot(group_beat_candidate.flatten(), label="Model Predict", color="green", linestyle="--")
            axs[1].set_title("Frame Label (Model Output)")
            axs[1].set_xlabel("Sample Index")
            axs[1].set_ylabel("Label Value")
            axs[1].legend()

            idx_layer = -1

            # axs[2].plot(weights_matrices[idx_layer][0][740], label="wei token 740", color="green", linestyle="--")
            # axs[2].plot(weights_matrices[idx_layer][0][741], label="wei token in 741", color="red", linestyle="--")
            # axs[2].plot(weights_matrices[idx_layer][0][742], label="wei token in 742", color="yellow", linestyle="--")
            # axs[2].set_title("Attention Weights token position 740")
            # axs[2].set_xlabel("Key Index")
            # axs[2].set_ylabel("Query Index")
            # axs[2].legend()

            # colors = plt.cm.get_cmap('tab20', 64)  # Generate 64 unique colors
            # for i in range(64):
            #     axs[2].plot(weights_matrices[i][0][600], color=colors(i), linestyle="--")
            # axs[2].set_title("Attention Weights token position 600")
            # axs[2].set_xlabel("Key Index")
            # axs[2].set_ylabel("Query Index")
            # axs[2].legend(ncol=4, fontsize='small')

            colors = plt.cm.get_cmap('tab20', 64)  # Generate 64 unique colors
            # for i in range(64):
            axs[3].plot(weights_matrices[63][0][640], color=colors(63), linestyle="--")
            axs[3].set_title("Attention Weights token position 640")
            axs[3].set_xlabel("Key Index")
            axs[3].set_ylabel("Query Index")
            axs[3].legend(ncol=4, fontsize='small')


            # colors = plt.cm.get_cmap('tab20', 64)  # Generate 64 unique colors
            # for i in range(64):
            #     axs[4].plot(weights_matrices[i][0][641], color=colors(i), linestyle="--")
            # axs[4].set_title("Attention Weights token position 641")
            # axs[4].set_xlabel("Key Index")
            # axs[4].set_ylabel("Query Index")
            # axs[4].legend(ncol=4, fontsize='small')

            # axs[4].plot(weights_matrices[idx_layer][0][500], label="wei token 500", color="green", linestyle="--")
            # axs[4].plot(weights_matrices[idx_layer][0][520], label="wei token in 520", color="red", linestyle="--")
            # axs[4].plot(weights_matrices[idx_layer][0][480], label="wei token in 480", color="yellow", linestyle="--")
            # axs[4].set_title("Attention Weights token position 500")
            # axs[4].set_xlabel("Key Index")
            # axs[4].set_ylabel("Query Index")
            # axs[4].legend()
            # plt.tight_layout()
            # plt.show()

            # Lấy mỗi 5 token một lần
            tokens = frame_ecg.flatten()
            seq_len = len(tokens)
            attention_matrix = weights_matrices[-1][0]
            indices = list(range(0, seq_len, 10))
            reduced_tokens = [tokens[i] for i in indices]
            reduced_attention = attention_matrix[np.ix_(indices, indices)]

            # Gọi hàm vẽ với tập nhỏ hơn
            plot_attention_lines_horizontal(reduced_tokens, reduced_attention, max_links=500)


            # Adjust layout and display the plot

            # print("weights_matrices[idx_layer][0][640]: ", weights_matrices[idx_layer][0][640])
            # # Save the plot to a file instead of displaying it
            # save_path = os.path.join(path_save, f"{event_id}.png")
            # plt.savefig(save_path)
            # print(f"Plot saved to {save_path}")


        # Reset weights_matrices after processing each event_path
        weights_matrices.clear()

    print("error: ", error)
    print("len(all_events): ", len(all_events))
    print(f"error/all_events = {error}/{len(all_events)}")


if __name__ == '__main__':
    error_log_file = '/media/server2/MegaDataset/DataTraining/Log_File_Error/error_event_paths_data_training.txt'
    study_path = "/media/server2/MegaDataset/DataTraining/LabelCorrect/"
    # study_path = "/media/server2/MegaDataset/DataTraining/strip-include-to-report-10_11_12-2024/"

    get_peak_symbols()