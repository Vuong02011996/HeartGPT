from Beat_Classify.helper import butter_bandpass_filter
from Beat_Classify.dataset.data_process import read_data
from sklearn.preprocessing import minmax_scale  # for rescaling
from wfdb.processing import resample_sig
import numpy as np
import wfdb as wf

from Beat_Classify.define import vocab_size, before, after, sampling_rate, path_save_data, path_physionet


def process_one_file_mitdb(file_name, type_beat):
    # Read data
    file_name = path_physionet + file_name
    print(f"Processing in file {file_name}")
    signal = wf.rdrecord(file_name, channels=[0]).p_signal[:, 0]
    annotation = wf.rdann(file_name, extension="atr")
    header = wf.rdheader(file_name)
    fs_origin = header.fs
    r_peaks, labels = annotation.sample, np.array(annotation.symbol)

    # remove non-beat labels
    invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x']
    indices = [i for i, label in enumerate(labels) if label not in invalid_labels]
    r_peaks, labels = r_peaks[indices], labels[indices]

    # for correct R-peak location
    tol = 0.05
    newR = []
    for r_peak in r_peaks:
        r_left = np.maximum(r_peak - int(tol * sampling_rate), 0)
        r_right = np.minimum(r_peak + int(tol * sampling_rate), len(signal))
        newR.append(r_left + np.argmax(signal[r_left:r_right]))
    r_peaks = np.array(newR, dtype="int")

    # AAMI categories
    # AAMI = {
    #     "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # N
    #     "A": 1, "a": 1, "S": 1, "J": 1,  # SVEB
    #     "V": 2, "E": 2,  # VEB
    #     "F": 3,  # F
    #     "/": 4, "f": 4, "Q": 4  # Q
    # }

    AAMI = {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # N
        "A": 2, "a": 2, "S": 2, "J": 2,  # SVEB
        "V": 1, "E": 1,  # VEB
        "F": 3,  # F
        "/": 4, "f": 4, "Q": 4  # Q
    }
    categories = [AAMI[label] for label in labels]

    # heartbeat segmentation interval
    # Resample to sampling_rate
    signal = butter_bandpass_filter(signal, 1, 40, 250)
    signal, _ = resample_sig(signal, fs_origin, sampling_rate)
    r_peaks = (r_peaks * sampling_rate) // fs_origin

    # scale 0 -> 100
    signal = np.round((vocab_size - 1) * minmax_scale(signal), 0)
    # plot_signal(signal)

    signals = []
    labels = []
    for i in range(len(r_peaks)):
        if i == 0 or i == len(r_peaks) - 1:
            continue
        if type_beat == categories[i]:
            window_peak = signal[max(0, r_peaks[i] - before): min(r_peaks[i], len(signal)) + after]
            if len(window_peak) != before + after:
                continue
            signals.append(window_peak)
            labels.append(categories[i])
    signals = np.asarray(signals)
    labels = np.asarray(labels)

    return signals, labels


def run_data_follow_type(records, split='train'):
    # types_beat = [0, 1, 2, 3]
    types_beat = [0, 1]
    # symbols = ['N', 'S', 'V', "F"]
    symbols = ['N', 'V']
    for i, type_beat in enumerate(types_beat):
        all_windows = None
        all_labels = None
        for file in records:
            signals, labels = process_one_file_mitdb(file, type_beat)
            if len(signals) == 0:
                continue
            if all_windows is None:
                all_windows = signals
                all_labels = labels
            else:
                all_windows = np.concatenate((all_windows, signals), axis=0)
                all_labels = np.concatenate((all_labels, labels), axis=0)

        # Lưu X_shuffled và y_shuffled vào file .npy
        np.save(path_save_data + f'all_windows_{split}_{symbols[i]}.npy', all_windows)
        np.save(path_save_data + f'all_labels_{split}_{symbols[i]}.npy', all_labels)

def process_data_training():
    train_records = [
        '101',
        '106',
        '108', '109', '112', '114', '115', '116', '118', '119',
        '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
        '223', '230',

        # '100',
        # '103', '105', '111', '113', '117', '121', '123', '200', '202',
        # '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
        # '233', '234'
    ]
    # test_records = [
    #     '100',
    #     '103', '105', '111', '113', '117', '121', '123', '200', '202',
    #     '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    #     '233', '234'
    # ]
    run_data_follow_type(train_records, split='train')
    # run_data(test_records, split='test_100')

        
def load_data_save():
    path_save = '/Data/Data_ECG/'
    types_beat = [0, 1, 2, 3]
    symbols = ['N', 'S', 'V', "F"]
    split = 'train'
    for i, type_beat in enumerate(types_beat):
        all_windows = np.load(path_save + f'all_windows_{split}_{symbols[i]}.npy')
        all_labels = np.load(path_save + f'all_labels_{split}_{symbols[i]}.npy')
        print(f'Type_{symbols[i]} have {len(all_labels)} sample')
        # for i in range(len(all_windows)):
        #     if all_labels[i] == 2:
        #         plot_signal(all_windows[i])

if __name__ == '__main__':
    # pre_process_data('100')
    process_data_training()
    # load_data_save()