from Data.data_process import read_data
from sklearn.preprocessing import minmax_scale  # for rescaling
from wfdb.processing import resample_sig, resample_singlechan
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

sampling_rate = 100

def plot_signal(signal):
    # # region debug
    t = np.arange(0, len(signal), 1) / sampling_rate
    plt.plot(t, signal)
    plt.show()

def pre_process_data(file_name='100'):
    data = read_data(file_name)
    signal = data["signal"]
    r_peaks = data["r_peaks"]
    categories = data["categories"]
    fs_origin = data["fs_origin"]
    print("fs_origin: ", fs_origin)

    # heartbeat segmentation interval
    before, after = 250, 250
    # Resample to sampling_rate
    signal, _ = resample_sig(signal,fs_origin, sampling_rate)
    r_peaks = (r_peaks * sampling_rate) // fs_origin

    # scale 0 -> 100
    signal = np.round(100 * minmax_scale(signal), 0)


    signals = []
    labels = []
    for i in range(len(r_peaks)):
        if i == 0 or i == len(r_peaks) - 1:
            continue

        if categories[i] == 4:  # remove AAMI Q class
            continue
        window_peak = signal[max(0, r_peaks[i] - before): min(r_peaks[i], len(signal)) + after]
        # if categories[i] == 1:
        #     plot_signal(window_peak)
        if len(window_peak) != before + after:
            continue

        signals.append(window_peak)
        labels.append(categories[i])

    return signals, labels

def pre_process_data_follow_type(file_name='100', type_beat = 0):
    data = read_data(file_name)
    signal = data["signal"]
    r_peaks = data["r_peaks"]
    categories = data["categories"]
    fs_origin = data["fs_origin"]
    print("fs_origin: ", fs_origin)

    # heartbeat segmentation interval
    before, after = 250, 250
    # Resample to sampling_rate
    signal, _ = resample_sig(signal,fs_origin, sampling_rate)
    r_peaks = (r_peaks * sampling_rate) // fs_origin

    # scale 0 -> 100
    signal = np.round(100 * minmax_scale(signal), 0)


    signals = []
    labels = []
    for i in range(len(r_peaks)):
        if i == 0 or i == len(r_peaks) - 1:
            continue

        if categories[i] == 4:  # remove AAMI Q class
            continue
        window_peak = signal[max(0, r_peaks[i] - before): min(r_peaks[i], len(signal)) + after]
        # if categories[i] == 1:
        #     plot_signal(window_peak)
        if len(window_peak) != before + after:
            continue
        if type_beat == categories[i]:
            signals.append(window_peak)
            labels.append(categories[i])

    return signals, labels

def run_data(train_records, split='train'):
    all_windows = None
    all_labels = None

    for file in train_records:
        signals, labels = pre_process_data(file)
        if all_windows is None:
            all_windows = signals
            all_labels = labels
        else:
            all_windows = np.concatenate((all_windows, signals), axis=0)
            all_labels = np.concatenate((all_labels, labels), axis=0)


    # Lưu X_shuffled và y_shuffled vào file .npy
    path_save = '/Data/Data_ECG/'
    np.save(path_save + f'all_windows_{split}.npy', all_windows)
    np.save(path_save + f'all_labels_{split}.npy', all_labels)

def run_data_follow_type(records, split='train'):


    types_beat = [0, 1, 2, 3]
    symbols = ['N', 'S', 'V', "F"]
    for i, type_beat in enumerate(types_beat):
        all_windows = None
        all_labels = None
        for file in records:
            signals, labels = pre_process_data_follow_type(file, type_beat)
            if len(signals) == 0:
                continue
            if all_windows is None:
                all_windows = signals
                all_labels = labels
            else:
                all_windows = np.concatenate((all_windows, signals), axis=0)
                all_labels = np.concatenate((all_labels, labels), axis=0)

        # Lưu X_shuffled và y_shuffled vào file .npy
        path_save = '/Data/Data_ECG/'
        np.save(path_save + f'all_windows_{split}_{symbols[i]}.npy', all_windows)
        np.save(path_save + f'all_labels_{split}_{symbols[i]}.npy', all_labels)

def save_data_training():
    train_records = [
        '101',
        '106',
        '108', '109', '112', '114', '115', '116', '118', '119',
        '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
        '223', '230',

        '100',
        '103', '105', '111', '113', '117', '121', '123', '200', '202',
        '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
        '233', '234'
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
    # save_data_training()
    load_data_save()