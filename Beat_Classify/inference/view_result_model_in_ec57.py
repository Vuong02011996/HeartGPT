import numpy as np
import wfdb as wf
from Beat_Classify.define import path2db, path_physionet, sampling_rate, vocab_size, before, after
from Beat_Classify.helper import butter_bandpass_filter
from sklearn.preprocessing import minmax_scale  # for rescaling
from wfdb.processing import resample_sig

"""
Show all beat error between atr vs ai
+ read all beat in atr vs ai
+ For beat atr or ai(because beat ec57 have available so make sure beat and atr map exactly position)
+ Depend on condition Sn, Vn, ...Vo take beat in position and cut ecg signal in that position and save to file .npy
"""

def process_one_file_mitdb(file_name):
    # Read data
    file_name = path_physionet + file_name
    print(f"Processing in file {file_name}")
    signal = wf.rdrecord(file_name, channels=[0]).p_signal[:, 0]
    annotation_atr = wf.rdann(file_name, extension="atr")
    annotation_ai = wf.rdann(file_name, extension="ai")
    header = wf.rdheader(file_name)
    fs_origin = header.fs
    r_peaks_atr, labels_atr = annotation_atr.sample, np.array(annotation_atr.symbol)
    # Process beat atr to the same with beat model ai detect
    # remove non-beat labels
    invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x']
    indices = [i for i, label in enumerate(labels_atr) if label not in invalid_labels]
    r_peaks, labels = r_peaks_atr[indices], labels_atr[indices]

    # for correct R-peak location
    tol = 0.05
    newR = []
    for r_peak in r_peaks:
        r_left = np.maximum(r_peak - int(tol * sampling_rate), 0)
        r_right = np.minimum(r_peak + int(tol * sampling_rate), len(signal))
        newR.append(r_left + np.argmax(signal[r_left:r_right]))
    r_peaks = np.array(newR, dtype="int")
    AAMI = {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # N
        # "A": 2, "a": 2, "S": 2, "J": 2,  # SVEB
        "A": 1, "a": 1, "S": 1, "J": 1,  # SVEB
        # "V": 1, "E": 1,  # VEB
        "V": 2, "E": 2,  # VEB
        "F": 3,  # F
        "/": 4, "f": 4, "Q": 4  # Q
    }
    categories = [AAMI[label] for label in labels]
    symbols = labels

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
    index_remove_r_peaks = []

    for i in range(len(r_peaks)):

        if categories[i] not in [0, 2]:
            index_remove_r_peaks.append(i)
            continue

        if i == 0 or i == len(r_peaks) - 1:
            index_remove_r_peaks.append(i)
            continue

        if categories[i] == 4:  # remove AAMI Q class
            index_remove_r_peaks.append(i)
            continue
        window_peak = signal[max(0, r_peaks[i] - before): min(r_peaks[i], len(signal)) + after]
        # if categories[i] == 1:
        #     plot_signal(window_peak)
        if len(window_peak) != before + after:
            index_remove_r_peaks.append(i)
            continue

        # plot_signal(window_peak)

        signals.append(window_peak)
        # labels.append(categories[i])
        labels.append(symbols[i])
    signals = np.asarray(signals)
    labels = np.asarray(labels)
    r_peaks = np.delete(r_peaks, index_remove_r_peaks)
    # Get peak of model ai
    r_peaks_ai, labels_ai = annotation_ai.sample, np.array(annotation_ai.symbol)
    diff_peak = np.setdiff1d(r_peaks, r_peaks_ai)
    diff_label = np.setdiff1d(labels, labels_ai)
    print(labels[:10])
    print(labels_ai[:10])


    a = 0


if __name__ == '__main__':
    all_records = [
        # '101',
        # '106',
        '118',
        # '108', '109', '112', '114', '115', '116', '118', '119',
        # '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
        # '223', '230',

        # '100',
        # '103', '105', '111', '113', '117', '121', '123', '200', '202',
        # '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
        # '233', '234'
    ]

    for file_name in all_records:
        process_one_file_mitdb(file_name)