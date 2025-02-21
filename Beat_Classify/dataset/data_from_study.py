import os
import glob
import wfdb as wf
from sklearn.preprocessing import minmax_scale  # for rescaling
from wfdb.processing import resample_sig, resample_singlechan
import numpy as np
from scipy.signal import (
    butter,
    filtfilt,
    iirnotch,
    iirfilter,
    sosfilt,
    zpk2sos
)
from sklearn.model_selection import train_test_split
import shutil


import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


def butter_bandpass_filter(
        signal,
        lowcut: float,
        highcut: float,
        fs: int,
        order: int = 5,
        padlen: int = None
):
    b, a = butter(
        N=order,
        Wn=[lowcut / (fs / 2), highcut / (fs / 2)],
        btype='band'
    )
    y = filtfilt(b, a, signal, padlen=padlen)

    return y


def plot_signal(signal, symbol=None):
    plt.figure()
    plt.plot(signal, label='Signal')

    # Calculate the middle index
    middle_index = len(signal) // 2

    # Plot the red point at the middle index
    plt.plot(middle_index, signal[middle_index], 'ro', label='Middle Point')
    # plt.plot(peak, signal[peak], 'bo', label='Middle Point')
    if symbol is not None:
        plt.suptitle(str(symbol))
    plt.legend()
    plt.show()


def plot_info_event(ecg_signal, beats, symbols, start_highlight, stop_highlight, comment, show_highlight=False):
    BEAT_COLORS = {
        'N': 'white',
        'S': 'orange',
        'A': 'orange',
        'V': 'blue',
        '|': 'purple',
        'Others': 'purple',
        'Q': 'purple',
        'R': 'cyan',
        'M': 'olive'
    }

    BEAT_COLORS_EC = {
        'NOTABEAT': 'grey',
        'N': 'black',
        'S': 'orange',
        'A': 'orange',
        'V': 'blue',
        '|': 'purple',
        'Others': 'purple',
        'Q': 'purple',
        'R': 'white',
        'M': 'white'
    }

    y_max = max(ecg_signal)
    plt.plot(ecg_signal)
    plt.plot(beats, ecg_signal[beats], 'ro')
    plt.vlines(beats, ymin=-y_max - 0.3, ymax=y_max, lw=0.5, color='r', linestyles='dotted')

    [
        plt.annotate(
            s,
            xy=(b, y_max),
            xycoords='data',
            textcoords='data',
            bbox=dict(
                boxstyle='round',
                fc=BEAT_COLORS[s],
                ec=BEAT_COLORS_EC[s]
            )
        )
        for b, s in zip(beats, symbols)
    ]

    if show_highlight:
        plt.axvspan(start_highlight, stop_highlight, color='yellow', alpha=0.5)
    plt.suptitle(comment)
    plt.show()
    plt.close()


def get_result_from_technical(file):
    before, after = 250, 250
    extend_signal = 500
    vocab_size = 1000
    sampling_rate = 100
    window_peaks = []
    labels = []

    try:
        header = wf.rdheader(file)
        if len(header.sig_name) > 3:
            print("event_have_header_file_error")
            return window_peaks, labels
    except Exception as e:
        print("event_have_header_file_error")
        return window_peaks, labels
    cmt = ""
    startSample = 0
    stopSample = 0
    fs = 250
    ann = wf.rdann(file, "atr")
    _cmt = []
    comment = "None"
    for cmt in header.comments:
        if "comment:" in cmt:
            comment = cmt.split(": ")[-1]
            _cmt.append(cmt)
        elif "startSample:" in cmt:
            startSample = int(cmt.split(": ")[-1])
        elif "stopSample:" in cmt:
            stopSample = int(cmt.split(": ")[-1])
        elif "channel:" in cmt:
            channel = int(cmt.split(": ")[-1])
            _cmt.append(cmt)
        elif "studyID: " in cmt:
            studyID = cmt
            _cmt.append(cmt)
        elif "eventID: " in cmt:
            eventID = cmt
            _cmt.append(cmt)
        else:
            _cmt.append(cmt)
    # Get signal
    record = wf.rdrecord(file)
    fs_origin = header.fs
    assert record.fs == fs_origin

    channel = 1
    for cmt in record.comments:
        if "channel:" in cmt:
            channel = int(cmt.split(": ")[-1])
    # Error when hea have one chanel
    if record.p_signal.shape[1] > 1:
        ecg_signals = np.nan_to_num(record.p_signal)[:, channel]
    else:
        ecg_signals = np.nan_to_num(record.p_signal)[:, 0]

    # Bandpass filter
    ecg_signals = butter_bandpass_filter(ecg_signals, 1, 40, fs)

    # get data ann
    symbols = ann.symbol
    r_peaks = ann.sample
    # plot_info_event(ecg_signals, r_peaks, symbols, startSample, stopSample, comment, show_highlight=True)

    # Resample from 250->100
    signal, _ = resample_sig(ecg_signals, fs_origin, sampling_rate)
    # Scale signal to 0->255
    signal = np.round((vocab_size) * minmax_scale(signal), 0)

    r_peaks = (r_peaks * sampling_rate) // fs_origin
    startSample = (startSample * sampling_rate) // fs_origin
    stopSample = (stopSample * sampling_rate) // fs_origin
    # plot_info_event(signal, r_peaks, symbols, startSample, stopSample, comment, show_highlight=True)

    # Take signal to sure window peak start and end > 250

    startSample = startSample - extend_signal
    stopSample = stopSample + extend_signal
    highlight_signal = signal[startSample:stopSample]
    highlight_sample = r_peaks[np.flatnonzero((r_peaks >= startSample) & (r_peaks <= stopSample))] - startSample
    highlight_symbol = np.asarray(symbols)[np.flatnonzero((r_peaks >= startSample) & (r_peaks <= stopSample))]
    # plot_info_event(highlight_signal, highlight_sample, highlight_symbol, startSample, stopSample, comment,
    #                 show_highlight=False)

    r_peaks = highlight_sample
    signal = highlight_signal
    symbols = highlight_symbol
    assert len(r_peaks) == len(symbols)


    for i in range(len(r_peaks)):
        if r_peaks[i] - extend_signal < 0 or r_peaks[i] + extend_signal >= len(signal):
            continue
        window_peak = signal[r_peaks[i] - before : r_peaks[i] + after]
        # plot_signal(window_peak, symbols[i])
        # print(i)
        window_peaks.append(window_peak)
        labels.append(symbols[i])
        # if categories[i] == 1:
        #     plot_signal(window_peak)
        # if len(window_peak) != before + after:
        #     continue
        # if type_beat == categories[i]:
        #     signals.append(window_peak)
        #     labels.append(categories[i])
    return window_peaks, labels

def process_study(path_study):
    path_study += '/'
    study_id = os.path.basename(os.path.normpath(path_study))
    # List all event in study
    list_folder = [os.path.join(path_study, i) for i in os.listdir(path_study) if
                   os.path.isdir(os.path.join(path_study, i))]
    print("Total event_id: {}".format(len(list_folder)))
    window_peaks_study, labels_study = [], []
    for event_path in list_folder:
        # For debug
        # if not event_path == '/media/server2/MegaDataset/Vuong_Data/strips/389423/67482df9830efa000178a73f':
        #     continue

        print("Processing event ID: ", event_path)
        list_files = [i[:-4] for i in glob.glob(event_path + "/*.hea")]

        # Event no data -> continue
        if len(list_files) == 0:
            print("event_id_no_header_file")
            continue

        file = list_files[0]
        # For each event in the study create one folder to save the result and run bxb in that folder
        save_path = event_path + '/' + study_id + '_' + os.path.basename(event_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = study_id + '_' + os.path.basename(event_path)

        # Get user result
        window_peaks, labels = get_result_from_technical(file)
        # Only check event have peaks
        if len(window_peaks) > 0:
            assert len(window_peaks) == len(labels)
            if len(window_peaks_study) == 0:
                window_peaks_study = window_peaks
                labels_study = labels
            else:
                window_peaks_study = np.concatenate((window_peaks_study, window_peaks), axis=0)
                labels_study = np.concatenate((labels_study, labels), axis=0)
        else:
            print(f"Event {event_path} no beat")
    return window_peaks_study, labels_study

def process_and_saved_data(list_study_train, path_save, split):
    all_windows_train = []
    all_labels_train = []
    for path_study in list_study_train:
        print("Processing study: ", path_study)
        # For debug
        # path_study = '/media/server2/MegaDataset/Vuong_Data/strips/389423'
        window_peaks_study, labels_study = process_study(path_study)
        # Only check study have peak
        if len(window_peaks_study) > 0:
            if len(all_windows_train) == 0:
                all_windows_train = window_peaks_study
                all_labels_train = labels_study
            else:
                all_windows_train = np.concatenate((all_windows_train, window_peaks_study), axis=0)
                all_labels_train = np.concatenate((all_labels_train, labels_study), axis=0)
    # unique_peaks = set(all_labels_train)
    types_beat = [0, 1, 2]
    symbols = ['N', 'S', 'V']
    for i, type_beat in enumerate(types_beat):
        # how to get all index in array all_labels_train have value = s
        all_windows = all_windows_train[np.where(all_labels_train == symbols[i])]
        all_labels = all_labels_train[np.where(all_labels_train == symbols[i])]
        # Convert all_labels from  ['N', 'S', 'V'] to [0, 1, 2]
        all_labels = np.array([symbols.index(label) for label in all_labels])
        np.save(path_save + f'all_windows_{split}_{symbols[i]}.npy', all_windows)
        np.save(path_save + f'all_labels_{split}_{symbols[i]}.npy', all_labels)


def main():
    # List all study in folder_data
    list_study = [os.path.join(folder_data, i) for i in os.listdir(folder_data) if
                   os.path.isdir(os.path.join(folder_data, i))]
    list_study = list_study[:]
    # Split the list into 90% training and 10% testing
    list_study_train, list_study_test = train_test_split(list_study, test_size=0.1, random_state=42)
    print("Training set:", list_study_train)
    print("Testing set:", list_study_test)
    # I convert list_study_test to numpy array to use np.save , save list_study_test
    list_study_test = np.array(list_study_test)
    np.save('/home/server2/Desktop/Vuong/Reference_Project/HeartGPT/Data/list_study_test.npy', list_study_test)
    # How to load list_study_test and check it
    list_study_test_save = np.load('/home/server2/Desktop/Vuong/Reference_Project/HeartGPT/Data/list_study_test.npy')
    # How to check file list_study_test_save the same list_study_test
    assert (list_study_test == list_study_test_save).all()

    path_save = '/home/server2/Desktop/Vuong/Reference_Project/HeartGPT/Data/Data_Study/'
    # Check if the folder save data exists
    if os.path.exists(path_save):
        shutil.rmtree(path_save)
    os.makedirs(path_save)
    process_and_saved_data(list_study_train, path_save, split='train')
    process_and_saved_data(list_study_test, path_save, split='test')




if __name__ == '__main__':
    folder_data = "/media/server2/MegaDataset/Vuong_Data/strips/"
    main()
