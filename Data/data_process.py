import numpy
import wfdb as wf
from pathlib import Path
import numpy as np
import scipy.signal as sg
from sklearn.preprocessing import minmax_scale  # for rescaling
import pywt
import cv2
from sklearn.preprocessing import RobustScaler
import onnx
import torch
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import resample
import pandas as pd
import matplotlib
from wfdb.processing import resample_sig, resample_singlechan
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

PATH = '/home/server2/Desktop/Vuong/Data/PhysionetData/mitdb/'
path_data_save = "/home/server2/Desktop/Vuong/Reference_Project/HeartGPT/Data/Data_ECG/"
sampling_rate = 100

# non-beat labels
invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x']
# for correct R-peak location
tol = 0.05

def read_data(file_name):
    file_name = PATH + file_name
    print("Processing record: ", file_name)
    # read ML II signal & r-peaks position and labels
    # signal = wfdb.rdrecord((PATH / record).as_posix(), channels=[0]).p_signal[:, 0]
    # annotation = wfdb.rdann((PATH / record).as_posix(), extension="atr")

    signal = wf.rdrecord(file_name, channels=[0]).p_signal[:, 0]
    annotation = wf.rdann(file_name, extension="atr")
    header = wf.rdheader(file_name)
    fs_origin = header.fs
    r_peaks, labels = annotation.sample, np.array(annotation.symbol)

    # filtering uses a 200-ms width median filter and 600-ms width median filter
    # baseline = sg.medfilt(sg.medfilt(signal, int(0.2 * sampling_rate) - 1), int(0.6 * sampling_rate) - 1)
    # filtered_signal = signal - baseline
    filtered_signal = signal
    # Resample the signal
    # filtered_signal = resample(signal, 100)

    # remove non-beat labels
    indices = [i for i, label in enumerate(labels) if label not in invalid_labels]
    r_peaks, labels = r_peaks[indices], labels[indices]

    # align r-peaks
    newR = []
    for r_peak in r_peaks:
        r_left = np.maximum(r_peak - int(tol * sampling_rate), 0)
        r_right = np.minimum(r_peak + int(tol * sampling_rate), len(filtered_signal))
        newR.append(r_left + np.argmax(filtered_signal[r_left:r_right]))
    r_peaks = np.array(newR, dtype="int")

    # remove inter-patient variation
    normalized_signal = filtered_signal / np.mean(filtered_signal[r_peaks])

    # AAMI categories
    AAMI = {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # N
        "A": 1, "a": 1, "S": 1, "J": 1,  # SVEB
        "V": 2, "E": 2,  # VEB
        "F": 3,  # F
        "/": 4, "f": 4, "Q": 4  # Q
    }
    categories = [AAMI[label] for label in labels]
    data = {
        # "record": record,
        "signal": normalized_signal, "r_peaks": r_peaks, "categories": categories, "fs_origin": fs_origin
    }
    return data


def re_process_data(file_name, show=False):
    data = read_data(file_name)
    signal = data["signal"]
    r_peaks = data["r_peaks"]


    # if show:
    #     # # region debug
    #     t = np.arange(0, len(data["signal"]), 1) / sampling_rate
    #     plt.plot(t, data["signal"])
    #     # # Mapping dictionary
    #     # labels = data["categories"]
    #     # label_map = {0: 'N', 1: 'S', 2: 'V', 3: 'F'}
    #     # # Convert the array to labels
    #     # a_labels = [label_map[num] for num in labels]
    #     # # plt.plot(t, lbl_draw)
    #     # for b, s in zip(r_peaks, a_labels):
    #     #     plt.annotate(s, xy=(t[b], data["signal"][b]))
    #
    #     plt.show()

    # Rescale and round the signal
    signal = np.round(100 * minmax_scale(signal), 0)
    Y = numpy.zeros(len(signal))
    Y[r_peaks] = 1

     # Plot the signal with peak Y
    if show:
        t = np.arange(0, len(signal), 1) / sampling_rate
        plt.plot(t, signal, label='Signal')

        # Mark the points where Y == 1 with a circle
        plt.scatter(t[r_peaks], signal[r_peaks], color='red', label='R-peaks', marker='o')

        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Signal with R-peaks')
        plt.legend()
        plt.show()


    return signal, Y


def read_signal_save_csv(file_name):
    file_name = PATH + file_name
    print("Processing record: ", file_name)
    # read ML II signal & r-peaks position and labels
    # signal = wfdb.rdrecord((PATH / record).as_posix(), channels=[0]).p_signal[:, 0]
    # annotation = wfdb.rdann((PATH / record).as_posix(), extension="atr")

    signal = wf.rdrecord(file_name, channels=[0]).p_signal[:, 0]
        # Create a DataFrame to save the signal
        # Create a DataFrame to save the signal
    df = pd.DataFrame(signal[:500])
    output_csv = path_data_save + 'output_signal.csv'
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)



if __name__ == '__main__':
    train_records = [
        '101',
        # '106', '108', '109', '112', '114', '115', '116', '118', '119',
        # '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
        # '223', '230'
    ]
    test_records = [
        '100',
        '103', '105', '111', '113', '117', '121', '123', '200', '202',
        '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
        '233', '234'
    ]
    # for file in train_records:
    #     re_process_data(file)

    # read_signal_save_csv('101')
