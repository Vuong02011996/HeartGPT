from scipy.signal import (
    butter,
    filtfilt,
    iirnotch,
    iirfilter,
    sosfilt,
    zpk2sos
)

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
