import math
import warnings
import numpy as np

from numpy.typing import NDArray
from scipy.ndimage.filters import maximum_filter1d

from scipy.signal import (
    butter,
    filtfilt,
    iirnotch,
    iirfilter,
    sosfilt,
    zpk2sos
)


def beat_annotations(
        annotation
) -> [NDArray, NDArray]:
    """ Get rid of non-beat markers """
    # good = ['N', 'L', 'R', 'A', 'V', '/', 'a', '!', 'F', 'j', 'f', 'E', 'J', 'e', 'Q', 'S']
    good = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
    # normal = ['N', 'L', 'R', 'j', 'e']
    # supra_ventricular = ['a', 'S', 'A', 'J']
    # ventricular = ['!', 'E', 'V']
    # fusion = ['F']
    # unknow = ['P', 'Q', 'f']

    ids = np.isin(annotation.symbol, good)
    samples = annotation.sample[ids]
    symbols = np.asarray(annotation.symbol)[ids]

    return samples, symbols


def smooth(
        x,
        window_len: int = 11,
        window:     str = 'hanning'
):
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window_len % 2 == 0:
        window_len += 1

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y[int(window_len / 2):-int(window_len / 2)]


def bwr(
        raw:    NDArray,
        fs:     int,
        l1:     float = 0.2,
        l2:     float = 0.6
) -> NDArray:
    
    f1 = int(l1 * fs / 2)
    f2 = int(l2 * fs / 2)

    if f1 % 2 == 0:
        f1 += 1

    if f2 % 2 == 0:
        f2 += 1

    out1 = smooth(raw, f1)
    out2 = smooth(out1, f2)

    return raw - out2


def bwr_smooth(
        raw:    NDArray,
        fs:     int,
        l1:     float = 0.2,
        l2:     float = 0.6
) -> NDArray:
    f1 = int(l1 * fs / 2)
    f2 = int(l2 * fs / 2)

    if f1 % 2 == 0:
        f1 += 1

    if f2 % 2 == 0:
        f2 += 1

    out1 = smooth(raw, f1)
    out2 = smooth(out1, f2)

    return raw - out2


def norm(
        raw:        NDArray,
        window_len: float | int,
) -> NDArray:
    
    if window_len % 2 == 0:
        window_len += 1

    abs_raw = np.abs(raw)
    while True:
        g = maximum_filter1d(abs_raw, size=window_len)
        if np.max(abs_raw) < 5.0:
            break

        abs_raw[g > 5.0] = 0

    g_smooth = smooth(g, window_len, window='hamming')
    g_mean = max(np.mean(g_smooth) / 2.0, 0.1)
    g_smooth = np.clip(g_smooth, g_mean, None)
    g_smooth[g_smooth < 0.01] = 1
    normalized = np.divide(raw, g_smooth)

    return normalized


def norm2(
        raw:        NDArray, 
        baseline:   NDArray, 
        window_len: int, 
        fs:         int
) -> NDArray:
    
    if window_len % 2 == 0:
        window_len += 1

    abs_raw = abs(raw)

    baseline = smooth(baseline, window_len=int(2.5 * fs), window='hanning')
    baseline += np.median(abs_raw) / 2
    crossings = raw - baseline
    start_crossings = len(np.flatnonzero(np.diff(np.sign(crossings))))

    num_up_crossings = start_crossings
    while num_up_crossings > (0.1 * start_crossings):
        baseline = baseline + 0.05
        crossings = raw - baseline
        num_up_crossings = len(np.flatnonzero(np.diff(np.sign(crossings))))

    g = maximum_filter1d(abs_raw, size=window_len)
    g_smooth = smooth(g, window_len, window='hamming')
    g_mean = np.mean(baseline) / 2.0
    g_max = np.mean(baseline)
    g_smooth = np.clip(g_smooth, g_mean, g_max)
    g_smooth[g_smooth < 0.01] = 1
    normalized = np.divide(raw, g_smooth)

    return normalized


def butter_lowpass_filter(
        signal:     NDArray,
        cutoff:     float,
        fs:         int,
        order:      int = 5,
        padlen:     int = None
) -> NDArray:
    
    (b, a) = butter(
            N=order,
            Wn=cutoff / (fs / 2),
            btype='low',
            analog=False
    )
    y = filtfilt(b, a, signal, padlen=padlen)
    
    return y


def butter_highpass_filter(
        signal:     NDArray,
        cutoff:     float,
        fs:         int,
        order:      int = 5,
        padlen:     int = None,
        
):
    (b, a) = butter(
            N=order,
            Wn=cutoff / (fs / 2),
            btype='high',
            analog=False
    )
    y = filtfilt(b, a, signal, padlen=padlen)
    
    return y


def butter_bandpass_filter(
        signal:     NDArray,
        lowcut:     float,
        highcut:    float,
        fs:         int,
        order:      int = 5,
        padlen:     int = None
) -> NDArray:
    
    b, a = butter(
            N=order,
            Wn=[lowcut / (fs / 2), highcut / (fs / 2)],
            btype='band'
    )
    y = filtfilt(b, a, signal, padlen=padlen)

    return y


def butter_notch_filter(
        x:      NDArray,
        fs_cut: int,
        fs:     int,
        q:      float = 30.0
) -> NDArray:
    
    b, a = iirnotch(w0=fs_cut / (fs / 2), Q=q)
    y = filtfilt(b, a, x)

    return y


def multi_butter_bandpass_filter(
        signal:     NDArray,
        low_cut:    float,
        high_cut:   float,
        fs:         int,
        order:      int = 5,
        padlen:     int = None
) -> NDArray:
    
    (b, a) = butter(
            N=order,
            Wn=[low_cut / (fs / 2), high_cut / (fs / 2)],
            btype='band'
    )
    
    y = np.vstack(list(map(
            lambda x: filtfilt(b, a, signal[:, x], padlen=padlen),
            range(len(signal))
    ))).T

    return y


def highpass(
        signal:     NDArray,
        freq:       int,
        fs:         int,
        corners:    int = 4,
        zerophase:  bool = False
) -> NDArray:
    
    fe = 0.5 * fs
    f = freq / fe
    if f > 1:
        msg = "Selected corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, f, btype='highpass', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, float(k))
    
    if zerophase:
        ecg_filter = sosfilt(sos, sosfilt(sos, signal)[::-1])[::-1]
    else:
        ecg_filter = sosfilt(sos, signal)
        
    return ecg_filter


def iir_bandpass(
        signal:     NDArray,
        freqmin:    float,
        freqmax:    float,
        fs:         int,
        corners:    int = 4,
        zerophase:  bool = True
) -> NDArray:
    
    fe = 0.5 * fs
    low = freqmin / fe
    high = freqmax / fe
    
    if high - 1.0 > -1e-6:
        msg = (f"Selected high corner frequency ({freqmax}) of bandpass is at or "
               f"above Nyquist ({fe}). Applying a high-pass instead.")
        warnings.warn(msg)
        
        return highpass(
                signal,
                freq=freqmin,
                fs=fs,
                corners=corners,
                zerophase=zerophase
        )
    if low > 1:
        raise ValueError("Selected low corner frequency is above Nyquist.")
    
    z, p, k = iirfilter(corners, [low, high], btype='band', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, float(k))

    if zerophase:
        ecg_filter = sosfilt(sos, sosfilt(sos, signal)[::-1])[::-1]
    else:
        ecg_filter = sosfilt(sos, signal)
    
    return ecg_filter

