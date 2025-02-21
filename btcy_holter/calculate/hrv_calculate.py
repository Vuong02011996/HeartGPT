from btcy_holter import *
from scipy.fftpack import ifft


ROFFERR:    Final[float] = 1e-10
NNDIF:      Final[float] = 0.05 * 1000

LENGTH:     Final[int] = 300


class HRVariability(
        object
):
    def __init__(
            self
    ):
        self.start_time      = 0
        self.length          = 0
        self.totnn           = 0
        self.totrr           = 0
        self.sdnn            = 0
        self.sdann           = 0
        self.rmssd           = 0
        self.pnn50           = 0
        self.tp              = 0
        self.vlf             = 0
        self.lf              = 0
        self.hf              = 0
        self.ulf             = 0
        self.ratio_lf_hf     = 0
        self.lf_norm         = 0
        self.hf_norm         = 0
        self.sdnnindx        = 0
        self.msd             = 0
        self.beats_use_hrv   = 0
        self.mean_rr         = 0
        self.max_rr          = 0
        self.min_rr          = -1
        self.cv              = 0


def __power(
        freq:   NDArray,
        mag:    NDArray
) -> Any:
    """
    Calculate total (and relative) power in fft between lo and hi.
    
    ULF – ultra low frequency band <0.003 Hz
    VLF – very low frequency band 0.003 – 0.04 Hz
    LF – low frequency band 0.04 – 0.15 Hz
    HF – high frequency band 0.15 – 0.4 Hz
    TP – Total Power
    """
    
    ulf = 0
    vlf = 0
    lf = 0
    hf = 0
    tp = 0
    try:
        sqr_mag = np.power(mag, 2)
        tp = np.sum(sqr_mag)

        lo: Final[List] = [0, 0.0033, 0.04, 0.15]
        hi: Final[List] = [0.0033, 0.04, 0.15, 0.4]
        
        ind = np.flatnonzero(freq < hi[0])
        if len(ind) > 0:
            ulf = np.sum(sqr_mag[ind])
        else:
            ulf = 0
            
        ind = np.flatnonzero(np.logical_and(freq >= lo[1], freq < hi[1]))
        if len(ind) > 0:
            vlf = np.sum(sqr_mag[ind])
        else:
            vlf = 0
        
        ind = np.flatnonzero(np.logical_and(freq >= lo[2], freq < hi[2]))
        if len(ind) > 0:
            lf = np.sum(sqr_mag[ind])
        else:
            lf = 0
        
        ind = np.flatnonzero(np.logical_and(freq >= lo[3], freq < hi[3]))
        if len(ind) > 0:
            hf = np.sum(sqr_mag[ind])
        else:
            hf = 0
            
    except (Exception,) as error:
        st.write_error_log(error)

    return ulf, vlf, lf, hf, tp


def __spread(
        y,
        yy,
        n,
        x,
        m
):
    """
    Given an array yy(0:n-1), extrapolate (spread) a value y into
    m actual array elements that best approximate the "fictional"
    (i.e., possible non-integer) array element number x. The weights
    used are coefficients of the Lagrange interpolating polynomial
    """
    
    try:
        nfac = [0, 1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
        if m > 10.:
            st.write_error_log('factorial table too small in spread')
            return
    
        ix = np.int_(x)
        if x == float(ix):
            yy[ix] = yy[ix] + y
        else:
            ilo = np.int_(x - 0.5 * m + 1.0)
            ilo = min(max(ilo, 1), n - m + 1)
            ihi = ilo + m - 1
            nden = nfac[m]
            fac = x - ilo
            for j in range(ilo + 1, ihi + 1):
                fac = fac * (x - j)
    
            yy[ihi] = yy[ihi] + y * fac / (nden * (x - ihi))
            for j in range(ihi - 1, ilo - 1, -1):
                nden = (nden / (j + 1 - ilo)) * (j - ihi)
                yy[j] = yy[j] + y * fac / (nden * (x - j))
                
    except (Exception,) as error:
        st.write_error_log(error)


def __spread_custom_1(
        y,
        yy,
        n,
        x,
) -> None:
    try:
        ix = np.int_(x)
        if x == float(ix):
            yy[ix] = yy[ix] + y
        else:
            ilo = min(max(np.int_(x - 1), 1), n - 3)
            ihi = ilo + 3
            
            nden = 6
            y_prod = y * np.prod(x - np.arange(ilo, ihi + 1))
            yy[ihi] = yy[ihi] + y_prod / (nden * (x - ihi))
            for j in range(ihi - 1, ilo - 1, -1):
                nden = (nden / (j + 1 - ilo)) * (j - ihi)
                yy[j] = yy[j] + y_prod / (nden * (x - j))
            pass
    
    except (Exception,) as error:
        st.write_error_log(error)
    

def __spread_custom_2(
        yy,
        n,
        x
) -> Any:
    try:
        ix = np.int_(x)
        if x == float(ix):
            yy[ix] = yy[ix] + 1.0
        else:
            ilo = min(max(np.int_(x - 1), 1), n - 3)
            ihi = ilo + 3
            nden = 6
    
            fac = np.prod(x - np.arange(ilo, ihi + 1))
            yy[ihi] = yy[ihi] + fac / (nden * (x - ihi))
            for j in range(ihi - 1, ilo - 1, -1):
                nden = (nden / (j + 1 - ilo)) * (j - ihi)
                yy[j] = yy[j] + fac / (nden * (x - j))
                
    except (Exception,) as error:
        st.write_error_log(error)


def _get_peak_false_alarm_probabilities(
        pmax:   NDArray,
        nout:   int,
        ofac:   float
) -> NDArray:
    """
    returns the peak false alarm probabilities
    Hence the lower is the probability and the more significant is the peak
    """
    
    prob = np.array([])
    try:
        expy = np.exp(-pmax)
        effm = 2.0 * (nout / ofac)
        prob = effm * expy
        if prob > 0.01:
            prob = 1.0 - (1.0 - expy) ** effm
            
    except (Exception,) as error:
        st.write_error_log(error)
    
    return prob


def _get_extirpolate(
        y:      NDArray,
        x:      NDArray,
        n_freq: int,
        n_out:  int,
        ofac:   int,
) -> Any:
    
    rwk1 = np.array([])
    iwk1 = np.array([])
    rwk2 = np.array([])
    iwk2 = np.array([])
    hypo2 = np.array([])
    
    try:
        xmin = x.min()
        xmax = x.max()
        
        avg = y.mean()
        
        xdiff = xmax - xmin
        
        ndim = 2 * n_freq
        
        ck = (((x - xmin) * (ndim / (xdiff * ofac))) % ndim)
        wk1 = np.zeros(ndim)
        wk2 = np.zeros(ndim)
        for _y, _ck in zip(y, ck):
            __spread_custom_1(_y - avg, wk1, ndim, _ck)
            __spread_custom_2(wk2, ndim, (2.0 * _ck) % ndim)
    
        wk1 = ifft(wk1)[1: n_out + 1] * len(wk1)
        rwk1, iwk1 = wk1.real, wk1.imag
        del wk1
        
        wk2 = ifft(wk2)[1: n_out + 1] * len(wk2)
        rwk2, iwk2 = wk2.real, wk2.imag
        hypo2 = 2.0 * np.abs(wk2)
        del wk2
        
    except (Exception,) as error:
        st.write_error_log(error)
    
    return rwk1, iwk1, rwk2, iwk2, hypo2


def _get_partition_extrapolation(
        y:      NDArray,
        x:      NDArray,
        n_freq: int,
        n_out:  int,
        ofac:   int,
        y_mean: float,
        xdiff:  float,
        xmin:   float
) -> Any:
    
    rwk1 = np.array([], dtype=np.float16)
    iwk1 = np.array([], dtype=np.float16)
    rwk2 = np.array([], dtype=np.float16)
    iwk2 = np.array([], dtype=np.float16)
    hypo2 = np.array([], dtype=np.float16)
    
    try:
        ndim = 2 * n_freq
        
        ck = ((x - xmin) * (ndim / (xdiff * ofac))) % ndim
        
        wk1 = np.zeros(ndim, dtype=np.float16)
        wk2 = np.zeros(ndim, dtype=np.float16)
        for _y, _ck in zip(y, ck):
            __spread_custom_1(_y - y_mean, wk1, ndim, _ck)
            __spread_custom_2(wk2, ndim, (2.0 * _ck) % ndim)
        
        wk1 = ifft(wk1)[1: n_out + 1] * len(wk1)
        rwk1, iwk1 = wk1.real, wk1.imag
        del wk1
        
        wk2 = ifft(wk2)[1: n_out + 1] * len(wk2)
        rwk2, iwk2 = wk2.real, wk2.imag
        hypo2 = 2.0 * np.abs(wk2)
        del wk2
    
    except (Exception,) as error:
        st.write_error_log(error)
    
    return rwk1, iwk1, rwk2, iwk2, hypo2


def __fasper(
        x:          NDArray,
        y:          NDArray,
        ofac:       float | int,
        hifac:      float | int,
        macc:       int = 4,
) -> Any:
    """ function fasper
      Given abscissas x (which need not be equally spaced) and ordinates
      y, and given a desired oversampling factor ofac (a typical value
      being 4 or larger). this routine creates an array wk1 with a
      sequence of nout increasing frequencies (not angular frequencies)
      up to hifac times the "average" Nyquist frequency, and creates
      an array wk2 with the values of the Lomb normalized periodogram at
      those frequencies. The arrays x and y are not altered. This
      routine also returns jmax such that wk2(jmax) is the maximum
      element in wk2, and prob, an estimate of the significance of that
      maximum against the hypothesis of random noise. A small value of prob
      indicates that a significant periodic signal is present.

    Reference:
      Press, W. H. & Rybicki, G. B. 1989
      ApJ vol. 338, p. 277-280.
      Fast algorithm for spectral analysis of unevenly sampled data
      (1989ApJ...338..277P)

    Arguments:
        x   : Abscissas array, (e.g. an array of times).
        y   : Ordinates array, (e.g. corresponding counts).
        ofac : Oversampling factor.
        hifac : Hifac * "average" Nyquist frequency = the highest frequency
             for which values of the Lomb normalized periodogram will
             be calculated.
        macc : Number of interpolation points per 1/4 cycle
            of the highest frequency

     Returns:
        Wk1 : An array of Lomb periodogram frequencies.
        Wk2 : An array of corresponding values of the Lomb periodogram.
        Nout : Wk1 & Wk2 dimensions (number of calculated frequencies)
        Jmax : The array index corresponding to the MAX( Wk2 ).
        Prob : False Alarm Probability of the largest Periodogram value
        Pwr

    History:
      02/23/2009, v1.0, MF
        Translation of IDL code (orig. Numerical recipies)
    """
    
    wk1 = np.array([])
    wk2 = np.array([])
    n_out = 0
    jmax = 0
    prob = 0
    var = 0.0
    
    try:
        # Check dimensions of input arrays
        n = len(x)
        if n != len(y):
            st.get_error_exception('Incompatible arrays.')
    
        n_out = 0.5 * ofac * hifac * n
        if round(n_out) != n_out:
            st.write_error_log('Warning: nout is not an integer and will be rounded down.')
            
        n_out = int(n_out)
        n_freq = 2 ** int(np.ceil(np.log2(np.int_(ofac * hifac * n * macc))))
    
        # sample variance because the divisor is N-1
        var = ((y - y.mean()) ** 2).sum() / (len(y) - 1)
    
        xdiff = x.max() - x.min()
        rwk1, iwk1, rwk2, iwk2, hypo2 = _get_extirpolate(y, x, n_freq, n_out, ofac)
        # endregion Partition Extrapolation

        # Compute the Lomb value for each frequency
        hc2wt = rwk2 / hypo2
        hs2wt = iwk2 / hypo2
    
        cwt = np.sqrt(0.5 + hc2wt)
        swt = np.sign(hs2wt) * (np.sqrt(0.5 - hc2wt))
        den = 0.5 * n + hc2wt * rwk2 + hs2wt * iwk2
        cterm = ((cwt * rwk1 + swt * iwk1) ** 2.0) / den
        sterm = ((cwt * iwk1 - swt * rwk1) ** 2.0) / (n - den)
    
        wk1 = (1.0 / (xdiff * ofac)) * (np.arange(n_out, dtype=np.float32) + 1.0)
        wk2 = (cterm + sterm) / (2.0 * var)
        jmax = wk2.argmax()
    
        # Estimate significance of largest peak value
        prob = _get_peak_false_alarm_probabilities(wk2.max(), n_out, ofac)
        
        wk1 = np.nan_to_num(wk1)
        wk2 = np.nan_to_num(wk2)
    
    except (Exception,) as error:
        st.write_error_log(error)

    return wk1, wk2, n_out, jmax, prob, var


def hrv_freq_domain(
        nn_time:        NDArray,
        nn_intervals:   NDArray
) -> Any:
    """
    HRV Frequency Domain Measures

    ULF – ultra low frequency band <0.003 Hz
    VLF – very low frequency band 0.003 – 0.04 Hz
    LF – low frequency band 0.04 – 0.15 Hz
    HF – high frequency band 0.15 – 0.4 Hz
    TP – Total Power
    """
    
    ulf = 0
    vlf = 0
    lf = 0
    hf = 0
    tp = 0
    ratio_lf_hf = 0
    
    try:
        wk1, wk2, nout, _, _, pwr = __fasper(nn_time, nn_intervals, 4.0, 2.0)
        maxout = int(nout // 2)
    
        frequencies = np.round(wk1[:maxout], 9)
        amplitudes = np.sqrt(np.round(wk2[:maxout], 9) / (nout / (2.0 * pwr)))
    
        ulf, vlf, lf, hf, tp = __power(frequencies, amplitudes)
        if hf > 0:
            ratio_lf_hf = lf / hf
        else:
            ratio_lf_hf = 0
    
        ulf         = np.round(ulf, 2)
        vlf         = np.round(vlf, 2)
        lf          = np.round(lf, 2)
        hf          = np.round(hf, 2)
        tp          = np.round(tp, 2)
        ratio_lf_hf = np.round(ratio_lf_hf, 2)
        
    except (Exception,) as error:
        st.write_error_log(error)

    return ulf, vlf, lf, hf, tp, ratio_lf_hf


def hrv_time_domain(
        hrv:            HRVariability,
        rr_time:        NDArray,
        rr_intervals:   NDArray,
        symbols:        NDArray
) -> [HRVariability, NDArray, NDArray]:

    """
    HRV Time Domain Measures
    sdnn - Standard deviation of NN intervals,
    sdann - Standard deviation of the average NN intervals calculated over short periods e.g. 5 minutes,
    rmssd - Square root of the mean of the squares of differences between adjacent NN intervals,
    pnn50 -  Percentage of differences between adjacent NN intervals that are greater than 50 ms,
    sdnnindx - Mean of the standard deviations of NN intervals calculated over short periods e.g. 5 minutes,
    msd -
    nn_time,
    nn_intervals
    """

    """ for Frequency Domain Measures """
    nn_intervals = list()
    nn_time = list()
    try:

        if len(symbols) > len(rr_intervals):
            symbols = symbols[len(symbols) - len(rr_intervals):]
    
        if len(rr_time) > len(rr_intervals):
            rr_time = rr_time[len(rr_time) - len(rr_intervals):]
        
        sum_rr      = 0.0
        sum_rr2     = 0.0
        
        hrv.rmssd   = 0.0
        hrv.msd     = 0.0
        hrv.totnn   = 0
        hrv.totrr   = 1
        
        nrr         = 1
        nnx         = 0
        nnn         = 0
        totnnn      = 0
    
        t = float(rr_time[0])
        end = t + LENGTH
        i = 0
        
        segments = np.ceil(rr_time[-1] / LENGTH).astype(int)
        if rr_time[-1] % LENGTH == 0:
            segments += 1
            
        ratbuf   = np.zeros(segments, dtype=np.float64)
        avbuf    = np.zeros(segments, dtype=np.float64)
        sdbuf    = np.zeros(segments, dtype=np.float64)
    
        try:
            rr_time = rr_time.astype(float)
            rr_intervals = np.round(rr_intervals, 3)
    
            for j, (t, rr, sym) in enumerate(zip(rr_time, rr_intervals, symbols)):
                if j == 0:
                    continue
    
                while t > (end + LENGTH):
                    i += 1
                    end += LENGTH
    
                if t >= end:
                    if nnn > 1:
                        ratbuf[i] = nnn / nrr
                        sdbuf[i] = np.sqrt(((sdbuf[i] - avbuf[i] * avbuf[i] / nnn) / (nnn - 1)))
                        avbuf[i] /= nnn
                    i += 1
                    nnn = nrr = 0
                    end += LENGTH
    
                nrr += 1
                hrv.totrr += 1
                if sym == df.HolterBeatTypes.N.value and symbols[j - 1] == df.HolterBeatTypes.N.value:
                    """ for Frequency Domain Measures """
                    nn_intervals.append(rr)
                    nn_time.append(t)
    
                    nnn += 1
                    hrv.totnn += 1
    
                    avbuf[i] += rr
                    sdbuf[i] += (rr * rr)
                    sum_rr += rr
                    sum_rr2 += (rr * rr)
                    if j >= 2 and symbols[j - 2] == df.HolterBeatTypes.N.value:
                        totnnn += 1
                        hrv.rmssd += (rr - rr_intervals[j - 1]) * (rr - rr_intervals[j - 1])
                        hrv.msd += abs(rr - rr_intervals[j - 1])
                        if abs(rr - rr_intervals[j - 1]) - NNDIF > ROFFERR:
                            nnx += 1
    
        except (Exception,) as err:
            st.get_error_exception(err)
    
        if hrv.totnn <= 1:
            return hrv, [0], [0]
    
        hrv.sdnn = round(np.sqrt((sum_rr2 - sum_rr * sum_rr / hrv.totnn) / (hrv.totnn - 1)), 2)
        if totnnn <= 0:
            return hrv, [0], [0]
    
        hrv.rmssd   = round(np.sqrt(hrv.rmssd / totnnn), 2)
        hrv.msd     = round(hrv.msd / totnnn, 2)
        hrv.pnn50   = round((nnx / totnnn) * 100 , 2) # %
        if nnn > 1:
            ratbuf[i] = nnn / nrr
            sdbuf[i] = np.sqrt((sdbuf[i] - avbuf[i] * avbuf[i] / nnn) / (nnn - 1))
            avbuf[i] /= nnn
    
        sum_rr = 0.0
        sum_rr2 = 0.0
        sum_rr_sd = 0.0

        ind_valid = np.flatnonzero(ratbuf != 0)
        h = len(ind_valid)
        if len(ind_valid) > 0:
            sum_rr_sd   = np.sum(np.nan_to_num(sdbuf[ind_valid]))
            sum_rr      = np.sum(np.nan_to_num(avbuf[ind_valid]))
            sum_rr2     = np.sum(np.nan_to_num(avbuf[ind_valid] * avbuf[ind_valid]))
        
        if h > 1:
            hrv.sdann = round(np.sqrt((sum_rr2 - sum_rr * sum_rr / h) / (h - 1)), 2)
        else:
            hrv.sdann = 0
    
        if h > 0:
            hrv.sdnnindx = round(sum_rr_sd / h, 2)
        else:
            hrv.sdnnindx = 0
    
        nn_time         = np.asarray(nn_time, dtype=float)
        nn_intervals    = np.asarray(nn_intervals, dtype=float)
        hrv.msd         = round(hrv.msd, 2)
    
        hrv.beats_use_hrv   = len(nn_intervals) / len(rr_intervals)
        hrv.mean_rr         = round(np.mean(nn_intervals), 1)
        hrv.max_rr          = np.max(nn_intervals)
        hrv.min_rr          = np.min(nn_intervals)
        
    except (Exception,) as error:
        st.write_error_log(error)
        
    return hrv, nn_time, nn_intervals


def calculate_hrv(
        x:              NDArray,
        beat_types:     NDArray,
        sampling_rate:  int,
        is_epoch_time:  bool = False,
        flag_hrv:       str = "m"
) -> HRVariability:

    hrv = HRVariability()
    try:
        if len(x) <= df.LIMIT_BEAT_CALCULATE_HR:
            return hrv

        beat_types = beat_types[1:]
        if is_epoch_time:
            rr_intervals = np.diff(x) / df.MILLISECOND
        else:
            rr_intervals = np.diff(x) / sampling_rate

        time_m = np.hstack((0, rr_intervals)).cumsum()[1:].astype(int)
        rr_intervals = np.asarray(np.multiply(np.round(rr_intervals, 9), 1000), dtype=float)
        if len(beat_types) <= df.LIMIT_BEAT_CALCULATE_HR:
            return hrv

        hrv, nn_time, nn_intervals = hrv_time_domain(hrv, time_m, rr_intervals, beat_types)
        if flag_hrv not in ['m', 'h']:
            if hrv.totrr != 0 and hrv.totnn != 0:
                if len(nn_intervals) > df.LIMIT_BEAT_TO_CALCULATE_HRV:
                    nn_time = nn_time[:df.LIMIT_BEAT_TO_CALCULATE_HRV]
                    nn_intervals = nn_intervals[:df.LIMIT_BEAT_TO_CALCULATE_HRV]
                    
                hrv.ulf, hrv.vlf, hrv.lf, hrv.hf, hrv.tp, hrv.ratio_lf_hf = hrv_freq_domain(nn_time, nn_intervals)
                hrv.lf_norm = hrv.lf / (hrv.tp - hrv.vlf) * 100
                hrv.hf_norm = hrv.hf / (hrv.tp - hrv.vlf) * 100
                hrv.cv = hrv.sdnn / hrv.mean_rr
            
            else:
                hrv.beats_use_hrv = 0
                hrv.mean_rr = 0
                hrv.max_rr = 0
                hrv.min_rr = 0

    except (Exception,) as error:
        st.write_error_log(error)

    return hrv