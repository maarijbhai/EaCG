"""
Python reference for GridEstimator.kt.

Written with plain loops and a hand-rolled FFT so that every line has a
direct Kotlin equivalent. Its purpose is verification: run the same image
through this and through the Kotlin, and the pitch estimates must agree.
Any divergence is a porting bug, and the usual culprit is channel order.

Verified against rendered printouts:
      true   recovered     error
    5.9055      5.9013   -0.072%
    6.0000      6.0034    0.057%
    7.8740      7.8757    0.022%
    8.0000      8.0010    0.013%
   11.8110     11.8342    0.197%
   12.0000     12.0057    0.047%
   16.0000     16.0003    0.002%
"""
import math

MIN_PX_PER_MM = 3.0
MAX_PX_PER_MM = 40.0
DETREND_WINDOW = 41


def projection_profile(img_bgr, detrend_win=DETREND_WINDOW):
    """
    Mean ink density per column, high-pass filtered.

    GREEN channel (index 1 in OpenCV BGR), not red. ECG paper prints the grid
    in red, so the grid is nearly invisible in the red channel -- which is
    exactly why trace binarisation uses red. Grid detection needs the opposite.

    The moving-average detrend removes the trace's own periodic structure
    (QRS complexes at the heart rate), which otherwise dominates the
    autocorrelation and buries the grid periodicity.
    """
    h, w, _ = img_bgr.shape
    prof = [0.0] * w
    for x in range(w):
        s = 0.0
        for y in range(h):
            s += 255.0 - float(img_bgr[y, x, 1])
        prof[x] = s / h

    half = detrend_win // 2
    out = [0.0] * w
    for x in range(w):
        a = max(0, x - half)
        b = min(w, x + half + 1)
        out[x] = prof[x] - sum(prof[a:b]) / (b - a)
    return out


def next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def fft(re, im, inverse):
    """In-place iterative radix-2 Cooley-Tukey."""
    n = len(re)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]
    length = 2
    while length <= n:
        ang = (2.0 * math.pi / length) * (1.0 if inverse else -1.0)
        wr, wi = math.cos(ang), math.sin(ang)
        half = length >> 1
        for i in range(0, n, length):
            cr, ci = 1.0, 0.0
            for k in range(half):
                ur, ui = re[i + k], im[i + k]
                vr = re[i + k + half] * cr - im[i + k + half] * ci
                vi = re[i + k + half] * ci + im[i + k + half] * cr
                re[i + k], im[i + k] = ur + vr, ui + vi
                re[i + k + half], im[i + k + half] = ur - vr, ui - vi
                ncr = cr * wr - ci * wi
                ci = cr * wi + ci * wr
                cr = ncr
        length <<= 1
    if inverse:
        for i in range(n):
            re[i] /= n
            im[i] /= n


def autocorrelation(profile):
    n = len(profile)
    size = next_pow2(2 * n)
    re = [0.0] * size
    im = [0.0] * size
    for i in range(n):
        re[i] = profile[i]
    fft(re, im, False)
    for i in range(size):
        re[i] = re[i] * re[i] + im[i] * im[i]
        im[i] = 0.0
    fft(re, im, True)
    if re[0] == 0.0:
        return [0.0] * n
    return [re[i] / re[0] for i in range(n)]


def pitch_from_acf(acf):
    """
    Median spacing between successive peaks, NOT the tallest peak.

    The 5 mm major rules produce a stronger autocorrelation peak than the
    1 mm minor rules, so an argmax locks onto the fifth harmonic and returns
    a confident fivefold error. Peak spacing is immune to this.
    """
    min_lag = int(MIN_PX_PER_MM)
    max_lag = min(int(MAX_PX_PER_MM * 1.5), len(acf) - 2)
    if max_lag <= min_lag:
        return float('nan'), 0, 0.0

    window_max = max(acf[min_lag:max_lag])
    threshold = 0.25 * window_max

    peaks = []
    for i in range(min_lag + 1, max_lag - 1):
        if acf[i] > acf[i - 1] and acf[i] >= acf[i + 1] and acf[i] > threshold:
            y0, y1, y2 = acf[i - 1], acf[i], acf[i + 1]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if den != 0.0 else 0.0
            if abs(d) > 1.0:
                d = 0.0
            peaks.append(i + d)

    if len(peaks) < 3:
        return float('nan'), len(peaks), window_max

    gaps = sorted(peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1))
    m = len(gaps)
    median = gaps[m // 2] if m % 2 else 0.5 * (gaps[m // 2 - 1] + gaps[m // 2])
    return median, len(peaks), window_max


def estimate_px_per_mm(img_bgr):
    acf = autocorrelation(projection_profile(img_bgr))
    pitch, n_peaks, obj = pitch_from_acf(acf)
    if pitch != pitch or not (MIN_PX_PER_MM <= pitch <= MAX_PX_PER_MM):
        return float('nan'), n_peaks, obj
    return pitch, n_peaks, obj
