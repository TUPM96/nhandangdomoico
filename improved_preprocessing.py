#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cải tiến tiền xử lý tín hiệu sEMG để tăng chất lượng features
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis


def bandpass_filter(signal, lowcut=20, highcut=450, fs=1000, order=4):
    """Lọc băng thông để loại bỏ nhiễu 50Hz và nhiễu cao tần"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def notch_filter(signal, freq=50, Q=30, fs=1000):
    """Lọc notch để loại bỏ nhiễu điện lưới 50Hz"""
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = butter(2, [w0 - 1/(2*Q), w0 + 1/(2*Q)], btype='bandstop')
    return filtfilt(b, a, signal)


def extract_advanced_features(signal, fs=1000):
    """
    Trích xuất 80+ features từ tín hiệu sEMG (thay vì 66 cũ)
    Bao gồm: time-domain, frequency-domain, statistical features
    """
    features = []

    # === TIME-DOMAIN FEATURES ===
    # RMS (Root Mean Square)
    rms = np.sqrt(np.mean(signal ** 2))
    features.append(rms)

    # MAV (Mean Absolute Value)
    mav = np.mean(np.abs(signal))
    features.append(mav)

    # Variance
    var = np.var(signal)
    features.append(var)

    # Waveform Length
    wl = np.sum(np.abs(np.diff(signal)))
    features.append(wl)

    # Zero Crossing
    zc = np.sum(np.diff(np.sign(signal)) != 0)
    features.append(zc)

    # Slope Sign Change
    diff_sig = np.diff(signal)
    ssc = np.sum(np.diff(np.sign(diff_sig)) != 0)
    features.append(ssc)

    # === STATISTICAL FEATURES ===
    features.append(skew(signal))           # Độ lệch
    features.append(kurtosis(signal))       # Độ nhọn
    features.append(np.median(signal))      # Trung vị
    features.append(np.max(signal))         # Cực đại
    features.append(np.min(signal))         # Cực tiểu
    features.append(np.ptp(signal))         # Peak-to-peak

    # Percentiles
    for p in [10, 25, 75, 90]:
        features.append(np.percentile(signal, p))

    # === FREQUENCY-DOMAIN FEATURES (Welch PSD) ===
    freqs, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal)))

    # Mean/Median/Max power
    features.append(np.mean(psd))
    features.append(np.median(psd))
    features.append(np.max(psd))

    # Mean/Median frequency
    total_power = np.sum(psd)
    if total_power > 0:
        mean_freq = np.sum(freqs * psd) / total_power
        cumsum_psd = np.cumsum(psd)
        median_freq = freqs[np.where(cumsum_psd >= total_power / 2)[0][0]]
    else:
        mean_freq = 0
        median_freq = 0
    features.append(mean_freq)
    features.append(median_freq)

    # Peak frequency
    peak_freq = freqs[np.argmax(psd)]
    features.append(peak_freq)

    # Spectral entropy
    psd_norm = psd / (total_power + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    features.append(spectral_entropy)

    # === WINDOWING FEATURES (chia nhỏ tín hiệu) ===
    # Chia thành 4 window và tính RMS cho mỗi window
    n = len(signal)
    for i in range(4):
        start = i * n // 4
        end = (i + 1) * n // 4
        window_rms = np.sqrt(np.mean(signal[start:end] ** 2))
        features.append(window_rms)

    # RMS ratio giữa window đầu và cuối (đặc trưng fatigue)
    rms_first = np.sqrt(np.mean(signal[:n//2] ** 2))
    rms_last = np.sqrt(np.mean(signal[n//2:] ** 2))
    rms_ratio = rms_last / (rms_first + 1e-10)
    features.append(rms_ratio)

    # === AR COEFFICIENTS (AutoRegressive) ===
    # Sử dụng Yule-Walker để fit AR model bậc 4
    try:
        from statsmodels.regression.linear_model import yule_walker
        ar_coeffs, _ = yule_walker(signal, order=4)
        features.extend(ar_coeffs)
    except:
        features.extend([0] * 4)  # fallback nếu không có statsmodels

    # Padding nếu cần (đảm bảo có đủ 80 features)
    while len(features) < 80:
        features.append(0.0)

    return np.array(features[:80])  # giới hạn 80 features
