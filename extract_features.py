"""
Extract features từ raw EMG dataset gốc
Input: EMG time-series từ dataset/fatigue/ và dataset/non fatigue/
Output: CSV file với extracted features
"""

import numpy as np
import pandas as pd
import os
from scipy import signal
from scipy.stats import kurtosis, skew
import glob

def extract_emg_features(emg_signal, sampling_rate=1000):
    """
    Extract features từ EMG time-series signal

    Parameters:
    -----------
    emg_signal : array-like
        Raw EMG signal (amplitudo values)
    sampling_rate : int
        Sampling rate (Hz), default 1000Hz

    Returns:
    --------
    dict : Dictionary chứa các features
    """
    # Ensure numpy array
    emg_signal = np.array(emg_signal)

    # Remove mean (DC offset)
    emg_signal = emg_signal - np.mean(emg_signal)

    features = {}

    # ============ TIME-DOMAIN FEATURES ============

    # 1. Root Mean Square (RMS)
    features['emg_rms'] = np.sqrt(np.mean(emg_signal ** 2))

    # 2. Mean Absolute Value (MAV)
    features['emg_mav'] = np.mean(np.abs(emg_signal))

    # 3. Variance
    features['emg_variance'] = np.var(emg_signal)

    # 4. Standard Deviation
    features['emg_std'] = np.std(emg_signal)

    # 5. Waveform Length (WL)
    features['emg_waveform_length'] = np.sum(np.abs(np.diff(emg_signal)))

    # 6. Zero Crossing (ZC) - số lần tín hiệu cross zero
    zero_crossings = np.where(np.diff(np.sign(emg_signal)))[0]
    features['emg_zero_crossing'] = len(zero_crossings)

    # 7. Slope Sign Changes (SSC)
    diff_signal = np.diff(emg_signal)
    ssc = np.sum(np.diff(np.sign(diff_signal)) != 0)
    features['emg_ssc'] = ssc

    # 8. Kurtosis (measure of "tailedness")
    features['emg_kurtosis'] = kurtosis(emg_signal)

    # 9. Skewness (measure of asymmetry)
    features['emg_skewness'] = skew(emg_signal)

    # 10. Peak Value
    features['emg_peak'] = np.max(np.abs(emg_signal))

    # ============ FREQUENCY-DOMAIN FEATURES ============

    # Compute Power Spectral Density using Welch's method
    freqs, psd = signal.welch(emg_signal, fs=sampling_rate, nperseg=min(256, len(emg_signal)))

    # 11. Median Frequency (MDF)
    cumsum_psd = np.cumsum(psd)
    total_power = cumsum_psd[-1]
    median_idx = np.where(cumsum_psd >= total_power / 2)[0]
    if len(median_idx) > 0:
        features['emg_median_freq'] = freqs[median_idx[0]]
    else:
        features['emg_median_freq'] = 0

    # 12. Mean Frequency (MNF)
    features['emg_mean_freq'] = np.sum(freqs * psd) / np.sum(psd)

    # 13. Peak Frequency
    peak_freq_idx = np.argmax(psd)
    features['emg_peak_freq'] = freqs[peak_freq_idx]

    # 14. Total Power
    features['emg_total_power'] = np.sum(psd)

    # 15. Power in specific bands
    # Low freq band (0-50Hz)
    low_band_mask = (freqs >= 0) & (freqs < 50)
    features['emg_power_low'] = np.sum(psd[low_band_mask])

    # Mid freq band (50-150Hz)
    mid_band_mask = (freqs >= 50) & (freqs < 150)
    features['emg_power_mid'] = np.sum(psd[mid_band_mask])

    # High freq band (150-500Hz)
    high_band_mask = (freqs >= 150) & (freqs < 500)
    features['emg_power_high'] = np.sum(psd[high_band_mask])

    return features

def process_emg_file(file_path, label, label_name):
    """
    Xử lý một file EMG và extract features

    Parameters:
    -----------
    file_path : str
        Path đến EMG CSV file
    label : int
        Label (0=Non-Fatigue, 1=Fatigue)
    label_name : str
        Tên label

    Returns:
    --------
    dict : Features dictionary với label
    """
    try:
        # Đọc CSV
        df = pd.read_csv(file_path)

        # Tìm cột chứa EMG data (có thể là 'amplitudo', hoặc cột cuối)
        if 'amplitudo' in df.columns:
            emg_data = df['amplitudo'].values
        else:
            # Lấy cột cuối cùng (thường là EMG data)
            emg_data = df.iloc[:, -1].values

        # Remove NaN
        emg_data = emg_data[~np.isnan(emg_data)]

        # Extract features
        features = extract_emg_features(emg_data)

        # Thêm label
        features['label'] = label
        features['class_name'] = label_name
        features['file_name'] = os.path.basename(file_path)

        return features

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

def extract_all_features(dataset_dir='dataset', output_dir='data_extracted'):
    """
    Extract features từ tất cả EMG files

    Parameters:
    -----------
    dataset_dir : str
        Thư mục chứa dataset
    output_dir : str
        Thư mục lưu features extracted
    """
    print("="*70)
    print(" "*15 + "EXTRACT FEATURES TỪ DATASET GỐC")
    print("="*70)

    # Paths
    fatigue_dir = os.path.join(dataset_dir, 'fatigue')
    non_fatigue_dir = os.path.join(dataset_dir, 'non fatigue')

    # Tạo output dir
    os.makedirs(output_dir, exist_ok=True)

    all_features = []

    # ========== XỬ LÝ FATIGUE FILES ==========
    print(f"\n📁 Đang xử lý Fatigue files từ {fatigue_dir}...")
    fatigue_files = glob.glob(os.path.join(fatigue_dir, '*.csv'))
    print(f"   Tìm thấy {len(fatigue_files)} files")

    for i, file_path in enumerate(fatigue_files, 1):
        print(f"   [{i}/{len(fatigue_files)}] Processing: {os.path.basename(file_path)}", end='')
        features = process_emg_file(file_path, label=1, label_name='Fatigue')
        if features:
            all_features.append(features)
            print(" ✓")
        else:
            print(" ✗")

    # ========== XỬ LÝ NON-FATIGUE FILES ==========
    print(f"\n📁 Đang xử lý Non-Fatigue files từ {non_fatigue_dir}...")
    non_fatigue_files = glob.glob(os.path.join(non_fatigue_dir, '*.csv'))
    print(f"   Tìm thấy {len(non_fatigue_files)} files")

    for i, file_path in enumerate(non_fatigue_files, 1):
        print(f"   [{i}/{len(non_fatigue_files)}] Processing: {os.path.basename(file_path)}", end='')
        features = process_emg_file(file_path, label=0, label_name='Non-Fatigue')
        if features:
            all_features.append(features)
            print(" ✓")
        else:
            print(" ✗")

    # ========== TẠO DATAFRAME ==========
    print(f"\n📊 Tạo DataFrame...")
    df = pd.DataFrame(all_features)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Lưu file
    output_path = os.path.join(output_dir, 'extracted_features.csv')
    df.to_csv(output_path, index=False)

    print(f"\n{'='*70}")
    print("THỐNG KÊ")
    print('='*70)
    print(f"\nTổng số files: {len(all_features)}")
    print(f"  - Fatigue: {(df['label']==1).sum()} files")
    print(f"  - Non-Fatigue: {(df['label']==0).sum()} files")

    print(f"\nSố features: {len(df.columns) - 3}")  # Trừ label, class_name, file_name

    # Feature columns
    feature_cols = [col for col in df.columns if col not in ['label', 'class_name', 'file_name']]
    print(f"\nFeatures extracted:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

    print(f"\n✓ Đã lưu features tại: {output_path}")

    # Thống kê chi tiết
    print(f"\n{'='*70}")
    print("THỐNG KÊ FEATURES THEO CLASS")
    print('='*70)

    stats_by_class = df.groupby('class_name')[feature_cols].agg(['mean', 'std'])
    print("\nMean values:")
    print(df.groupby('class_name')[feature_cols[:5]].mean().round(2))

    # Phân bố
    print(f"\nPhân bố:")
    print(df['class_name'].value_counts())

    return df

def create_train_test_split(input_csv='data_extracted/extracted_features.csv',
                            output_dir='data_extracted',
                            test_size=0.25,
                            random_state=42):
    """
    Chia train/test từ extracted features
    """
    from sklearn.model_selection import train_test_split

    print(f"\n{'='*70}")
    print("CHIA TRAIN/TEST SET")
    print('='*70)

    # Load data
    df = pd.read_csv(input_csv)

    # Tách features và labels
    feature_cols = [col for col in df.columns if col not in ['label', 'class_name', 'file_name']]
    X = df[feature_cols]
    y = df['label']

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Tạo DataFrames
    train_df = X_train.copy()
    train_df['label'] = y_train
    train_df['class_name'] = train_df['label'].map({0: 'Non-Fatigue', 1: 'Fatigue'})

    test_df = X_test.copy()
    test_df['label'] = y_test
    test_df['class_name'] = test_df['label'].map({0: 'Non-Fatigue', 1: 'Fatigue'})

    # Lưu files
    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n✓ Train set: {len(train_df)} samples → {train_path}")
    print(f"  - Non-Fatigue: {(train_df['label']==0).sum()}")
    print(f"  - Fatigue: {(train_df['label']==1).sum()}")

    print(f"\n✓ Test set: {len(test_df)} samples → {test_path}")
    print(f"  - Non-Fatigue: {(test_df['label']==0).sum()}")
    print(f"  - Fatigue: {(test_df['label']==1).sum()}")

    return train_df, test_df

if __name__ == "__main__":
    # Extract features từ dataset gốc
    df = extract_all_features(
        dataset_dir='dataset',
        output_dir='data_extracted'
    )

    # Chia train/test
    train_df, test_df = create_train_test_split(
        input_csv='data_extracted/extracted_features.csv',
        output_dir='data_extracted',
        test_size=0.25
    )

    print(f"\n{'='*70}")
    print("✓ HOÀN TẤT!")
    print('='*70)
    print("\nBước tiếp theo:")
    print("  python train_models.py  # Train với data thực")
