"""
Script để tạo dữ liệu synthetic cho bài toán nhận dạng mỏi cơ
Tạo features dựa trên các chỉ số sinh lý điển hình
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def generate_fatigue_muscle_data(n_samples=1000, seed=42):
    """
    Tạo dữ liệu synthetic cho nhận dạng mỏi cơ

    Features:
    - emg_rms: Root Mean Square của tín hiệu EMG (mV)
    - emg_mav: Mean Absolute Value của tín hiệu EMG (mV)
    - emg_median_freq: Tần số trung vị của tín hiệu EMG (Hz)
    - emg_mean_freq: Tần số trung bình của tín hiệu EMG (Hz)
    - muscle_force: Lực cơ (N)
    - heart_rate: Nhịp tim (bpm)
    - work_duration: Thời gian làm việc (phút)
    - rest_time: Thời gian nghỉ ngơi (phút)
    - movement_frequency: Tần số chuyển động (lần/phút)
    - muscle_tension: Độ căng cơ (0-100)

    Labels:
    - 0: Không mỏi (Non-Fatigue)
    - 1: Mỏi (Fatigue)
    """
    np.random.seed(seed)

    # Tạo số lượng mẫu cho mỗi class
    n_non_fatigue = n_samples // 2
    n_fatigue = n_samples - n_non_fatigue

    # ============= KHÔNG MỎI (Non-Fatigue) =============
    # Tăng variance và tạo overlap giữa classes để realistic hơn
    non_fatigue_data = {
        # EMG features - giá trị thấp hơn khi không mỏi
        'emg_rms': np.random.normal(0.18, 0.05, n_non_fatigue),  # mV - tăng std
        'emg_mav': np.random.normal(0.15, 0.045, n_non_fatigue),  # mV - tăng std
        'emg_median_freq': np.random.normal(78, 12, n_non_fatigue),  # Hz - tăng std, giảm mean
        'emg_mean_freq': np.random.normal(82, 12, n_non_fatigue),  # Hz - tăng std

        # Muscle force - ổn định
        'muscle_force': np.random.normal(42, 8, n_non_fatigue),  # N - tăng variance

        # Heart rate - bình thường
        'heart_rate': np.random.normal(80, 12, n_non_fatigue),  # bpm - tăng std

        # Work duration - thời gian làm việc ngắn
        'work_duration': np.random.normal(25, 10, n_non_fatigue),  # phút - tăng mean và std

        # Rest time - nghỉ đủ
        'rest_time': np.random.normal(6, 3, n_non_fatigue),  # phút - tăng std

        # Movement frequency - bình thường
        'movement_frequency': np.random.normal(18, 5, n_non_fatigue),  # lần/phút - tăng std

        # Muscle tension - thấp
        'muscle_tension': np.random.normal(40, 12, n_non_fatigue),  # 0-100 - tăng std
    }

    # ============= MỎI (Fatigue) =============
    fatigue_data = {
        # EMG features - tăng khi mỏi
        'emg_rms': np.random.normal(0.24, 0.05, n_fatigue),  # mV - giảm mean, tăng std
        'emg_mav': np.random.normal(0.20, 0.045, n_fatigue),  # mV - giảm mean, tăng std
        'emg_median_freq': np.random.normal(68, 12, n_fatigue),  # Hz - tăng mean, tăng std
        'emg_mean_freq': np.random.normal(72, 12, n_fatigue),  # Hz - tăng mean, tăng std

        # Muscle force - giảm
        'muscle_force': np.random.normal(36, 8, n_fatigue),  # N - tăng mean, tăng std

        # Heart rate - tăng
        'heart_rate': np.random.normal(90, 12, n_fatigue),  # bpm - giảm mean, tăng std

        # Work duration - làm việc lâu
        'work_duration': np.random.normal(38, 10, n_fatigue),  # phút - giảm mean

        # Rest time - nghỉ ít
        'rest_time': np.random.normal(4, 3, n_fatigue),  # phút - tăng mean và std

        # Movement frequency - giảm
        'movement_frequency': np.random.normal(14, 5, n_fatigue),  # lần/phút - tăng mean và std

        # Muscle tension - cao
        'muscle_tension': np.random.normal(58, 12, n_fatigue),  # 0-100 - giảm mean, tăng std
    }

    # Tạo DataFrame
    df_non_fatigue = pd.DataFrame(non_fatigue_data)
    df_non_fatigue['label'] = 0
    df_non_fatigue['class_name'] = 'Non-Fatigue'

    df_fatigue = pd.DataFrame(fatigue_data)
    df_fatigue['label'] = 1
    df_fatigue['class_name'] = 'Fatigue'

    # Kết hợp dữ liệu
    df = pd.concat([df_non_fatigue, df_fatigue], ignore_index=True)

    # Shuffle dữ liệu
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Đảm bảo các giá trị nằm trong phạm vi hợp lý
    df['emg_rms'] = df['emg_rms'].clip(0.05, 0.5)
    df['emg_mav'] = df['emg_mav'].clip(0.04, 0.4)
    df['emg_median_freq'] = df['emg_median_freq'].clip(40, 120)
    df['emg_mean_freq'] = df['emg_mean_freq'].clip(45, 125)
    df['muscle_force'] = df['muscle_force'].clip(10, 80)
    df['heart_rate'] = df['heart_rate'].clip(50, 140)
    df['work_duration'] = df['work_duration'].clip(1, 90)
    df['rest_time'] = df['rest_time'].clip(0.5, 20)
    df['movement_frequency'] = df['movement_frequency'].clip(5, 40)
    df['muscle_tension'] = df['muscle_tension'].clip(10, 90)

    return df

def save_train_test_data(output_dir='data_generated', n_samples=2000, test_size=0.25, seed=42):
    """
    Tạo và lưu dữ liệu train/test
    """
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    # Tạo dữ liệu
    print(f"Đang tạo {n_samples} mẫu dữ liệu...")
    df = generate_fatigue_muscle_data(n_samples=n_samples, seed=seed)

    # Tách features và labels
    feature_columns = [col for col in df.columns if col not in ['label', 'class_name']]
    X = df[feature_columns]
    y = df['label']

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Thêm lại label vào dataframe
    train_df = X_train.copy()
    train_df['label'] = y_train
    train_df['class_name'] = train_df['label'].map({0: 'Non-Fatigue', 1: 'Fatigue'})

    test_df = X_test.copy()
    test_df['label'] = y_test
    test_df['class_name'] = test_df['label'].map({0: 'Non-Fatigue', 1: 'Fatigue'})

    # Lưu file
    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    full_path = os.path.join(output_dir, 'full_data.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    df.to_csv(full_path, index=False)

    # In thông tin
    print(f"\n✓ Đã tạo và lưu dữ liệu thành công!")
    print(f"  - Tổng số mẫu: {len(df)}")
    print(f"  - Train set: {len(train_df)} mẫu ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - Test set: {len(test_df)} mẫu ({len(test_df)/len(df)*100:.1f}%)")
    print(f"\nPhân bố train set:")
    print(f"  - Non-Fatigue: {(train_df['label']==0).sum()} mẫu")
    print(f"  - Fatigue: {(train_df['label']==1).sum()} mẫu")
    print(f"\nPhân bố test set:")
    print(f"  - Non-Fatigue: {(test_df['label']==0).sum()} mẫu")
    print(f"  - Fatigue: {(test_df['label']==1).sum()} mẫu")
    print(f"\nCác file đã được lưu tại:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"  - {full_path}")
    print(f"\nCác features: {', '.join(feature_columns)}")

    # In thống kê cơ bản
    print(f"\n{'='*60}")
    print("THỐNG KÊ DỮ LIỆU")
    print('='*60)
    print("\nThống kê theo class:")
    print(df.groupby('class_name')[feature_columns].mean().round(2))

    return train_df, test_df, df

if __name__ == "__main__":
    # Tạo dữ liệu
    train_df, test_df, full_df = save_train_test_data(
        output_dir='data_generated',
        n_samples=2000,  # Tạo 2000 mẫu
        test_size=0.25,  # 25% cho test
        seed=42
    )

    print("\n✓ Hoàn tất! Bạn có thể sử dụng dữ liệu để train model.")
