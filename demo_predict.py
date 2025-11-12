"""
Script demo sử dụng model đã train để predict
"""

import numpy as np
import joblib
import pandas as pd
from train_models import FatigueMuscleClassifier

def predict_fatigue(model_path, sample_data, verbose=True):
    """
    Dự đoán mỏi cơ cho một mẫu

    Parameters:
    -----------
    model_path : str
        Path đến model file
    sample_data : dict hoặc array
        Dữ liệu mẫu với 10 features
    verbose : bool
        In chi tiết kết quả
    """
    # Load model
    classifier = FatigueMuscleClassifier.load_model(model_path)

    # Chuẩn bị dữ liệu
    if isinstance(sample_data, dict):
        feature_names = [
            'emg_rms', 'emg_mav', 'emg_median_freq', 'emg_mean_freq',
            'muscle_force', 'heart_rate', 'work_duration', 'rest_time',
            'movement_frequency', 'muscle_tension'
        ]
        sample_array = np.array([[sample_data[f] for f in feature_names]])
    else:
        sample_array = np.array(sample_data).reshape(1, -1)

    # Chuẩn hóa
    sample_scaled = classifier.scaler.transform(sample_array)

    # Predict
    prediction = classifier.model.predict(sample_scaled)[0]

    # Probability (nếu có)
    if hasattr(classifier.model, 'predict_proba'):
        proba = classifier.model.predict_proba(sample_scaled)[0]
        prob_non_fatigue = proba[0]
        prob_fatigue = proba[1]
    else:
        prob_non_fatigue = None
        prob_fatigue = None

    result = {
        'prediction': int(prediction),
        'class_name': 'Fatigue' if prediction == 1 else 'Non-Fatigue',
        'prob_non_fatigue': float(prob_non_fatigue) if prob_non_fatigue is not None else None,
        'prob_fatigue': float(prob_fatigue) if prob_fatigue is not None else None
    }

    if verbose:
        print(f"\n{'='*60}")
        print("KẾT QUẢ DỰ ĐOÁN MỎI CƠ")
        print('='*60)
        print(f"\nDữ liệu đầu vào:")
        if isinstance(sample_data, dict):
            for key, value in sample_data.items():
                print(f"  {key:25s}: {value:.2f}")
        else:
            print(f"  {sample_array[0]}")

        print(f"\n{'='*60}")
        print(f"Kết quả: {result['class_name']}")
        if result['prob_non_fatigue'] is not None:
            print(f"\nXác suất:")
            print(f"  Non-Fatigue: {result['prob_non_fatigue']*100:.2f}%")
            print(f"  Fatigue:     {result['prob_fatigue']*100:.2f}%")
        print('='*60)

    return result

def demo_examples():
    """
    Demo với một vài ví dụ
    """
    print("="*70)
    print(" "*15 + "DEMO NHẬN DẠNG MỎI CƠ")
    print("="*70)

    # Ví dụ 1: Người không mỏi
    print("\n\n" + "="*70)
    print("VÍ DỤ 1: NGƯỜI KHÔNG MỎI (Fresh)")
    print("="*70)

    non_fatigue_sample = {
        'emg_rms': 0.15,           # EMG thấp
        'emg_mav': 0.12,           # EMG thấp
        'emg_median_freq': 85,     # Tần số cao
        'emg_mean_freq': 90,       # Tần số cao
        'muscle_force': 45,        # Lực cơ ổn định
        'heart_rate': 75,          # Nhịp tim bình thường
        'work_duration': 15,       # Làm việc ít
        'rest_time': 8,            # Nghỉ đủ
        'movement_frequency': 20,  # Chuyển động bình thường
        'muscle_tension': 35       # Căng cơ thấp
    }

    # Test với cả 3 models
    models = {
        'LDA': 'models/lda_model.pkl',
        'KNN': 'models/knn_model.pkl',
        'SVM': 'models/svm_model.pkl'
    }

    for model_name, model_path in models.items():
        print(f"\n{'-'*60}")
        print(f"Model: {model_name}")
        print('-'*60)
        result = predict_fatigue(model_path, non_fatigue_sample, verbose=False)
        print(f"Dự đoán: {result['class_name']}")
        if result['prob_fatigue'] is not None:
            print(f"Xác suất Fatigue: {result['prob_fatigue']*100:.2f}%")

    # Ví dụ 2: Người mỏi
    print("\n\n" + "="*70)
    print("VÍ DỤ 2: NGƯỜI MỎI (Fatigued)")
    print("="*70)

    fatigue_sample = {
        'emg_rms': 0.28,           # EMG cao
        'emg_mav': 0.24,           # EMG cao
        'emg_median_freq': 60,     # Tần số thấp
        'emg_mean_freq': 65,       # Tần số thấp
        'muscle_force': 32,        # Lực cơ giảm
        'heart_rate': 95,          # Nhịp tim cao
        'work_duration': 45,       # Làm việc lâu
        'rest_time': 3,            # Nghỉ ít
        'movement_frequency': 12,  # Chuyển động chậm
        'muscle_tension': 70       # Căng cơ cao
    }

    for model_name, model_path in models.items():
        print(f"\n{'-'*60}")
        print(f"Model: {model_name}")
        print('-'*60)
        result = predict_fatigue(model_path, fatigue_sample, verbose=False)
        print(f"Dự đoán: {result['class_name']}")
        if result['prob_fatigue'] is not None:
            print(f"Xác suất Fatigue: {result['prob_fatigue']*100:.2f}%")

    # Ví dụ 3: Trường hợp biên
    print("\n\n" + "="*70)
    print("VÍ DỤ 3: TRƯỜNG HỢP BIÊN (Borderline)")
    print("="*70)

    borderline_sample = {
        'emg_rms': 0.21,           # Giữa
        'emg_mav': 0.18,           # Giữa
        'emg_median_freq': 72,     # Giữa
        'emg_mean_freq': 76,       # Giữa
        'muscle_force': 38,        # Giữa
        'heart_rate': 85,          # Giữa
        'work_duration': 30,       # Giữa
        'rest_time': 5,            # Giữa
        'movement_frequency': 16,  # Giữa
        'muscle_tension': 50       # Giữa
    }

    for model_name, model_path in models.items():
        print(f"\n{'-'*60}")
        print(f"Model: {model_name}")
        print('-'*60)
        result = predict_fatigue(model_path, borderline_sample, verbose=False)
        print(f"Dự đoán: {result['class_name']}")
        if result['prob_fatigue'] is not None:
            print(f"Xác suất Fatigue: {result['prob_fatigue']*100:.2f}%")

    print("\n" + "="*70)
    print("✓ DEMO HOÀN TẤT")
    print("="*70)

def predict_from_csv(model_path, csv_path, output_path=None):
    """
    Dự đoán cho nhiều mẫu từ file CSV

    Parameters:
    -----------
    model_path : str
        Path đến model file
    csv_path : str
        Path đến CSV file chứa dữ liệu
    output_path : str, optional
        Path để lưu kết quả predictions
    """
    # Load model
    classifier = FatigueMuscleClassifier.load_model(model_path)

    # Load data
    df = pd.read_csv(csv_path)

    # Lấy features
    feature_columns = [
        'emg_rms', 'emg_mav', 'emg_median_freq', 'emg_mean_freq',
        'muscle_force', 'heart_rate', 'work_duration', 'rest_time',
        'movement_frequency', 'muscle_tension'
    ]

    # Kiểm tra xem có đủ features không
    missing_features = [f for f in feature_columns if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in CSV: {missing_features}")

    X = df[feature_columns]

    # Chuẩn hóa và predict
    X_scaled = classifier.scaler.transform(X)
    predictions = classifier.model.predict(X_scaled)

    # Thêm predictions vào dataframe
    df['predicted_label'] = predictions
    df['predicted_class'] = df['predicted_label'].map({0: 'Non-Fatigue', 1: 'Fatigue'})

    # Thêm probability nếu có
    if hasattr(classifier.model, 'predict_proba'):
        proba = classifier.model.predict_proba(X_scaled)
        df['prob_non_fatigue'] = proba[:, 0]
        df['prob_fatigue'] = proba[:, 1]

    # In kết quả
    print(f"\n{'='*60}")
    print(f"DỰ ĐOÁN CHO {len(df)} MẪU TỪ FILE CSV")
    print('='*60)
    print(f"\nPhân bố dự đoán:")
    print(df['predicted_class'].value_counts())

    # Nếu có ground truth, tính accuracy
    if 'label' in df.columns:
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(df['label'], df['predicted_label'])
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nClassification Report:")
        print(classification_report(df['label'], df['predicted_label'],
                                   target_names=['Non-Fatigue', 'Fatigue']))

    # Lưu kết quả
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n✓ Kết quả đã được lưu tại: {output_path}")

    return df

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'csv':
            # Predict từ CSV
            if len(sys.argv) < 4:
                print("Usage: python demo_predict.py csv <model_path> <csv_path> [output_path]")
                sys.exit(1)
            model_path = sys.argv[2]
            csv_path = sys.argv[3]
            output_path = sys.argv[4] if len(sys.argv) > 4 else None
            predict_from_csv(model_path, csv_path, output_path)
        else:
            print("Usage:")
            print("  python demo_predict.py              # Chạy demo examples")
            print("  python demo_predict.py csv <model_path> <csv_path> [output_path]")
    else:
        # Chạy demo
        demo_examples()
