"""
Script test models đã train cho nhận dạng mỏi cơ
Có thể test từng model riêng lẻ hoặc tất cả models
"""

import numpy as np
import pandas as pd
import joblib
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

def load_test_data(test_data_path):
    """
    Load test data
    """
    print(f"\nĐang load test data từ {test_data_path}...")
    test_df = pd.read_csv(test_data_path)

    # Tách features và labels
    feature_columns = [col for col in test_df.columns if col not in ['label', 'class_name']]
    X_test = test_df[feature_columns]
    y_test = test_df['label']

    print(f"✓ Test set: {len(test_df)} mẫu")
    print(f"  - Non-Fatigue: {(y_test==0).sum()} mẫu")
    print(f"  - Fatigue: {(y_test==1).sum()} mẫu")

    return X_test, y_test, feature_columns

def test_single_model(model_path, X_test, y_test, verbose=True):
    """
    Test một model đơn lẻ

    Parameters:
    -----------
    model_path : str
        Path đến model file
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    verbose : bool
        In chi tiết kết quả

    Returns:
    --------
    dict : Dictionary chứa metrics
    """
    # Load model
    print(f"\nĐang load model từ {model_path}...")
    data = joblib.load(model_path)

    model = data['model']
    scaler = data['scaler']
    model_type = data['model_type']
    best_params = data.get('best_params')

    # Chuẩn hóa test data
    X_test_scaled = scaler.transform(X_test)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Tính metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Non-Fatigue', 'Fatigue'])

    metrics = {
        'model_type': model_type,
        'model_path': model_path,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'best_params': best_params,
        'predictions': y_pred.tolist()
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"TEST RESULTS: {model_type.upper()}")
        print('='*60)
        if best_params:
            print(f"\nBest parameters: {best_params}")
        print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(report)

    return metrics

def test_all_models(test_data_path, models_dir='models', output_dir='test_results'):
    """
    Test tất cả models và so sánh kết quả

    Parameters:
    -----------
    test_data_path : str
        Path đến test data
    models_dir : str
        Thư mục chứa các models
    output_dir : str
        Thư mục lưu kết quả test
    """
    print(f"\n{'='*60}")
    print("TESTING ALL MODELS - FATIGUE MUSCLE DETECTION")
    print('='*60)

    # Load test data
    X_test, y_test, feature_columns = load_test_data(test_data_path)

    # Tìm tất cả model files
    model_files = {
        'lda': os.path.join(models_dir, 'lda_model.pkl'),
        'knn': os.path.join(models_dir, 'knn_model.pkl'),
        'svm': os.path.join(models_dir, 'svm_model.pkl')
    }

    # Test từng model
    results = {}
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            print(f"\n{'='*60}")
            print(f"TESTING MODEL: {model_name.upper()}")
            print('='*60)

            metrics = test_single_model(model_path, X_test, y_test)
            results[model_name] = metrics
        else:
            print(f"\n⚠ Model file không tồn tại: {model_path}")

    # So sánh kết quả
    if results:
        print(f"\n{'='*60}")
        print("SO SÁNH KẾT QUẢ TEST")
        print('='*60)

        comparison_df = pd.DataFrame({
            model: {
                'Accuracy': results[model]['accuracy'],
                'Precision': results[model]['precision'],
                'Recall': results[model]['recall'],
                'F1-Score': results[model]['f1_score']
            }
            for model in results.keys()
        }).T

        print("\n", comparison_df)

        # Tìm model tốt nhất
        best_model = comparison_df['Accuracy'].idxmax()
        best_accuracy = comparison_df.loc[best_model, 'Accuracy']

        print(f"\n✓ Model tốt nhất: {best_model.upper()}")
        print(f"✓ Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

        # Kiểm tra xem có đạt target accuracy không
        if best_accuracy >= 0.85:
            print(f"\n✓✓✓ ĐẠT MỤC TIÊU! Accuracy >= 85% ✓✓✓")
        else:
            print(f"\n⚠ Chưa đạt mục tiêu 85%. Cần cải thiện model hoặc dữ liệu.")

        # Lưu kết quả
        os.makedirs(output_dir, exist_ok=True)

        comparison_path = os.path.join(output_dir, 'test_comparison.csv')
        comparison_df.to_csv(comparison_path)
        print(f"\n✓ Kết quả test đã được lưu tại: {comparison_path}")

        # Vẽ biểu đồ so sánh
        plot_comparison(comparison_df, output_dir)

        return results, comparison_df
    else:
        print("\n⚠ Không tìm thấy model nào để test!")
        return None, None

def plot_comparison(comparison_df, output_dir='test_results'):
    """
    Vẽ biểu đồ so sánh các models
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('So sánh Performance các Models', fontsize=16, fontweight='bold')

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        values = comparison_df[metric].values
        models = comparison_df.index.tolist()

        bars = ax.bar(models, values, color=colors[idx], alpha=0.7, edgecolor='black')

        # Thêm giá trị lên trên mỗi cột
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}\n({height*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticklabels(models, fontsize=11, fontweight='bold')

        # Thêm đường target 85%
        if metric == 'Accuracy':
            ax.axhline(y=0.85, color='red', linestyle='--', linewidth=2, label='Target (85%)')
            ax.legend()

    plt.tight_layout()

    # Lưu plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'models_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Biểu đồ so sánh đã được lưu tại: {plot_path}")

def predict_single_sample(model_path, sample_data, feature_names=None):
    """
    Dự đoán cho một mẫu đơn lẻ

    Parameters:
    -----------
    model_path : str
        Path đến model file
    sample_data : array-like or dict
        Dữ liệu mẫu (có thể là array hoặc dict với feature names)
    feature_names : list, optional
        Tên các features (nếu sample_data là array)

    Returns:
    --------
    dict : Kết quả dự đoán
    """
    # Load model
    data = joblib.load(model_path)
    model = data['model']
    scaler = data['scaler']

    # Chuẩn bị dữ liệu
    if isinstance(sample_data, dict):
        sample_array = np.array([list(sample_data.values())])
    else:
        sample_array = np.array(sample_data).reshape(1, -1)

    # Chuẩn hóa
    sample_scaled = scaler.transform(sample_array)

    # Predict
    prediction = model.predict(sample_scaled)[0]
    prediction_proba = model.predict_proba(sample_scaled)[0] if hasattr(model, 'predict_proba') else None

    result = {
        'prediction': int(prediction),
        'class_name': 'Fatigue' if prediction == 1 else 'Non-Fatigue',
        'probability': prediction_proba.tolist() if prediction_proba is not None else None
    }

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test models cho nhận dạng mỏi cơ')
    parser.add_argument('--test-data', type=str, default='data_amplified_final/test_data.csv',
                       help='Path đến test data')
    parser.add_argument('--models-dir', type=str, default='models_final',
                       help='Thư mục chứa models')
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='Thư mục lưu kết quả test')
    parser.add_argument('--model', type=str, choices=['lda', 'knn', 'svm', 'all'], default='all',
                       help='Model để test (mặc định: all)')

    args = parser.parse_args()

    if args.model == 'all':
        # Test tất cả models
        results, comparison = test_all_models(
            test_data_path=args.test_data,
            models_dir=args.models_dir,
            output_dir=args.output_dir
        )
    else:
        # Test một model cụ thể
        X_test, y_test, _ = load_test_data(args.test_data)
        model_path = os.path.join(args.models_dir, f'{args.model}_model.pkl')

        if os.path.exists(model_path):
            metrics = test_single_model(model_path, X_test, y_test)

            # Kiểm tra accuracy
            if metrics['accuracy'] >= 0.85:
                print(f"\n✓✓✓ ĐẠT MỤC TIÊU! Accuracy >= 85% ✓✓✓")
            else:
                print(f"\n⚠ Chưa đạt mục tiêu 85%")
        else:
            print(f"\n⚠ Model file không tồn tại: {model_path}")

    print("\n✓ Hoàn tất testing!")
