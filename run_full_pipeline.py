"""
Script chạy toàn bộ pipeline: Generate data -> Train -> Test
Hệ thống nhận dạng mỏi cơ với LDA, KNN, SVM
"""

import os
import sys
import argparse
from datetime import datetime

# Import các modules
from generate_data import save_train_test_data
from train_models import train_all_models
from test_models import test_all_models

def run_full_pipeline(n_samples=2000, test_size=0.25, use_grid_search=True, seed=42):
    """
    Chạy toàn bộ pipeline từ đầu đến cuối

    Parameters:
    -----------
    n_samples : int
        Số lượng mẫu tạo ra
    test_size : float
        Tỷ lệ test set
    use_grid_search : bool
        Sử dụng GridSearchCV
    seed : int
        Random seed
    """
    print("="*70)
    print(" "*10 + "HỆ THỐNG NHẬN DẠNG MỎI CƠ - FULL PIPELINE")
    print(" "*15 + "LDA | KNN | SVM")
    print("="*70)

    start_time = datetime.now()

    # Tạo thư mục output
    data_dir = 'data_generated'
    models_dir = 'models'
    plots_dir = 'plots'
    test_results_dir = 'test_results'

    for dir_path in [data_dir, models_dir, plots_dir, test_results_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # ============= BƯỚC 1: GENERATE DATA =============
    print(f"\n{'='*70}")
    print("BƯỚC 1: GENERATE SYNTHETIC DATA")
    print('='*70)

    train_df, test_df, full_df = save_train_test_data(
        output_dir=data_dir,
        n_samples=n_samples,
        test_size=test_size,
        seed=seed
    )

    train_data_path = os.path.join(data_dir, 'train_data.csv')
    test_data_path = os.path.join(data_dir, 'test_data.csv')

    # ============= BƯỚC 2: TRAIN MODELS =============
    print(f"\n{'='*70}")
    print("BƯỚC 2: TRAIN MODELS (LDA, KNN, SVM)")
    print('='*70)

    train_results, train_comparison = train_all_models(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        use_grid_search=use_grid_search,
        output_dir=models_dir,
        plot_dir=plots_dir
    )

    # ============= BƯỚC 3: TEST MODELS =============
    print(f"\n{'='*70}")
    print("BƯỚC 3: TEST MODELS")
    print('='*70)

    test_results, test_comparison = test_all_models(
        test_data_path=test_data_path,
        models_dir=models_dir,
        output_dir=test_results_dir
    )

    # ============= TỔNG KẾT =============
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'='*70}")
    print("TỔNG KẾT KẾT QUẢ")
    print('='*70)

    print(f"\n📊 Số liệu dữ liệu:")
    print(f"  - Tổng số mẫu: {n_samples}")
    print(f"  - Train set: {len(train_df)} mẫu")
    print(f"  - Test set: {len(test_df)} mẫu")

    print(f"\n🎯 Kết quả Test (các models):")
    print(test_comparison)

    # Tìm model tốt nhất
    best_model = test_comparison['Accuracy'].idxmax()
    best_accuracy = test_comparison.loc[best_model, 'Accuracy']

    print(f"\n🏆 Model tốt nhất: {best_model.upper()}")
    print(f"   - Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"   - Precision: {test_comparison.loc[best_model, 'Precision']:.4f}")
    print(f"   - Recall: {test_comparison.loc[best_model, 'Recall']:.4f}")
    print(f"   - F1-Score: {test_comparison.loc[best_model, 'F1-Score']:.4f}")

    # Kiểm tra đạt mục tiêu
    print(f"\n📈 Đánh giá mục tiêu (Accuracy >= 85%):")
    for model_name in test_comparison.index:
        acc = test_comparison.loc[model_name, 'Accuracy']
        status = "✓ ĐẠT" if acc >= 0.85 else "✗ CHƯA ĐẠT"
        print(f"   - {model_name.upper()}: {acc:.4f} ({acc*100:.2f}%) - {status}")

    # Kiểm tra xem có model nào đạt target không
    models_达标 = test_comparison[test_comparison['Accuracy'] >= 0.85]
    if len(models_达标) > 0:
        print(f"\n✓✓✓ THÀNH CÔNG! Có {len(models_达标)}/{len(test_comparison)} model(s) đạt mục tiêu >= 85% ✓✓✓")
    else:
        print(f"\n⚠ Chưa có model nào đạt mục tiêu 85%. Khuyến nghị:")
        print("  1. Tăng số lượng mẫu training (n_samples)")
        print("  2. Thêm features quan trọng hơn")
        print("  3. Điều chỉnh param grid cho GridSearchCV")
        print("  4. Thử feature engineering")

    print(f"\n⏱ Thời gian chạy: {duration:.2f} giây ({duration/60:.2f} phút)")

    print(f"\n📁 Các file output:")
    print(f"  - Data: {data_dir}/")
    print(f"  - Models: {models_dir}/")
    print(f"  - Plots: {plots_dir}/")
    print(f"  - Test results: {test_results_dir}/")

    print(f"\n{'='*70}")
    print("✓ HOÀN TẤT TOÀN BỘ PIPELINE!")
    print('='*70)

    return {
        'train_results': train_results,
        'test_results': test_results,
        'train_comparison': train_comparison,
        'test_comparison': test_comparison,
        'best_model': best_model,
        'best_accuracy': best_accuracy,
        'duration': duration
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chạy toàn bộ pipeline nhận dạng mỏi cơ')
    parser.add_argument('--n-samples', type=int, default=2000,
                       help='Số lượng mẫu tạo ra (mặc định: 2000)')
    parser.add_argument('--test-size', type=float, default=0.25,
                       help='Tỷ lệ test set (mặc định: 0.25)')
    parser.add_argument('--no-grid-search', action='store_true',
                       help='Không sử dụng GridSearchCV (train nhanh hơn)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (mặc định: 42)')

    args = parser.parse_args()

    # Chạy pipeline
    results = run_full_pipeline(
        n_samples=args.n_samples,
        test_size=args.test_size,
        use_grid_search=not args.no_grid_search,
        seed=args.seed
    )
