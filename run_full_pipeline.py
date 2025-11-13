"""
Script chạy toàn bộ pipeline: Generate data -> Train -> Test
Hệ thống nhận dạng mỏi cơ với LDA, KNN, SVM
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_command(cmd, description):
    """Run a command and print output"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print('='*70)

    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"✗ Error running: {cmd}")
        sys.exit(1)

    return result

def run_full_pipeline(amplification=3.3, n_samples=3000, use_grid_search=True, seed=42):
    """
    Chạy toàn bộ pipeline từ đầu đến cuối

    Parameters:
    -----------
    amplification : float
        Amplification factor (default: 3.3)
    n_samples : int
        Số lượng mẫu tạo ra (default: 3000)
    use_grid_search : bool
        Sử dụng GridSearchCV (default: True)
    seed : int
        Random seed (default: 42)
    """
    print("="*70)
    print(" "*10 + "HỆ THỐNG NHẬN DẠNG MỎI CƠ - FULL PIPELINE")
    print(" "*15 + "LDA | KNN | SVM")
    print("="*70)

    start_time = datetime.now()

    # Tạo thư mục output
    output_dir = 'data_amplified_final'
    models_dir = 'models_final'
    plots_dir = 'plots_final'

    for dir_path in [output_dir, models_dir, plots_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # ============= BƯỚC 1: GENERATE DATA =============
    print(f"\n{'='*70}")
    print("BƯỚC 1: GENERATE SYNTHETIC DATA (AMPLIFIED)")
    print('='*70)

    cmd_generate = (
        f"python generate_improved_from_real.py "
        f"--amplification {amplification} "
        f"--n-samples {n_samples} "
        f"--output-dir {output_dir} "
        f"--seed {seed}"
    )

    run_command(cmd_generate, "Generating synthetic data with amplification...")

    # ============= BƯỚC 2: TRAIN MODELS =============
    print(f"\n{'='*70}")
    print("BƯỚC 2: TRAIN MODELS (LDA, KNN, SVM)")
    print('='*70)

    cmd_train = "python train_models.py"
    run_command(cmd_train, "Training all models with GridSearchCV...")

    # ============= BƯỚC 3: TEST MODELS =============
    print(f"\n{'='*70}")
    print("BƯỚC 3: TEST MODELS")
    print('='*70)

    cmd_test = "python test_models.py"
    run_command(cmd_test, "Testing and evaluating all models...")

    # ============= TỔNG KẾT =============
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'='*70}")
    print("TỔNG KẾT")
    print('='*70)

    print(f"\n✓ Pipeline completed successfully!")
    print(f"\n📊 Settings:")
    print(f"  - Amplification factor: {amplification}x")
    print(f"  - Total samples: {n_samples}")
    print(f"  - GridSearchCV: {'Enabled' if use_grid_search else 'Disabled'}")
    print(f"  - Random seed: {seed}")

    print(f"\n⏱ Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")

    print(f"\n📁 Output directories:")
    print(f"  - Data: {output_dir}/")
    print(f"  - Models: {models_dir}/")
    print(f"  - Plots: {plots_dir}/")

    print(f"\n🎯 Check results:")
    print(f"  - Model comparison: {models_dir}/model_comparison.csv")
    print(f"  - Confusion matrices: {plots_dir}/")
    print(f"  - Detailed results: {models_dir}/all_results.json")

    print(f"\n{'='*70}")
    print("✓ HOÀN TẤT TOÀN BỘ PIPELINE!")
    print('='*70)

    return duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Chạy toàn bộ pipeline nhận dạng mỏi cơ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chạy với settings mặc định (3.3x amplification, 3000 samples)
  python run_full_pipeline.py

  # Chạy với amplification factor khác
  python run_full_pipeline.py --amplification 3.5

  # Chạy với số samples khác
  python run_full_pipeline.py --n-samples 5000

  # Chạy không dùng GridSearch (nhanh hơn)
  python run_full_pipeline.py --no-grid-search
        """
    )

    parser.add_argument('--amplification', type=float, default=3.3,
                       help='Amplification factor (default: 3.3)')
    parser.add_argument('--n-samples', type=int, default=3000,
                       help='Number of samples to generate (default: 3000)')
    parser.add_argument('--no-grid-search', action='store_true',
                       help='Disable GridSearchCV (faster training)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Chạy pipeline
    duration = run_full_pipeline(
        amplification=args.amplification,
        n_samples=args.n_samples,
        use_grid_search=not args.no_grid_search,
        seed=args.seed
    )
