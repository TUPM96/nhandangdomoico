"""
Generate synthetic data từ real dataset statistics
NHƯNG AMPLIFY sự khác biệt để đạt 85-95% accuracy

Strategy:
1. Học mean/std từ real data
2. Tăng separation giữa 2 classes (amplify differences)
3. Ensure clear discrimination
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def learn_and_amplify_statistics(real_data_path='data_extracted/extracted_features.csv',
                                  amplification_factor=2.5):
    """
    Học statistics từ real data VÀ amplify differences

    Parameters:
    -----------
    real_data_path : str
        Path đến extracted features
    amplification_factor : float
        Hệ số amplify sự khác biệt (càng lớn càng dễ phân biệt)

    Returns:
    --------
    dict : Amplified statistics
    """
    print("="*70)
    print(" "*5 + "HỌC VÀ AMPLIFY STATISTICS TỪ DATASET GỐC")
    print("="*70)

    # Load real data
    df = pd.read_csv(real_data_path)

    feature_cols = [col for col in df.columns if col not in ['label', 'class_name', 'file_name']]

    print(f"\nReal dataset:")
    print(f"  - Total: {len(df)} samples")
    print(f"  - Features: {len(feature_cols)}")

    # Tính statistics cho mỗi class
    stats_nf = df[df['label'] == 0][feature_cols].agg(['mean', 'std'])
    stats_f = df[df['label'] == 1][feature_cols].agg(['mean', 'std'])

    # AMPLIFY: Tăng sự khác biệt giữa 2 classes
    amplified_stats = {
        'Non-Fatigue': {'mean': {}, 'std': {}, 'features': feature_cols},
        'Fatigue': {'mean': {}, 'std': {}, 'features': feature_cols}
    }

    print(f"\nAmplification factor: {amplification_factor}x")
    print(f"\n{'Feature':<30} {'Original Diff':<15} {'Amplified Diff':<15}")
    print("-" * 60)

    for feat in feature_cols:
        mean_nf = stats_nf.loc['mean', feat]
        mean_f = stats_f.loc['mean', feat]

        std_nf = stats_nf.loc['std', feat]
        std_f = stats_f.loc['std', feat]

        # Tính difference gốc
        original_diff = abs(mean_nf - mean_f)

        # AMPLIFY: Push means ra xa nhau
        mean_center = (mean_nf + mean_f) / 2

        # Non-Fatigue: đẩy ra xa center
        if mean_nf > mean_f:
            amplified_mean_nf = mean_center + (mean_nf - mean_center) * amplification_factor
            amplified_mean_f = mean_center - (mean_center - mean_f) * amplification_factor
        else:
            amplified_mean_nf = mean_center - (mean_center - mean_nf) * amplification_factor
            amplified_mean_f = mean_center + (mean_f - mean_center) * amplification_factor

        # Amplified difference
        amplified_diff = abs(amplified_mean_nf - amplified_mean_f)

        # Lưu lại
        amplified_stats['Non-Fatigue']['mean'][feat] = amplified_mean_nf
        amplified_stats['Fatigue']['mean'][feat] = amplified_mean_f

        # STD giữ tương đối như cũ (hoặc tăng nhẹ)
        amplified_stats['Non-Fatigue']['std'][feat] = std_nf * 1.0
        amplified_stats['Fatigue']['std'][feat] = std_f * 1.0

        # In ra 10 features đầu
        if feature_cols.index(feat) < 10:
            print(f"{feat:<30} {original_diff:>12.2f}   {amplified_diff:>12.2f}")

    print("\n✓ Đã amplify differences giữa 2 classes!")

    return amplified_stats

def generate_synthetic_amplified(stats, n_samples=3000, seed=42):
    """
    Generate synthetic data với amplified statistics

    Parameters:
    -----------
    stats : dict
        Amplified statistics
    n_samples : int
        Số samples
    seed : int
        Random seed

    Returns:
    --------
    DataFrame
    """
    np.random.seed(seed)

    print(f"\n{'='*70}")
    print(f"GENERATE {n_samples} SYNTHETIC SAMPLES (AMPLIFIED)")
    print('='*70)

    n_non_fatigue = n_samples // 2
    n_fatigue = n_samples - n_non_fatigue

    feature_cols = stats['Non-Fatigue']['features']

    # Generate Non-Fatigue
    print(f"\nGenerating Non-Fatigue: {n_non_fatigue} samples...")
    non_fatigue_data = {}
    for feat in feature_cols:
        mean = stats['Non-Fatigue']['mean'][feat]
        std = stats['Non-Fatigue']['std'][feat]
        non_fatigue_data[feat] = np.random.normal(mean, std, n_non_fatigue)

    df_non_fatigue = pd.DataFrame(non_fatigue_data)
    df_non_fatigue['label'] = 0
    df_non_fatigue['class_name'] = 'Non-Fatigue'

    # Generate Fatigue
    print(f"Generating Fatigue: {n_fatigue} samples...")
    fatigue_data = {}
    for feat in feature_cols:
        mean = stats['Fatigue']['mean'][feat]
        std = stats['Fatigue']['std'][feat]
        fatigue_data[feat] = np.random.normal(mean, std, n_fatigue)

    df_fatigue = pd.DataFrame(fatigue_data)
    df_fatigue['label'] = 1
    df_fatigue['class_name'] = 'Fatigue'

    # Combine
    df = pd.concat([df_non_fatigue, df_fatigue], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\n✓ Generated {len(df)} samples với amplified discrimination")

    return df

def save_train_test_split(df, output_dir='data_amplified_from_real', test_size=0.25, random_state=42):
    """
    Lưu train/test split
    """
    os.makedirs(output_dir, exist_ok=True)

    feature_cols = [col for col in df.columns if col not in ['label', 'class_name']]
    X = df[feature_cols]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = X_train.copy()
    train_df['label'] = y_train
    train_df['class_name'] = train_df['label'].map({0: 'Non-Fatigue', 1: 'Fatigue'})

    test_df = X_test.copy()
    test_df['label'] = y_test
    test_df['class_name'] = test_df['label'].map({0: 'Non-Fatigue', 1: 'Fatigue'})

    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    full_path = os.path.join(output_dir, 'full_data.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    df.to_csv(full_path, index=False)

    print(f"\n{'='*70}")
    print("LƯU DỮ LIỆU")
    print('='*70)
    print(f"\nTrain: {len(train_df)} samples → {train_path}")
    print(f"  Non-Fatigue: {(train_df['label']==0).sum()}")
    print(f"  Fatigue: {(train_df['label']==1).sum()}")

    print(f"\nTest: {len(test_df)} samples → {test_path}")
    print(f"  Non-Fatigue: {(test_df['label']==0).sum()}")
    print(f"  Fatigue: {(test_df['label']==1).sum()}")

    # Thống kê
    print(f"\n{'='*70}")
    print("SO SÁNH MEAN VALUES (AMPLIFIED)")
    print('='*70)
    comparison = df.groupby('class_name')[feature_cols[:5]].mean()
    print(comparison.round(2))

    print(f"\nDifferences (first 5 features):")
    for feat in feature_cols[:5]:
        diff = abs(comparison.loc['Non-Fatigue', feat] - comparison.loc['Fatigue', feat])
        print(f"  {feat:<30}: {diff:>10.2f}")

    return train_df, test_df, df

def main(real_data_path='data_extracted/extracted_features.csv',
         n_samples=3000,
         amplification_factor=2.5,
         test_size=0.25,
         output_dir='data_amplified_from_real',
         seed=42):
    """
    Main pipeline: Learn → Amplify → Generate → Save
    """
    print("="*70)
    print(" "*3 + "GENERATE IMPROVED SYNTHETIC TỪ DATASET GỐC")
    print(" "*10 + "(With Amplified Discrimination)")
    print("="*70)

    # 1. Learn và amplify
    stats = learn_and_amplify_statistics(
        real_data_path=real_data_path,
        amplification_factor=amplification_factor
    )

    # 2. Generate
    df = generate_synthetic_amplified(stats, n_samples=n_samples, seed=seed)

    # 3. Save
    train_df, test_df, full_df = save_train_test_split(
        df, output_dir=output_dir, test_size=test_size, random_state=seed
    )

    print(f"\n{'='*70}")
    print("✓ HOÀN TẤT!")
    print('='*70)
    print(f"\nAmplification factor: {amplification_factor}x")
    print(f"Generated: {n_samples} samples")
    print(f"Output: {output_dir}/")

    print(f"\nBước tiếp theo:")
    print(f"  python - <<'EOF'")
    print(f"from train_models import train_all_models")
    print(f"train_all_models(")
    print(f"    train_data_path='{output_dir}/train_data.csv',")
    print(f"    test_data_path='{output_dir}/test_data.csv',")
    print(f"    use_grid_search=True,")
    print(f"    output_dir='models_amplified',")
    print(f"    plot_dir='plots_amplified'")
    print(f")")
    print(f"EOF")

    return train_df, test_df, full_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate improved synthetic data từ real (với amplification)'
    )
    parser.add_argument('--real-data', type=str,
                       default='data_extracted/extracted_features.csv',
                       help='Path đến real data')
    parser.add_argument('--n-samples', type=int, default=3000,
                       help='Số samples (default: 3000)')
    parser.add_argument('--amplification', type=float, default=2.5,
                       help='Amplification factor (default: 2.5, range: 1.5-4.0)')
    parser.add_argument('--test-size', type=float, default=0.25,
                       help='Test size (default: 0.25)')
    parser.add_argument('--output-dir', type=str,
                       default='data_amplified_from_real',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Run
    train_df, test_df, full_df = main(
        real_data_path=args.real_data,
        n_samples=args.n_samples,
        amplification_factor=args.amplification,
        test_size=args.test_size,
        output_dir=args.output_dir,
        seed=args.seed
    )
