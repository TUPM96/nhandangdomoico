"""
Generate synthetic data DỰA TRÊN STATISTICS TỪ DATASET GỐC
Học mean/std từ extracted_features.csv → Generate nhiều samples tương tự
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def learn_statistics_from_real_data(real_data_path='data_extracted/extracted_features.csv'):
    """
    Học statistics (mean, std) từ real data

    Returns:
    --------
    dict : Statistics cho mỗi class
    """
    print("="*70)
    print(" "*10 + "HỌC STATISTICS TỪ DATASET GỐC")
    print("="*70)

    # Load real data
    df = pd.read_csv(real_data_path)

    # Feature columns
    feature_cols = [col for col in df.columns if col not in ['label', 'class_name', 'file_name']]

    print(f"\nReal dataset:")
    print(f"  - Total: {len(df)} samples")
    print(f"  - Non-Fatigue: {(df['label']==0).sum()} samples")
    print(f"  - Fatigue: {(df['label']==1).sum()} samples")
    print(f"  - Features: {len(feature_cols)}")

    # Tính statistics cho mỗi class
    stats = {}

    for label, class_name in [(0, 'Non-Fatigue'), (1, 'Fatigue')]:
        class_data = df[df['label'] == label]

        stats[class_name] = {
            'mean': {},
            'std': {},
            'features': feature_cols
        }

        print(f"\n{class_name} statistics:")
        for feat in feature_cols:
            mean = class_data[feat].mean()
            std = class_data[feat].std()

            stats[class_name]['mean'][feat] = mean
            stats[class_name]['std'][feat] = std

            print(f"  {feat:25s}: mean={mean:10.2f}, std={std:10.2f}")

    return stats

def generate_synthetic_from_stats(stats, n_samples=2000, seed=42):
    """
    Generate synthetic data dựa trên statistics đã học

    Parameters:
    -----------
    stats : dict
        Statistics từ learn_statistics_from_real_data()
    n_samples : int
        Số samples cần generate
    seed : int
        Random seed

    Returns:
    --------
    DataFrame : Synthetic data
    """
    np.random.seed(seed)

    print(f"\n{'='*70}")
    print(f"GENERATE {n_samples} SYNTHETIC SAMPLES")
    print('='*70)

    # Tính số samples cho mỗi class (balanced)
    n_non_fatigue = n_samples // 2
    n_fatigue = n_samples - n_non_fatigue

    # Get feature list
    feature_cols = stats['Non-Fatigue']['features']

    # ============= NON-FATIGUE =============
    print(f"\nGenerating Non-Fatigue: {n_non_fatigue} samples...")
    non_fatigue_data = {}

    for feat in feature_cols:
        mean = stats['Non-Fatigue']['mean'][feat]
        std = stats['Non-Fatigue']['std'][feat]

        # Generate với distribution tương tự
        # Increase std một chút để có variance (tránh overfit)
        generated = np.random.normal(mean, std * 1.2, n_non_fatigue)
        non_fatigue_data[feat] = generated

    df_non_fatigue = pd.DataFrame(non_fatigue_data)
    df_non_fatigue['label'] = 0
    df_non_fatigue['class_name'] = 'Non-Fatigue'

    # ============= FATIGUE =============
    print(f"Generating Fatigue: {n_fatigue} samples...")
    fatigue_data = {}

    for feat in feature_cols:
        mean = stats['Fatigue']['mean'][feat]
        std = stats['Fatigue']['std'][feat]

        # Generate
        generated = np.random.normal(mean, std * 1.2, n_fatigue)
        fatigue_data[feat] = generated

    df_fatigue = pd.DataFrame(fatigue_data)
    df_fatigue['label'] = 1
    df_fatigue['class_name'] = 'Fatigue'

    # Combine
    df = pd.concat([df_non_fatigue, df_fatigue], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Clip extreme values (giữ trong phạm vi hợp lý)
    for feat in feature_cols:
        # Lấy min/max từ real data + margin
        min_val = min(stats['Non-Fatigue']['mean'][feat], stats['Fatigue']['mean'][feat])
        max_val = max(stats['Non-Fatigue']['mean'][feat], stats['Fatigue']['mean'][feat])

        # Add margin
        margin = abs(max_val - min_val) * 0.5
        df[feat] = df[feat].clip(min_val - margin, max_val + margin)

    print(f"\n✓ Generated {len(df)} samples")
    print(f"  - Non-Fatigue: {(df['label']==0).sum()}")
    print(f"  - Fatigue: {(df['label']==1).sum()}")

    return df

def save_train_test_data(df, output_dir='data_synthetic_from_real', test_size=0.25, random_state=42):
    """
    Chia và lưu train/test
    """
    os.makedirs(output_dir, exist_ok=True)

    # Features và labels
    feature_cols = [col for col in df.columns if col not in ['label', 'class_name']]
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

    # Lưu
    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    full_path = os.path.join(output_dir, 'full_data.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    df.to_csv(full_path, index=False)

    print(f"\n{'='*70}")
    print("ĐÃ LƯU DỮ LIỆU")
    print('='*70)
    print(f"\nTrain set: {len(train_df)} samples → {train_path}")
    print(f"  - Non-Fatigue: {(train_df['label']==0).sum()}")
    print(f"  - Fatigue: {(train_df['label']==1).sum()}")

    print(f"\nTest set: {len(test_df)} samples → {test_path}")
    print(f"  - Non-Fatigue: {(test_df['label']==0).sum()}")
    print(f"  - Fatigue: {(test_df['label']==1).sum()}")

    print(f"\nFull data: {len(df)} samples → {full_path}")

    # Thống kê
    print(f"\n{'='*70}")
    print("THỐNG KÊ SYNTHETIC DATA")
    print('='*70)

    print("\nMean values by class:")
    print(df.groupby('class_name')[feature_cols[:5]].mean().round(2))

    print(f"\nFeatures: {', '.join(feature_cols[:5])}...")
    print(f"Total features: {len(feature_cols)}")

    return train_df, test_df, df

def main(real_data_path='data_extracted/extracted_features.csv',
         n_samples=2000,
         test_size=0.25,
         output_dir='data_synthetic_from_real',
         seed=42):
    """
    Main pipeline: Học từ real data → Generate synthetic → Save
    """
    print("="*70)
    print(" "*5 + "GENERATE SYNTHETIC DATA TỪ DATASET GỐC")
    print("="*70)

    # 1. Học statistics từ real data
    stats = learn_statistics_from_real_data(real_data_path)

    # 2. Generate synthetic data
    df = generate_synthetic_from_stats(stats, n_samples=n_samples, seed=seed)

    # 3. Save train/test
    train_df, test_df, full_df = save_train_test_data(
        df, output_dir=output_dir, test_size=test_size, random_state=seed
    )

    print(f"\n{'='*70}")
    print("✓ HOÀN TẤT!")
    print('='*70)
    print("\nBước tiếp theo:")
    print("  python train_models.py \\")
    print(f"    --train-data {output_dir}/train_data.csv \\")
    print(f"    --test-data {output_dir}/test_data.csv")

    return train_df, test_df, full_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic data từ real dataset statistics')
    parser.add_argument('--real-data', type=str, default='data_extracted/extracted_features.csv',
                       help='Path đến extracted features từ real data')
    parser.add_argument('--n-samples', type=int, default=2000,
                       help='Số samples cần generate (default: 2000)')
    parser.add_argument('--test-size', type=float, default=0.25,
                       help='Tỷ lệ test set (default: 0.25)')
    parser.add_argument('--output-dir', type=str, default='data_synthetic_from_real',
                       help='Thư mục output (default: data_synthetic_from_real)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Run main pipeline
    train_df, test_df, full_df = main(
        real_data_path=args.real_data,
        n_samples=args.n_samples,
        test_size=args.test_size,
        output_dir=args.output_dir,
        seed=args.seed
    )
