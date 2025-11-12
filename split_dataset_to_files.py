"""
Split generated dataset into individual files like the original dataset structure.

Creates:
- dataset_generated/fatigue/sample_XXX_F.csv
- dataset_generated/non_fatigue/sample_XXX_NF.csv
"""

import pandas as pd
import os
from pathlib import Path

def split_dataset_to_files(input_csv='data_amplified_final/full_data.csv',
                          output_base='dataset_generated'):
    """
    Split CSV dataset into individual files.

    Args:
        input_csv: Path to the full dataset CSV
        output_base: Base directory for output
    """
    print(f"Reading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Create output directories
    fatigue_dir = Path(output_base) / 'fatigue'
    non_fatigue_dir = Path(output_base) / 'non_fatigue'
    fatigue_dir.mkdir(parents=True, exist_ok=True)
    non_fatigue_dir.mkdir(parents=True, exist_ok=True)

    print(f"Total samples: {len(df)}")
    print(f"Fatigue samples: {len(df[df['label'] == 1])}")
    print(f"Non-Fatigue samples: {len(df[df['label'] == 0])}")

    # Split by label
    fatigue_samples = df[df['label'] == 1].reset_index(drop=True)
    non_fatigue_samples = df[df['label'] == 0].reset_index(drop=True)

    # Save fatigue samples
    print("\nCreating fatigue sample files...")
    for idx, row in fatigue_samples.iterrows():
        sample_num = str(idx + 1).zfill(4)
        filename = f"sample_{sample_num}_F.csv"
        filepath = fatigue_dir / filename

        # Drop the label column for individual files
        sample_data = row.drop('label').to_frame().T
        sample_data.to_csv(filepath, index=False)

        if (idx + 1) % 500 == 0:
            print(f"  Created {idx + 1} fatigue samples...")

    print(f"✓ Created {len(fatigue_samples)} fatigue sample files")

    # Save non-fatigue samples
    print("\nCreating non-fatigue sample files...")
    for idx, row in non_fatigue_samples.iterrows():
        sample_num = str(idx + 1).zfill(4)
        filename = f"sample_{sample_num}_NF.csv"
        filepath = non_fatigue_dir / filename

        # Drop the label column for individual files
        sample_data = row.drop('label').to_frame().T
        sample_data.to_csv(filepath, index=False)

        if (idx + 1) % 500 == 0:
            print(f"  Created {idx + 1} non-fatigue samples...")

    print(f"✓ Created {len(non_fatigue_samples)} non-fatigue sample files")

    print(f"\n{'='*60}")
    print("DATASET STRUCTURE CREATED")
    print(f"{'='*60}")
    print(f"Output directory: {output_base}/")
    print(f"  - fatigue/       : {len(fatigue_samples)} files")
    print(f"  - non_fatigue/   : {len(non_fatigue_samples)} files")
    print(f"\nEach file contains 17 features extracted from EMG signals")
    print("Files follow naming convention: sample_XXXX_F.csv / sample_XXXX_NF.csv")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Split dataset into individual files')
    parser.add_argument('--input', default='data_amplified_final/full_data.csv',
                        help='Input CSV file (default: data_amplified_final/full_data.csv)')
    parser.add_argument('--output', default='dataset_generated',
                        help='Output base directory (default: dataset_generated)')

    args = parser.parse_args()

    split_dataset_to_files(args.input, args.output)
    print("\n✓ Done!")
