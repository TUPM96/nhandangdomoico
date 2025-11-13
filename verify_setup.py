"""
Verify Setup Script - Kiểm tra tất cả files và dependencies cần thiết

Kiểm tra:
1. Python scripts
2. Data directories và files
3. Models và plots
4. Dependencies
"""

import os
import sys
from pathlib import Path

def check_file(filepath, description):
    """Check if file exists"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"✗ {description}: {filepath} - MISSING!")
        return False

def check_directory(dirpath, description):
    """Check if directory exists"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        files = len(list(Path(dirpath).rglob('*')))
        print(f"✓ {description}: {dirpath} ({files} files)")
        return True
    else:
        print(f"✗ {description}: {dirpath} - MISSING!")
        return False

def check_module(module_name):
    """Check if Python module is installed"""
    try:
        __import__(module_name)
        print(f"✓ Module: {module_name}")
        return True
    except ImportError:
        print(f"✗ Module: {module_name} - NOT INSTALLED!")
        return False

def main():
    print("=" * 70)
    print("MUSCLE FATIGUE DETECTION SYSTEM - SETUP VERIFICATION")
    print("=" * 70)

    all_ok = True

    # Check Python scripts
    print("\n[1] CHECKING PYTHON SCRIPTS")
    print("-" * 70)
    scripts = [
        ('generate_improved_from_real.py', 'Generate synthetic data'),
        ('extract_features.py', 'Extract features from EMG'),
        ('train_models.py', 'Train models'),
        ('test_models.py', 'Test models'),
        ('run_full_pipeline.py', 'Full pipeline'),
        ('demo_predict.py', 'Demo prediction'),
        ('split_dataset_to_files.py', 'Split dataset to files'),
    ]
    for script, desc in scripts:
        if not check_file(script, desc):
            all_ok = False

    # Check data directories
    print("\n[2] CHECKING DATA DIRECTORIES")
    print("-" * 70)
    dirs = [
        ('dataset', 'Original dataset'),
        ('dataset/fatigue', 'Fatigue samples'),
        ('dataset/non fatigue', 'Non-fatigue samples'),
        ('dataset_generated', 'Generated dataset'),
        ('dataset_generated/fatigue', 'Generated fatigue'),
        ('dataset_generated/non_fatigue', 'Generated non-fatigue'),
        ('data_extracted', 'Extracted features'),
        ('data_amplified_final', 'Final amplified data'),
        ('models_final', 'Trained models'),
        ('plots_final', 'Confusion matrices'),
    ]
    for dirpath, desc in dirs:
        if not check_directory(dirpath, desc):
            all_ok = False

    # Check critical data files
    print("\n[3] CHECKING DATA FILES")
    print("-" * 70)
    data_files = [
        ('data_extracted/extracted_features.csv', 'Extracted features (52 samples)'),
        ('data_amplified_final/train_data.csv', 'Training data (2100 samples)'),
        ('data_amplified_final/test_data.csv', 'Test data (900 samples)'),
        ('data_amplified_final/full_data.csv', 'Full data (3000 samples)'),
    ]
    for filepath, desc in data_files:
        if not check_file(filepath, desc):
            all_ok = False

    # Check models
    print("\n[4] CHECKING TRAINED MODELS")
    print("-" * 70)
    models = [
        ('models_final/svm_model.pkl', 'SVM model (91.07%)'),
        ('models_final/lda_model.pkl', 'LDA model (90.27%)'),
        ('models_final/knn_model.pkl', 'KNN model (86.93%)'),
        ('models_final/model_comparison.csv', 'Model comparison'),
        ('models_final/all_results.json', 'All results'),
    ]
    for filepath, desc in models:
        if not check_file(filepath, desc):
            all_ok = False

    # Check plots
    print("\n[5] CHECKING PLOTS")
    print("-" * 70)
    plots = [
        ('plots_final/svm_confusion_matrix.png', 'SVM confusion matrix'),
        ('plots_final/lda_confusion_matrix.png', 'LDA confusion matrix'),
        ('plots_final/knn_confusion_matrix.png', 'KNN confusion matrix'),
    ]
    for filepath, desc in plots:
        if not check_file(filepath, desc):
            all_ok = False

    # Check documentation
    print("\n[6] CHECKING DOCUMENTATION")
    print("-" * 70)
    docs = [
        ('README.md', 'Main README'),
        ('SUCCESS_SUMMARY.md', 'Success summary'),
        ('ANSWERS_QUESTIONS.md', 'Technical Q&A'),
        ('requirements_new.txt', 'Requirements'),
    ]
    for filepath, desc in docs:
        if not check_file(filepath, desc):
            all_ok = False

    # Check Python dependencies
    print("\n[7] CHECKING PYTHON DEPENDENCIES")
    print("-" * 70)
    modules = [
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib',
        'seaborn',
        'joblib',
        'scipy',
    ]
    for module in modules:
        if not check_module(module):
            all_ok = False

    # Summary
    print("\n" + "=" * 70)
    if all_ok:
        print("✓ ALL CHECKS PASSED!")
        print("✓ System is ready to use")
        print("✓ Run: python run_full_pipeline.py")
    else:
        print("✗ SOME CHECKS FAILED!")
        print("✗ Please fix the issues above")
        return 1
    print("=" * 70)

    return 0

if __name__ == '__main__':
    sys.exit(main())
