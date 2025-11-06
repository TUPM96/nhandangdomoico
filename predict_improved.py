#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict/Evaluate với model đã train (best_svm_improved.pkl)
Sử dụng cùng pipeline tiền xử lý và feature extraction
"""

import os
import glob
import numpy as np
import pandas as pd
import joblib
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from improved_preprocessing import bandpass_filter, notch_filter, extract_advanced_features

FS = 1000.0
RANGES = [(15000, 25000), (30000, 35000)]


def load_and_preprocess(csv_path):
    """Load CSV → filter → extract features"""
    try:
        df = pd.read_csv(csv_path, header=None)
        sig = df.iloc[:, 0].values.astype(float)
        segments = [sig[r[0]:r[1]] for r in RANGES]
        sig = np.concatenate(segments)

        # Tiền xử lý
        sig = bandpass_filter(sig, lowcut=20, highcut=450, fs=FS)
        sig = notch_filter(sig, freq=50, Q=30, fs=FS)

        # Trích xuất features
        feats = extract_advanced_features(sig, fs=FS)
        return feats

    except Exception as e:
        print(f"[ERROR] {csv_path}: {e}")
        return None


def predict_single(model_path, csv_path):
    """Dự đoán 1 file CSV"""
    model_dict = joblib.load(model_path)

    feats = load_and_preprocess(csv_path)
    if feats is None:
        return

    # Transform features qua pipeline
    X = feats.reshape(1, -1)
    X_scaled = model_dict['scaler'].transform(X)
    X_selected = model_dict['feature_selector'].transform(X_scaled)

    # Predict
    y_pred = model_dict['model'].predict(X_selected)[0]
    y_proba = model_dict['model'].predict_proba(X_selected)[0, 1]

    label = "Fatigue" if y_pred == 1 else "Non-fatigue"
    print(f"{os.path.basename(csv_path)} → {label} (probability={y_proba:.4f})")


def evaluate_folder(model_path, data_root):
    """Đánh giá toàn bộ folder"""
    model_dict = joblib.load(model_path)

    # Tìm folder non-fatigue
    neg_dir = None
    for cand in ["non fatigue", "non_fatigue"]:
        if os.path.isdir(os.path.join(data_root, cand)):
            neg_dir = cand
            break

    y_true, y_pred, y_proba, files = [], [], [], []

    for folder, label in [("fatigue", 1), (neg_dir, 0)]:
        folder_path = os.path.join(data_root, folder)
        csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

        for csv_path in csv_files:
            feats = load_and_preprocess(csv_path)
            if feats is None:
                continue

            # Transform
            X = feats.reshape(1, -1)
            X_scaled = model_dict['scaler'].transform(X)
            X_selected = model_dict['feature_selector'].transform(X_scaled)

            # Predict
            pred = model_dict['model'].predict(X_selected)[0]
            proba = model_dict['model'].predict_proba(X_selected)[0, 1]

            y_true.append(label)
            y_pred.append(pred)
            y_proba.append(proba)
            files.append(os.path.basename(csv_path))

            lbl = "Fatigue" if pred == 1 else "Non-fatigue"
            print(f"{os.path.basename(csv_path)} → {lbl} (prob={proba:.4f})")

    # Metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)

    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")

    # Save predictions
    df = pd.DataFrame({'file': files, 'y_true': y_true, 'y_pred': y_pred, 'probability': y_proba})
    df.to_csv('predictions_improved.csv', index=False)
    print("\n💾 Saved: predictions_improved.csv")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Non-fatigue', 'Fatigue'],
                yticklabels=['Non-fatigue', 'Fatigue'])
    plt.title(f'Confusion Matrix (Accuracy={acc:.2%})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('cm_improved_eval.png', dpi=200)
    print("💾 Saved: cm_improved_eval.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='best_svm_improved.pkl', help='Path to model')
    parser.add_argument('--data', required=True, help='CSV file or folder')
    args = parser.parse_args()

    if os.path.isdir(args.data):
        evaluate_folder(args.model, args.data)
    else:
        predict_single(args.model, args.data)


if __name__ == "__main__":
    main()
