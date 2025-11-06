#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict/Evaluate sEMG fatigue using the SAME pipeline as training (sEMGSVMClassifier).

- Works with model saved as a dict: {'model','scaler','feature_selector'} in best_svm_model.pkl
- Single CSV: predict label + score
- Folder with {fatigue, non fatigue}: evaluate metrics + save predictions and plots
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import joblib

# import the training class (same file you used to train)
from sEMG_SVM_Classification import sEMGSVMClassifier

# === Use the SAME ranges & fs as training ===
FS = 1000.0
RANGES = [(15000, 25000), (30000, 35000)]  # identical to training code


def _predict_features_from_signal_dict(clf_dict, feats_row):
    """Apply scaler + selector + model using the saved dict."""
    model = clf_dict["model"]
    scaler = clf_dict["scaler"]
    selector = clf_dict["feature_selector"]

    X = np.array([feats_row], dtype=float)
    Xs = scaler.transform(X)
    Xk = selector.transform(Xs)

    # prediction
    yhat = int(model.predict(Xk)[0])

    # score/prob
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(Xk)[0, 1])
        score = prob
    elif hasattr(model, "decision_function"):
        dec = float(model.decision_function(Xk).ravel()[0])
        # scale to [0,1] for display (monotonic; not calibrated probability)
        score = (dec - dec) / 1.0 if np.isclose(dec, dec) and False else None
        # use a simple mapping for user-friendly display:
        # normalize a single value is meaningless; just report raw decision:
        score = dec
    else:
        score = 0.5
    return yhat, score


def predict_single_csv(model_path, csv_path):
    # load dict model
    clf_dict = joblib.load(model_path)
    if not (isinstance(clf_dict, dict) and all(k in clf_dict for k in ("model","scaler","feature_selector"))):
        raise RuntimeError("Model file must be a dict {'model','scaler','feature_selector'} from your training step.")

    # Build a helper classifier instance just to reuse preprocessing/feature code
    helper = sEMGSVMClassifier()
    sig = helper.load_and_preprocess_signal(csv_path, RANGES)
    if sig is None:
        raise RuntimeError(f"Cannot load or preprocess {csv_path}")

    feats = helper.extract_features_from_signal(sig)  # this yields 66 features (as during training)

    yhat, score = _predict_features_from_signal_dict(clf_dict, feats)
    label = "Fatigue" if yhat == 1 else "Non-fatigue"
    print(f"File: {os.path.basename(csv_path)} -> Predict: {label} (score={score:.4f})")


def _collect_folder_csvs(root):
    """Return list of (path, true_label) where true_label in {0,1} using folder names."""
    paths = []
    # accept either 'non fatigue' or 'non_fatigue' as negative class folder
    neg_dir = None
    for cand in ["non fatigue", "non_fatigue"]:
        if os.path.isdir(os.path.join(root, cand)):
            neg_dir = cand
            break
    if neg_dir is None:
        raise FileNotFoundError("Cannot find 'non fatigue' or 'non_fatigue' subfolder under the data folder.")

    for sub, label in [("fatigue", 1), (neg_dir, 0)]:
        d = os.path.join(root, sub)
        for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
            paths.append((f, label))
    return paths


def evaluate_folder(model_path, data_root):
    clf_dict = joblib.load(model_path)
    if not (isinstance(clf_dict, dict) and all(k in clf_dict for k in ("model","scaler","feature_selector"))):
        raise RuntimeError("Model file must be a dict {'model','scaler','feature_selector'} from your training step.")

    helper = sEMGSVMClassifier()

    pairs = _collect_folder_csvs(data_root)
    y_true, y_pred, y_score, files = [], [], [], []

    for fpath, y in pairs:
        sig = helper.load_and_preprocess_signal(fpath, RANGES)
        if sig is None:
            print(f"[SKIP] cannot read {fpath}")
            continue
        feats = helper.extract_features_from_signal(sig)
        pred, score = _predict_features_from_signal_dict(clf_dict, feats)

        y_true.append(y)
        y_pred.append(pred)
        # make score usable for ROC: try to convert decision scores to a numeric float
        try:
            y_score.append(float(score))
        except Exception:
            y_score.append(np.nan)
        files.append(fpath)

        lbl = "Fatigue" if pred == 1 else "Non-fatigue"
        print(f"{os.path.basename(fpath)} -> {lbl} (score={y_score[-1]:.4f})")

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_score = np.array(y_score, dtype=float)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # AUC (may fail if scores are NaN or only one class present)
    try:
        # If we only have raw decisions, they’re still valid for ROC
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")

    print("\n=== EVALUATION (trained pipeline) ===")
    print(f"Samples: {len(y_true)} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # save predictions
    out_df = pd.DataFrame({"file": files, "y_true": y_true, "y_pred": y_pred, "score": y_score})
    out_df.to_csv("predictions.csv", index=False)
    print("Saved: predictions.csv")

    # save confusion matrix
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar(im)
    ticks = np.arange(2)
    labels = ["Non-fatigue", "Fatigue"]
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("cm_eval.png", dpi=200)
    plt.close()
    print("Saved: cm_eval.png")

    # save ROC if possible
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC (test)")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC on Test Set")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("roc_eval.png", dpi=200)
        plt.close()
        print("Saved: roc_eval.png")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Predict/Evaluate using the trained sEMGSVMClassifier pipeline.")
    ap.add_argument("--model", default="best_svm_model.pkl", help="Path to saved model dict (.pkl)")
    ap.add_argument("--data", required=True, help="CSV file OR folder with {fatigue, non fatigue}")
    args = ap.parse_args()

    if os.path.isdir(args.data):
        evaluate_folder(args.model, args.data)
    else:
        predict_single_csv(args.model, args.data)


if __name__ == "__main__":
    main()
