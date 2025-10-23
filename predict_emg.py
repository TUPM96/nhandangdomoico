#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict/evaluate sEMG fatigue using a trained model (best_model.joblib).
- If --data points to a folder with {fatigue, non_fatigue} (or "non fatigue"), it evaluates and reports metrics.
- If --data points to a single CSV file, it predicts its label and probability.
Works with models saved by emg_classify_full.py (pipeline with StandardScaler + classifier).
"""
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# ---------- signal & features (must match training) ----------
def bandpass_notch_filter(x, fs, bp=(20, 450), notch=50.0, q=30.0):
    nyq = 0.5 * fs
    low = max(1.0, bp[0]) / nyq
    high = min(bp[1], nyq - 1.0) / nyq
    b, a = butter(4, [low, high], btype='bandpass')
    xf = filtfilt(b, a, x)
    w0 = notch / nyq
    if 0 < w0 < 1:
        b_notch, a_notch = iirnotch(w0, q)
        xf = filtfilt(b_notch, a_notch, xf)
    return xf

def zero_crossings(x, thresh=0.0):
    zc = 0
    for i in range(1, len(x)):
        if (x[i-1] < -thresh and x[i] > thresh) or (x[i-1] > thresh and x[i] < -thresh):
            zc += 1
    return zc

def slope_sign_changes(x, thresh=0.0):
    ssc = 0
    for i in range(1, len(x)-1):
        a = x[i] - x[i-1]
        b = x[i] - x[i+1]
        if (a * b > 0) and (abs(a) > thresh or abs(b) > thresh):
            ssc += 1
    return ssc

def waveform_length(x):
    return np.sum(np.abs(np.diff(x)))

def bandpowers_welch(x, fs, bands):
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    Pxx = np.maximum(Pxx, 1e-12)
    bp_vals = []
    for (f1, f2) in bands:
        mask = (f >= f1) & (f <= f2)
        bp_vals.append(np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0)
    return bp_vals, f, Pxx

def mnf_mdf_from_psd(f, Pxx):
    mnf = np.sum(f * Pxx) / np.sum(Pxx)
    cumsum = np.cumsum(Pxx)
    half = 0.5 * cumsum[-1]
    idx = np.searchsorted(cumsum, half)
    mdf = f[idx] if idx < len(f) else f[-1]
    return mnf, mdf

def extract_features_from_signal(x, fs):
    x = np.asarray(x).astype(float)
    rms = np.sqrt(np.mean(x ** 2))
    mav = np.mean(np.abs(x))
    wl = waveform_length(x)
    zc = zero_crossings(x, thresh=0.0)
    ssc = slope_sign_changes(x, thresh=0.0)
    vmin = np.min(x)
    vmax = np.max(x)
    xmean = np.mean(x)
    xstd = np.std(x) if np.std(x) > 0 else 1e-12
    xvar = np.var(x)
    xmad = np.mean(np.abs(x - xmean))
    xskew = skew(x, bias=False)
    xkurt = kurtosis(x, bias=False)
    bands = [(10, 30), (30, 70), (70, 100), (100, 200)]
    bp_vals, f, Pxx = bandpowers_welch(x, fs, bands)
    mnf, mdf = mnf_mdf_from_psd(f, Pxx)
    total_power = np.trapz(Pxx, f)
    spectral_centroid = np.sum(f * Pxx) / (np.sum(Pxx) + 1e-12)
    feats = [rms, mav, wl, zc, ssc, vmin, vmax, xmean, xstd, xvar, xmad, xskew, xkurt,
             mnf, mdf, spectral_centroid, total_power] + bp_vals
    return np.array(feats, dtype=float)

def _read_signal_csv(fpath):
    # robust CSV reader (pandas autodetect sep, accept headers and NaN)
    try:
        df = pd.read_csv(fpath, engine="python", sep=None)
    except Exception:
        df = pd.read_csv(fpath, engine="python", sep=",", on_bad_lines="skip")
    if "amplitudo" in df.columns:
        ser = pd.to_numeric(df["amplitudo"], errors="coerce")
    else:
        num_df = df.apply(pd.to_numeric, errors="coerce")
        valid_counts = num_df.notna().sum()
        if valid_counts.max() == 0:
            raise ValueError(f"No numeric data in {fpath}")
        best_col = valid_counts.idxmax()
        ser = num_df[best_col]
    sig = ser.dropna().to_numpy(dtype=float).squeeze()
    if sig.ndim != 1 or sig.size == 0:
        raise ValueError(f"Cannot extract 1-D signal from {fpath}")
    return sig

# ---------- evaluation helpers ----------
def build_test_matrix(root, fs=1000.0, expect_feature_vector=False):
    X, y, paths = [], [], []
    neg_dir = None
    for cand in ["non_fatigue", "non fatigue"]:
        if os.path.isdir(os.path.join(root, cand)):
            neg_dir = cand
            break
    if neg_dir is None:
        raise FileNotFoundError("Cannot find 'non_fatigue' or 'non fatigue' in test folder.")

    for label, sub in [(1, "fatigue"), (0, neg_dir)]:
        folder = os.path.join(root, sub)
        for f in sorted(glob(os.path.join(folder, "*.csv"))):
            if expect_feature_vector:
                df = pd.read_csv(f, engine="python", sep=None)
                num = df.apply(pd.to_numeric, errors="coerce").values.reshape(-1)
                vec = num[~np.isnan(num)]
                X.append(vec); y.append(label); paths.append(f)
            else:
                sig = _read_signal_csv(f)
                sig = bandpass_notch_filter(sig, fs=fs, bp=(20,450), notch=50.0, q=30.0)
                sig = sig - np.mean(sig)
                sig = sig / (np.std(sig) + 1e-12)
                if sig.shape[0] > int(2.0 * fs):
                    start = int(0.5 * fs)
                    sig = sig[start:start+int(1.0*fs)]
                X.append(extract_features_from_signal(sig, fs)); y.append(label); paths.append(f)

    # pad if feature-vectors differ
    X = np.array(X, dtype=object)
    if X.dtype == object:
        maxlen = max(len(v) for v in X)
        Xp = np.zeros((len(X), maxlen), dtype=float)
        for i, v in enumerate(X):
            v = np.asarray(v, dtype=float)
            Xp[i, :len(v)] = v if len(v) <= maxlen else v[:maxlen]
        X = Xp
    return X, np.array(y, dtype=int), paths

def save_cm(cm, labels=("Non-fatigue","Fatigue"), title="Confusion Matrix", out="cm_eval.png"):
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Predict/Evaluate sEMG fatigue with a trained model.")
    ap.add_argument("--model", default="best_model.joblib", help="Path to trained model (.joblib)")
    ap.add_argument("--data", required=True, help="Test CSV file OR folder with {fatigue, non_fatigue}")
    ap.add_argument("--fs", type=float, default=1000.0, help="Sampling rate for raw signals")
    ap.add_argument("--expect-feature-vector", action="store_true",
                    help="If set, input CSVs are treated as precomputed feature vectors")
    args = ap.parse_args()

    clf = joblib.load(args.model)

    if os.path.isdir(args.data):
        # Evaluate on folder
        X, y, paths = build_test_matrix(args.data, fs=args.fs, expect_feature_vector=args.expect_feature_vector)
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X)[:, 1]
        else:
            dec = clf.decision_function(X)
            y_score = (dec - dec.min()) / (dec.ptp() + 1e-12)
        y_pred = clf.predict(X)

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        cm = confusion_matrix(y, y_pred)
        try:
            auc = roc_auc_score(y, y_score)
        except Exception:
            auc = float("nan")

        print("=== EVALUATION ON TEST FOLDER ===")
        print(f"Samples: {len(y)} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        save_cm(cm, out="cm_eval.png")
        # Save predictions
        out_df = pd.DataFrame({
            "file": paths,
            "y_true": y,
            "y_pred": y_pred,
            "score": y_score
        })
        out_df.to_csv("predictions.csv", index=False)
        print("Saved: predictions.csv, cm_eval.png")

        # ROC curve on test
        try:
            fpr, tpr, _ = roc_curve(y, y_score)
            plt.figure()
            plt.plot(fpr, tpr, label="ROC (test)")
            plt.plot([0,1], [0,1], linestyle="--")
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

    else:
        # Predict single CSV
        f = args.data
        if args.expect_feature_vector:
            df = pd.read_csv(f, engine="python", sep=None)
            num = df.apply(pd.to_numeric, errors="coerce").values.reshape(-1)
            vec = num[~np.isnan(num)]
            X = vec.reshape(1, -1)
        else:
            sig = _read_signal_csv(f)
            sig = bandpass_notch_filter(sig, fs=args.fs, bp=(20,450), notch=50.0, q=30.0)
            sig = sig - np.mean(sig)
            sig = sig / (np.std(sig) + 1e-12)
            X = extract_features_from_signal(sig, fs=args.fs).reshape(1, -1)

        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X)[0, 1]
        else:
            dec = clf.decision_function(X).ravel()
            prob = (dec - dec.min()) / (dec.ptp() + 1e-12)
            prob = float(prob[0]) if np.ndim(prob) else float(prob)

        pred = int(clf.predict(X)[0])
        label = "Fatigue" if pred == 1 else "Non-fatigue"
        print(f"File: {os.path.basename(f)} -> Predict: {label} (prob={prob:.4f})")

if __name__ == "__main__":
    main()
