#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, time, warnings
from glob import glob
from dataclasses import dataclass, asdict
import numpy as np, pandas as pd

from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import skew, kurtosis

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import joblib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ================== Signal utils ==================
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
        if (a*b > 0) and (abs(a) > thresh or abs(b) > thresh):
            ssc += 1
    return ssc

def waveform_length(x): return np.sum(np.abs(np.diff(x)))

def bandpowers_welch(x, fs, bands):
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    Pxx = np.maximum(Pxx, 1e-12)
    bp_vals = []
    for (f1, f2) in bands:
        mask = (f >= f1) & (f <= f2)
        bp_vals.append(np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0)
    return bp_vals, f, Pxx

def mnf_mdf_from_psd(f, Pxx):
    mnf = np.sum(f*Pxx)/np.sum(Pxx)
    cs = np.cumsum(Pxx); half = 0.5*cs[-1]
    idx = np.searchsorted(cs, half)
    mdf = f[idx] if idx < len(f) else f[-1]
    return mnf, mdf

def extract_features_from_signal(x, fs):
    x = np.asarray(x).astype(float)
    # Time-domain
    rms = np.sqrt(np.mean(x**2))
    mav = np.mean(np.abs(x))
    wl  = waveform_length(x)
    zc  = zero_crossings(x, 0.0)
    ssc = slope_sign_changes(x, 0.0)
    vmin, vmax = np.min(x), np.max(x)
    xmean, xstd = np.mean(x), (np.std(x) if np.std(x) > 0 else 1e-12)
    xvar, xmad = np.var(x), np.mean(np.abs(x - xmean))
    xskew, xkurt = skew(x, bias=False), kurtosis(x, bias=False)
    # Frequency-domain
    bands = [(10,30),(30,70),(70,100),(100,200)]
    bp_vals, f, Pxx = bandpowers_welch(x, fs, bands)
    mnf, mdf = mnf_mdf_from_psd(f, Pxx)
    total_power = np.trapz(Pxx, f)
    spectral_centroid = np.sum(f*Pxx)/(np.sum(Pxx)+1e-12)

    feats = [rms,mav,wl,zc,ssc,vmin,vmax,xmean,xstd,xvar,xmad,xskew,xkurt,
             mnf,mdf,spectral_centroid,total_power] + bp_vals
    feat_names = ["RMS","MAV","WL","ZC","SSC","MIN","MAX","MEAN","STD","VAR","MAD","SKEW","KURT",
                  "MNF","MDF","SPEC_CENTROID","TOTAL_POWER"] + [f"BAND_{a}_{b}" for a,b in bands]
    return np.array(feats,float), feat_names

# ================== Robust CSV ==================
def _read_signal_csv(fpath):
    try: df = pd.read_csv(fpath, engine="python", sep=None)
    except Exception: df = pd.read_csv(fpath, engine="python", sep=",", on_bad_lines="skip")
    if "amplitudo" in df.columns:
        ser = pd.to_numeric(df["amplitudo"], errors="coerce")
    else:
        num_df = df.apply(pd.to_numeric, errors="coerce")
        valid = num_df.notna().sum()
        if valid.max() == 0: raise ValueError(f"No numeric data in {fpath}.")
        ser = num_df[valid.idxmax()]
    sig = ser.dropna().to_numpy(dtype=float).squeeze()
    if sig.ndim != 1 or sig.size == 0: raise ValueError(f"Cannot extract 1-D signal from {fpath}.")
    return sig

def load_dataset(root, fs=1000.0, expect_feature_vector=False):
    X,y,files = [],[],[]
    neg_dir = None
    for candidate in ["non_fatigue","non fatigue"]:
        if os.path.isdir(os.path.join(root,candidate)):
            neg_dir = candidate; break
    if neg_dir is None: raise FileNotFoundError("Cannot find 'non_fatigue' or 'non fatigue'.")

    paths = [(1, os.path.join(root,"fatigue")), (0, os.path.join(root,neg_dir))]
    feat_names_ref, skipped = None, []

    for label, p in paths:
        for f in sorted(glob(os.path.join(p,"*.csv"))):
            try:
                if expect_feature_vector:
                    df = pd.read_csv(f, engine="python", sep=None)
                    num = df.apply(pd.to_numeric, errors="coerce").values.reshape(-1)
                    vec = num[~np.isnan(num)]
                    if vec.size == 0: raise ValueError("Empty feature vector after cleaning.")
                    X.append(vec); y.append(label); files.append(f); continue
                sig = _read_signal_csv(f)
                sig = np.nan_to_num(sig)
                sig = bandpass_notch_filter(sig, fs, (20,450), 50.0, 30.0)
                sig = (sig - np.mean(sig)) / (np.std(sig)+1e-12)
                if sig.shape[0] > int(2.0*fs):
                    start = int(0.5*fs); end = start+int(1.0*fs)
                    sig = sig[start:end]
                feats, feat_names = extract_features_from_signal(sig, fs)
                if feat_names_ref is None: feat_names_ref = feat_names
                X.append(feats); y.append(label); files.append(f)
            except Exception as e:
                skipped.append((f,str(e))); continue

    if not X: raise RuntimeError("No valid CSV loaded.")

    X = np.array(X, dtype=object)
    if X.dtype == object:
        maxlen = max(len(v) for v in X)
        Xp = np.zeros((len(X), maxlen), float)
        for i,v in enumerate(X):
            v = np.asarray(v,float)
            Xp[i,:len(v)] = v if len(v)<=maxlen else v[:maxlen]
        X = Xp

    y = np.array(y,int)
    cols = feat_names_ref if feat_names_ref else [f"f{i}" for i in range(X.shape[1])]
    dfX = pd.DataFrame(X, columns=cols); dfX["label"]=y; dfX["file"]=files

    if skipped:
        print("Warning: skipped files:")
        for f,msg in skipped[:10]: print(" -", os.path.basename(f), "->", msg)
        if len(skipped)>10: print(f" ... and {len(skipped)-10} more.")

    return X,y,dfX

# ================== ML utils ==================
@dataclass
class RunResult:
    run:int; model:str; accuracy:float; precision:float; recall:float; f1:float; specificity:float; auc:float; train_time_s:float

def compute_metrics(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    auc = roc_auc_score(y_true, y_score) if y_score is not None else np.nan
    return acc,prec,rec,f1,spec,auc

def build_models(use_grid, force_model=""):
    """Chỉ build các model theo force_model (nếu chỉ định)."""
    models = {}
    # SVM RBF
    if force_model in ["", "SVM_RBF"]:
        svm = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))])
        models["SVM_RBF"] = GridSearchCV(
            svm, {"clf__C":[0.1,1,10,100], "clf__gamma":["scale", 0.1, 0.01, 0.001]},
            cv=5, n_jobs=-1, scoring="f1"
        ) if use_grid else svm
    # KNN
    if force_model in ["", "KNN"]:
        knn = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5, weights="uniform"))])
        models["KNN"] = GridSearchCV(
            knn, {"clf__n_neighbors":[3,5,7,9], "clf__weights":["uniform","distance"]},
            cv=5, n_jobs=-1, scoring="f1"
        ) if use_grid else knn
    # LDA
    if force_model in ["", "LDA"]:
        lda = Pipeline([("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis(solver="svd"))])
        models["LDA"] = GridSearchCV(
            lda, {"clf__solver":["svd","lsqr"], "clf__shrinkage":[None,"auto"]},
            cv=5, n_jobs=-1, scoring="f1"
        ) if use_grid else lda
    return models

def train_and_evaluate(X, y, test_size, seed, use_grid, force_model=""):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    models = build_models(use_grid, force_model)
    results = []
    best = (None, -1.0, None, (None, None, None))  # name, f1, estimator, (y_true, y_pred, y_score)

    for name, model in models.items():
        t0 = time.time(); model.fit(Xtr, ytr); train_time = time.time()-t0
        est = model.best_estimator_ if isinstance(model, GridSearchCV) else model
        y_pred = est.predict(Xte)
        if hasattr(est,"predict_proba"): y_score = est.predict_proba(Xte)[:,1]
        else:
            dec = est.decision_function(Xte)
            y_score = (dec - dec.min())/(dec.ptp()+1e-12)
        acc,prec,rec,f1,spec,auc = compute_metrics(yte, y_pred, y_score)
        results.append(RunResult(0, name, acc,prec,rec,f1,spec,auc, train_time))
        if f1 > best[1]: best = (name, f1, est, (yte, y_pred, y_score))
    return results, best

def plot_roc_confusion(y_true, y_score, y_pred, out_prefix="best_model"):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(); plt.plot(fpr, tpr, label="ROC"); plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
    plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(out_prefix + "_ROC.png", dpi=200); plt.close()
    # CM
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(); im = plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix"); plt.colorbar(im)
    ticks = np.arange(2); plt.xticks(ticks, ["Non-fatigue","Fatigue"], rotation=45); plt.yticks(ticks, ["Non-fatigue","Fatigue"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    plt.ylabel("True label"); plt.xlabel("Predicted label"); plt.tight_layout(); plt.savefig(out_prefix + "_CM.png", dpi=200); plt.close()

# --------- SVM-only charts & comparisons ---------
def plot_svm_only(y_true, y_pred, y_score, prefix="SVM"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(); plt.plot(fpr, tpr, label="SVM ROC"); plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("SVM ROC Curve")
    plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(f"{prefix}_ROC_Curve.png", dpi=200); plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(); im = plt.imshow(cm, interpolation="nearest"); plt.title("SVM Confusion Matrix"); plt.colorbar(im)
    ticks = np.arange(2); plt.xticks(ticks, ["Non-fatigue","Fatigue"], rotation=45); plt.yticks(ticks, ["Non-fatigue","Fatigue"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    plt.ylabel("True label"); plt.xlabel("Predicted label"); plt.tight_layout(); plt.savefig(f"{prefix}_Confusion_Matrix.png", dpi=200); plt.close()

def compare_svm_kernels(Xtr, ytr, Xte, yte, use_grid=False):
    kernels = ["linear","rbf","poly","sigmoid"]; results={}
    for k in kernels:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel=k, probability=True))])
        if use_grid:
            if k=="linear": params={"clf__C":[0.1,1,10]}
            elif k=="rbf": params={"clf__C":[0.1,1,10], "clf__gamma":["scale",0.01,0.001]}
            elif k=="poly": params={"clf__C":[0.1,1,10],"clf__degree":[2,3],"clf__gamma":["scale",0.01]}
            else: params={"clf__C":[0.1,1,10],"clf__gamma":["scale",0.01]}
            model = GridSearchCV(pipe, params, cv=5, n_jobs=-1, scoring="accuracy")
        else:
            model = pipe
        model.fit(Xtr,ytr); est = model.best_estimator_ if isinstance(model,GridSearchCV) else model
        acc = accuracy_score(yte, est.predict(Xte)); results[k]=acc
    plt.figure(); labels=list(results); vals=[results[k] for k in labels]; x=np.arange(len(labels))
    plt.bar(x,vals); plt.xticks(x,labels); plt.ylabel("Accuracy"); plt.title("SVM Kernel Comparison")
    plt.tight_layout(); plt.savefig("SVM_Kernel_Comparison.png", dpi=200); plt.close()
    return results

def plot_roc_3models(estimators, Xte, yte, out_file="ROC_3Models.png"):
    plt.figure()
    for name, est in estimators.items():
        if hasattr(est,"predict_proba"): score = est.predict_proba(Xte)[:,1]
        else:
            dec = est.decision_function(Xte)
            score = (dec - dec.min())/(dec.ptp()+1e-12)
        fpr,tpr,_ = roc_curve(yte, score); plt.plot(fpr,tpr,label=name)
    plt.plot([0,1],[0,1],'--'); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison: SVM vs KNN vs LDA"); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out_file, dpi=200); plt.close()

# ================== Main ==================
def main():
    ap = argparse.ArgumentParser(description="sEMG fatigue classification with SVM/KNN/LDA")
    ap.add_argument("--force-model", default="", choices=["SVM_RBF", "KNN", "LDA"],
                    help="Chỉ huấn luyện & đánh giá mô hình chỉ định")
    ap.add_argument("--data", default="dataset", help="Dataset root folder")
    ap.add_argument("--fs", type=float, default=1000.0, help="Sampling rate")
    ap.add_argument("--test-size", type=float, default=0.3, help="Test split ratio")
    ap.add_argument("--runs", type=int, default=10, help="Repeat runs")
    ap.add_argument("--seed", type=int, default=42, help="Random seed base")
    ap.add_argument("--grid", action="store_true", help="Use GridSearchCV")
    ap.add_argument("--feature-csv", default="", help="Export features CSV")
    ap.add_argument("--expect-feature-vector", action="store_true", help="Treat CSVs as feature vectors")
    ap.add_argument("--out-results", default="results.csv", help="Per-run results CSV")
    ap.add_argument("--out-summary", default="summary.csv", help="Summary CSV")
    ap.add_argument("--out-tex", default="summary.tex", help="LaTeX table")
    ap.add_argument("--out-model", default="best_model.joblib", help="Saved model path")
    args = ap.parse_args()

    # Load data
    X,y,dfX = load_dataset(args.data, fs=args.fs, expect_feature_vector=args.expect_feature_vector)
    if args.feature_csv: dfX.to_csv(args.feature_csv, index=False)

    # Training runs
    all_rows=[]; best_global=(None,-1.0,None,(None,None,None))
    for r in range(args.runs):
        results, best = train_and_evaluate(
            X, y, test_size=args.test_size, seed=args.seed + r, use_grid=args.grid, force_model=args.force_model
        )
        for rr in results:
            row = asdict(rr); row["run"]=r; all_rows.append(row)
        best_global = best

    # Save results + summary
    df = pd.DataFrame(all_rows); df.to_csv(args.out_results, index=False)
    agg = df.groupby("model").agg(
        accuracy_mean=("accuracy","mean"), accuracy_std=("accuracy","std"),
        precision_mean=("precision","mean"), recall_mean=("recall","mean"),
        f1_mean=("f1","mean"), f1_std=("f1","std"),
        specificity_mean=("specificity","mean"), auc_mean=("auc","mean"),
        train_time_mean=("train_time_s","mean")
    ).reset_index()
    agg.to_csv(args.out_summary, index=False)
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write(agg.round(4).to_latex(index=False,
                                      caption="Tổng hợp hiệu năng các mô hình (mean ± std)",
                                      label="tab:summary_models"))
    print(f"Saved results to: {args.out_results}")
    print(f"Saved summary to: {args.out_summary}")
    print(f"LaTeX table saved to: {args.out_tex}")

    # ---------- Comparison charts ----------
    def _plot_metric_bar(agg_df, metric, std_metric=None, title="", ylabel="", outfile="chart.png"):
        labels = agg_df["model"].tolist(); vals = agg_df[metric].values
        errs = agg_df[std_metric].values if std_metric and std_metric in agg_df.columns else None
        plt.figure(); x = range(len(labels))
        if errs is not None: plt.bar(x, vals, yerr=errs, capsize=5)
        else: plt.bar(x, vals)
        plt.xticks(list(x), labels); plt.ylabel(ylabel or metric); plt.title(title or metric)
        plt.tight_layout(); plt.savefig(outfile, dpi=200); plt.close()

    def _plot_metric_box(results_df, metric, title="", ylabel="", outfile="box.png"):
        groups, labels = [], []
        for name, sub in results_df.groupby("model"):
            groups.append(sub[metric].dropna().values); labels.append(name)
        plt.figure(); plt.boxplot(groups, labels=labels, showmeans=True)
        plt.ylabel(ylabel or metric); plt.title(title or metric)
        plt.tight_layout(); plt.savefig(outfile, dpi=200); plt.close()

    _plot_metric_bar(agg,"accuracy_mean","accuracy_std","So sánh Accuracy (mean±std)","Accuracy","cmp_accuracy_bar.png")
    _plot_metric_bar(agg,"f1_mean","f1_std","So sánh F1-score (mean±std)","F1-score","cmp_f1_bar.png")
    _plot_metric_bar(agg,"auc_mean",None,"So sánh AUC (mean)","AUC","cmp_auc_bar.png")
    _plot_metric_bar(agg,"train_time_mean",None,"So sánh thời gian huấn luyện (s)","Seconds","cmp_train_time_bar.png")
    _plot_metric_box(df,"f1","Phân phối F1-score qua các lần chạy","F1-score","cmp_f1_box.png")
    _plot_metric_box(df,"auc","Phân phối AUC qua các lần chạy","AUC","cmp_auc_box.png")
    print("Saved charts: cmp_accuracy_bar.png, cmp_f1_bar.png, cmp_auc_bar.png, cmp_train_time_bar.png, cmp_f1_box.png, cmp_auc_box.png")

    # ---------- SVM-specific & always export best_SVM_RBF_* ----------
    # Dùng split cố định theo last_seed để vẽ SVM-only & ROC_3Models (nếu cần)
    last_seed = args.seed + args.runs - 1
    Xtr_svm, Xte_svm, ytr_svm, yte_svm = train_test_split(
        X, y, test_size=args.test_size, random_state=last_seed, stratify=y
    )

    # Huấn luyện SVM_RBF (luôn) để sinh best_SVM_RBF_* bất kể mô hình thắng là gì
    if args.grid:
        svm_model = GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))]),
            {"clf__C":[0.1,1,10,100], "clf__gamma":["scale", 0.1, 0.01, 0.001]},
            cv=5, n_jobs=-1, scoring="f1"
        )
    else:
        svm_model = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))])

    svm_model.fit(Xtr_svm, ytr_svm)
    svm_best = svm_model.best_estimator_ if isinstance(svm_model, GridSearchCV) else svm_model
    y_pred_svm = svm_best.predict(Xte_svm)
    y_score_svm = svm_best.predict_proba(Xte_svm)[:, 1]

    # Lưu SVM-only & best_SVM_RBF_*
    plot_svm_only(yte_svm, y_pred_svm, y_score_svm, prefix="SVM")
    print("Saved SVM_Confusion_Matrix.png, SVM_ROC_Curve.png")
    compare_svm_kernels(Xtr_svm, ytr_svm, Xte_svm, yte_svm, use_grid=args.grid)
    print("Saved SVM_Kernel_Comparison.png")
    plot_roc_confusion(yte_svm, y_score_svm, y_pred_svm, out_prefix="best_SVM_RBF")
    print("Saved best_SVM_RBF_ROC.png, best_SVM_RBF_CM.png")

    # ---------- Save best model ----------
    best_name, _, best_estimator, (y_true, y_pred, y_score) = best_global
    # Nếu ép model, set tên hiển thị theo force-model; model lưu vẫn là best_estimator tìm được trong các run
    if args.force_model:
        best_name = args.force_model
    if best_estimator is not None:
        joblib.dump(best_estimator, args.out_model)
        print(f"Saved best model ({best_name}) -> {args.out_model}")
        if y_true is not None and y_score is not None:
            plot_roc_confusion(y_true, y_score, y_pred, out_prefix=f"best_{best_name}")
            print("Saved ROC and confusion matrix plots for best model.")

    # ---------- ROC 3 models (chỉ vẽ khi không ép model) ----------
    if not args.force_model:
        estims = {"SVM_RBF": svm_best}
        # KNN
        knn = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5, weights="uniform"))])
        if args.grid:
            knn = GridSearchCV(knn, {"clf__n_neighbors":[3,5,7,9], "clf__weights":["uniform","distance"]}, cv=5, n_jobs=-1, scoring="f1")
        knn.fit(Xtr_svm, ytr_svm)
        estims["KNN"] = knn.best_estimator_ if isinstance(knn, GridSearchCV) else knn
        # LDA
        lda = Pipeline([("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis(solver="svd"))])
        if args.grid:
            lda = GridSearchCV(lda, {"clf__solver":["svd","lsqr"], "clf__shrinkage":[None,"auto"]}, cv=5, n_jobs=-1, scoring="f1")
        lda.fit(Xtr_svm, ytr_svm)
        estims["LDA"] = lda.best_estimator_ if isinstance(lda, GridSearchCV) else lda
        plot_roc_3models(estims, Xte_svm, yte_svm, out_file="ROC_3Models.png")
        print("Saved ROC_3Models.png")

if __name__ == "__main__":
    main()
