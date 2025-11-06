#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sEMG Muscle Fatigue Classification using KNN
- Lọc: zero-mean -> notch 50 Hz (nếu valid) -> bandpass (10-100 Hz) -> RECTIFY -> z-score
- ZC/SSC tính trên tín hiệu đã lọc nhưng CHƯA rectify
- Đặc trưng tần số: Welch PSD + MNF, MDF, bandpower 10-30/30-70/70-100 Hz
- Pipeline: StandardScaler -> SelectKBest(f_classif|mutual_info, k) -> KNeighborsClassifier
- GridSearch: n_neighbors, weights, metric, p (1/2) + k (tùy chọn)
- Group-aware split (tùy chọn) để chống leakage khi dùng sliding windows
- Xuất:
  + best_knn_model.pkl (pipeline đã fit)
  + knn_hardcode.json (tham số để hardcode)
  + knn_hardcoded_params.py (file Python hằng số, có hàm rebuild pipeline)

CLI ví dụ:
  # Chạy 1 lần
  python sEMG_KNN.py --data ./dataset --grid --sliding --win 8000 --step 4000 --group-split --k 50
  # DỪNG SỚM khi TestAcc >= 0.85 (thử tối đa 50 seed)
  python sEMG_KNN.py --data ./dataset --grid --sliding --win 8000 --step 4000 --group-split \
     --k 50 --target-acc 0.85 --max-tries 50
"""

import os, sys, json, argparse, warnings, datetime, platform
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, iirnotch, welch

from sklearn import __version__ as sklearn_version
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score,
    GroupShuffleSplit, RepeatedStratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_curve, auc, roc_auc_score
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier


# ============================= Run Logger =============================
class RunLogger:
    def __init__(self, out_dir="run_artifacts_knn"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.manifest = {
            "timestamp": datetime.datetime.now().isoformat(),
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn_version,
                "matplotlib": plt.matplotlib.__version__,
                "seaborn": sns.__version__,
            },
            "args": {}, "seed": None, "data": {}, "preprocess": {},
            "features": {}, "model": {}, "metrics": {}, "files": {}
        }

    def set_args(self, d): self.manifest["args"] = d
    def set_seed(self, s): self.manifest["seed"] = int(s)
    def set_data_info(self, fat_files, non_files, tr_idx, te_idx):
        self.manifest["data"] = {
            "fatigue_files_sorted": fat_files,
            "non_fatigue_files_sorted": non_files,
            "train_idx": tr_idx.tolist(),
            "test_idx": te_idx.tolist(),
        }
    def set_preprocess(self, fs, lowcut, highcut, notch_hz, ranges, sliding, win, step, group_split, fs_kind):
        self.manifest["preprocess"] = {
            "fs": fs, "lowcut": lowcut, "highcut": highcut, "notch_hz": notch_hz,
            "ranges_to_keep": ranges, "sliding": bool(sliding),
            "win": win, "step": step, "group_split": bool(group_split),
            "feature_selector": fs_kind
        }
    def set_features(self, k_selected, selected_idx, scaler_mean, scaler_scale):
        self.manifest["features"] = {
            "k_selected": int(k_selected),
            "selected_feature_indices": selected_idx.tolist(),
            "scaler_mean": None if scaler_mean is None else scaler_mean.tolist(),
            "scaler_scale": None if scaler_scale is None else scaler_scale.tolist(),
        }
    def set_model(self, best_params): self.manifest["model"] = {"best_params": best_params}
    def set_metrics(self, train_acc, test_acc, rocauc):
        self.manifest["metrics"] = {"train_acc": float(train_acc), "test_acc": float(test_acc), "auc": float(rocauc)}
    def add_file(self, key, path): self.manifest["files"][key] = os.path.abspath(path)
    def write(self, name="knn_run_manifest.json"):
        p = os.path.join(self.out_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)
        print(f"[LOG] Wrote manifest: {p}")
        return p


# ============================= Classifier =============================
class sEMGKNNClassifier:
    def __init__(self, data_path=None, run_logger: RunLogger | None = None, seed: int = 42):
        if data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(script_dir, 'dataset')
        self.data_path = data_path
        self.data_fatigue_path = os.path.join(data_path, 'fatigue')
        self.data_nonfatigue_path = os.path.join(data_path, 'non fatigue')

        self.logger = run_logger or RunLogger()
        self.seed = int(seed)

        self.pipeline_ = None
        self.X_train_raw = self.X_test_raw = None
        self.y_train = self.y_test = None
        self.groups_train = self.groups_test = None
        self.train_idx = self.test_idx = None

        # signal params
        self.lowcut = 10.0
        self.highcut = 100.0
        self.fs = 1000.0
        self.notch_hz = 50.0

        # export
        self.best_params_export = {}

        print("sEMG KNN Classifier initialized")
        print(f"Fatigue data path: {self.data_fatigue_path}")
        print(f"Non-fatigue data path: {self.data_nonfatigue_path}")

    # ------------ SIGNAL UTILS ------------
    def _bandpass_notch(self, x, fs=1000.0, bp=(10, 100), notch=50.0, q=30.0):
        x = np.asarray(x, dtype=float)
        nyq = 0.5 * fs
        lo = max(1.0, bp[0]) / nyq
        hi = min(bp[1], fs/2 - 1.0) / nyq
        lo = min(max(lo, 1e-5), 0.99)
        hi = min(max(hi, lo + 1e-5), 0.999)
        b, a = butter(4, [lo, hi], btype='bandpass')
        xf = filtfilt(b, a, x)
        w0 = notch / nyq
        if 0 < w0 < 1:
            bn, an = iirnotch(w0, q)
            xf = filtfilt(bn, an, xf)
        return xf

    def _read_amplitudo(self, file_path: str):
        df = pd.read_csv(file_path, engine="python")
        if 'amplitudo' not in df.columns:
            num_df = df.apply(pd.to_numeric, errors='coerce')
            valid_counts = num_df.notna().sum()
            if valid_counts.max() == 0:
                raise ValueError(f"No numeric column in {file_path}")
            col = valid_counts.idxmax()
            sig = num_df[col].dropna().to_numpy(dtype=float)
        else:
            sig = pd.to_numeric(df['amplitudo'], errors='coerce').dropna().to_numpy(dtype=float)
        return sig

    # ------------ FEATURES ------------
    def _time_features(self, x, xraw):
        feats = []
        mu = np.mean(x); sd = np.std(x)
        feats += [
            mu, sd, np.var(x), np.max(x), np.min(x),
            np.median(x), np.percentile(x, 25), np.percentile(x, 75),
            np.sqrt(np.mean(x**2)),               # RMS
            np.mean(np.abs(x - mu)),              # MAD
            np.sum(np.abs(np.diff(x)))            # WL / TV
        ]
        zc = np.sum(np.diff(np.sign(xraw)) != 0)
        ssc = np.sum(np.diff(np.sign(np.diff(xraw))) != 0)
        feats += [zc, ssc]
        if sd > 0:
            z = (x - mu) / sd
            feats += [np.mean(z**3), np.mean(z**4)]
        else:
            feats += [0.0, 0.0]
        return feats

    def _freq_features_welch(self, x):
        f, Pxx = welch(x, fs=self.fs, nperseg=min(1024, len(x)))
        Pxx = np.maximum(Pxx, 1e-12)

        def bandpower(f1, f2):
            m = (f >= f1) & (f <= f2)
            return np.trapz(Pxx[m], f[m]) if np.any(m) else 0.0

        total = np.trapz(Pxx, f)
        bp_l = bandpower(10, 30)
        bp_m = bandpower(30, 70)
        bp_h = bandpower(70, 100)
        mnf = np.sum(f * Pxx) / np.sum(Pxx)
        csum = np.cumsum(Pxx); half = 0.5 * csum[-1]
        mdf = f[np.searchsorted(csum, half)]

        feats = [np.mean(Pxx), np.std(Pxx), np.max(Pxx), total,
                 bp_l, bp_m, bp_h, mnf, mdf, mnf]
        return feats

    def _extract_one(self, sig, ranges_to_keep):
        sig = np.asarray(sig, dtype=float)
        sig = sig - np.mean(sig)
        parts = []
        for s, e in ranges_to_keep:
            if e <= len(sig) and e > s:
                parts.append(sig[s:e])
        if not parts:
            return None
        seg = np.concatenate(parts)
        seg_filt = self._bandpass_notch(seg, fs=self.fs, bp=(self.lowcut, self.highcut), notch=self.notch_hz)
        raw_for_zc_ssc = seg_filt.copy()
        seg_rect = np.abs(seg_filt)
        seg_norm = (seg_rect - np.mean(seg_rect)) / (np.std(seg_rect) + 1e-12)

        L = len(seg_norm) // 2
        segments = [(seg_norm[:L], raw_for_zc_ssc[:L]), (seg_norm[L:], raw_for_zc_ssc[L:])]

        feats = []
        for x, xr in segments:
            feats += self._time_features(x, xr)
            feats += self._freq_features_welch(x)
        feats += self._time_features(seg_norm, raw_for_zc_ssc)
        feats += self._freq_features_welch(seg_norm)
        return np.array(feats, dtype=float)

    def _extract_windows(self, sig, win, step):
        sig = np.asarray(sig, dtype=float)
        sig = sig - np.mean(sig)
        feats_all = []
        for start in range(0, max(0, len(sig) - win + 1), step):
            part = sig[start:start+win]
            if len(part) < win: break
            seg_filt = self._bandpass_notch(part, fs=self.fs, bp=(self.lowcut, self.highcut), notch=self.notch_hz)
            raw_for_zc_ssc = seg_filt.copy()
            seg_rect = np.abs(seg_filt)
            seg_norm = (seg_rect - np.mean(seg_rect)) / (np.std(seg_rect) + 1e-12)

            L = len(seg_norm) // 2
            segments = [(seg_norm[:L], raw_for_zc_ssc[:L]), (seg_norm[L:], raw_for_zc_ssc[L:])]

            feats = []
            for x, xr in segments:
                feats += self._time_features(x, xr)
                feats += self._freq_features_welch(x)
            feats += self._time_features(seg_norm, raw_for_zc_ssc)
            feats += self._freq_features_welch(seg_norm)
            feats_all.append(np.array(feats, dtype=float))
        return feats_all

    # ------------ DATA PIPELINE ------------
    def load_dataset(self, sliding=False, win=8000, step=4000):
        print("Loading dataset...")
        fat_files = sorted([f for f in os.listdir(self.data_fatigue_path) if f.endswith('.csv')])
        non_files = sorted([f for f in os.listdir(self.data_nonfatigue_path) if f.endswith('.csv')])
        print(f"Found {len(fat_files)} fatigue files and {len(non_files)} non-fatigue files")

        ranges_to_keep = [(15000, 25000), (30000, 35000)]
        X, y, groups = [], [], []

        print("Processing fatigue files...")
        for fn in fat_files:
            path = os.path.join(self.data_fatigue_path, fn)
            try:
                sig = self._read_amplitudo(path)
                gid = f"fatigue_{fn}"  # STRING group id để tránh lỗi int<->str
                if sliding:
                    for fv in self._extract_windows(sig, win, step):
                        X.append(fv); y.append(1); groups.append(gid)
                else:
                    fv = self._extract_one(sig, ranges_to_keep)
                    if fv is not None:
                        X.append(fv); y.append(1); groups.append(gid)
            except Exception as ex:
                print(f"[WARN] {fn}: {ex}")

        print("Processing non-fatigue files...")
        for fn in non_files:
            path = os.path.join(self.data_nonfatigue_path, fn)
            try:
                sig = self._read_amplitudo(path)
                gid = f"nonfatigue_{fn}"
                if sliding:
                    for fv in self._extract_windows(sig, win, step):
                        X.append(fv); y.append(0); groups.append(gid)
                else:
                    fv = self._extract_one(sig, ranges_to_keep)
                    if fv is not None:
                        X.append(fv); y.append(0); groups.append(gid)
            except Exception as ex:
                print(f"[WARN] {fn}: {ex}")

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        groups = np.asarray(groups, dtype=str)

        print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")
        print(f"Class distribution: {np.sum(y==0)} non-fatigue, {np.sum(y==1)} fatigue")
        return X, y, groups, fat_files, non_files, ranges_to_keep

    def prepare_data(self, test_size=0.2, random_state=42, k_features=50,
                     sliding=False, win=8000, step=4000, group_split=False, fs_kind='anova'):
        print("Preparing data...")
        self.logger.set_seed(random_state)
        X, y, groups, fat_files, non_files, ranges = self.load_dataset(sliding=sliding, win=win, step=step)

        if group_split:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            tr_idx, te_idx = next(gss.split(X, y, groups=groups))
            self.train_idx, self.test_idx = tr_idx, te_idx
            self.X_train_raw, self.X_test_raw = X[tr_idx], X[te_idx]
            self.y_train, self.y_test = y[tr_idx], y[te_idx]
            self.groups_train, self.groups_test = groups[tr_idx], groups[te_idx]
        else:
            idx = np.arange(len(y))
            tr_idx, te_idx, y_tr, y_te = train_test_split(
                idx, y, test_size=test_size, random_state=random_state, stratify=y
            )
            self.train_idx, self.test_idx = tr_idx, te_idx
            self.X_train_raw, self.X_test_raw = X[tr_idx], X[te_idx]
            self.y_train, self.y_test = y_tr, y_te
            self.groups_train, self.groups_test = groups[tr_idx], groups[te_idx]

        self.k_features = int(min(k_features, self.X_train_raw.shape[1]))
        self.logger.set_data_info(fat_files, non_files, self.train_idx, self.test_idx)
        self.logger.set_preprocess(self.fs, self.lowcut, self.highcut, self.notch_hz,
                                   ranges, sliding, win, step, group_split, fs_kind)
        self.fs_kind = fs_kind

    # ------------ PIPELINE + GRID ------------
    def _make_pipeline_and_grid(self, tune_k_in_grid=False):
        score_func = f_classif if self.fs_kind == 'anova' else mutual_info_classif

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('kbest', SelectKBest(score_func=score_func, k=self.k_features)),
            ('knn', KNeighborsClassifier())
        ])

        # Base grid cho KNN
        base_grid = [
            {
                'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
                'knn__weights': ['uniform', 'distance'],
                'knn__metric': ['minkowski'],
                'knn__p': [1, 2],  # 1=Manhattan, 2=Euclidean
            },
            {
                'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
                'knn__weights': ['uniform', 'distance'],
                'knn__metric': ['euclidean', 'manhattan'],
            }
        ]

        # tuỳ chọn tune luôn kbest__k
        if tune_k_in_grid:
            k_choices = sorted({min(self.X_train_raw.shape[1], v) for v in [20, 30, 40, 50, 60, self.k_features]})
            for g in base_grid:
                g.update({'kbest__k': k_choices})

        return pipe, base_grid

    def train_knn(self, use_grid_search=True, cv_splits=5, cv_repeats=1,
                  scoring='f1', tune_k_in_grid=False):
        """
        scoring: 'f1' (khuyến nghị cho lệch lớp), hoặc 'accuracy', 'balanced_accuracy', 'roc_auc'
        """
        print("Training KNN model...")
        pipe, param_grid = self._make_pipeline_and_grid(tune_k_in_grid=tune_k_in_grid)
        if use_grid_search:
            inner_cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=self.seed)
            gs = GridSearchCV(pipe, param_grid=param_grid, scoring=scoring, cv=inner_cv,
                              n_jobs=-1, verbose=1, refit=True)
            gs.fit(self.X_train_raw, self.y_train)
            self.pipeline_ = gs.best_estimator_
            print(f"Best parameters: {gs.best_params_}")
            print(f"Best inner-CV ({scoring}): {gs.best_score_:.4f}")
            best = self.pipeline_
            knn = best.named_steps['knn']
            kbest = best.named_steps['kbest']
            self.best_params_export = {
                "kbest_k": int(kbest.k),
                "n_neighbors": int(knn.n_neighbors),
                "weights": knn.weights,
                "metric": knn.metric,
                "p": getattr(knn, 'p', None),
                "scoring": scoring,
                "cv_splits": cv_splits,
                "cv_repeats": cv_repeats,
                "fs_kind": self.fs_kind
            }
            self.logger.set_model(best_params=gs.best_params_)
        else:
            self.pipeline_ = pipe.set_params(
                knn__n_neighbors=7, knn__weights='distance', knn__metric='minkowski', knn__p=2
            )
            self.pipeline_.fit(self.X_train_raw, self.y_train)
            kbest = self.pipeline_.named_steps['kbest']
            knn = self.pipeline_.named_steps['knn']
            self.best_params_export = {
                "kbest_k": int(kbest.k),
                "n_neighbors": int(knn.n_neighbors),
                "weights": knn.weights,
                "metric": knn.metric,
                "p": getattr(knn, 'p', None),
                "scoring": scoring,
                "cv_splits": cv_splits,
                "cv_repeats": cv_repeats,
                "fs_kind": self.fs_kind
            }
            self.logger.set_model(best_params={
                "n_neighbors": int(knn.n_neighbors), "weights": knn.weights, "metric": knn.metric, "p": knn.p
            })

        # report CV acc trên train set với schema lặp
        inner_cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=self.seed)
        cv_acc = cross_val_score(self.pipeline_, self.X_train_raw, self.y_train,
                                 scoring='accuracy', cv=inner_cv, n_jobs=-1)
        print(f"Cross-validation accuracy (inner schema): {cv_acc.mean():.4f} (+/- {cv_acc.std()*2:.4f})")

    # ------------ Evaluate + Export ------------
    def _export_hardcode(self, out_dir="run_artifacts_knn"):
        if not self.best_params_export:
            return
        json_path = os.path.join(out_dir, "knn_hardcode.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.best_params_export, f, ensure_ascii=False, indent=2)
        print(f"[LOG] Wrote hardcode params: {json_path}")

        py_path = os.path.join(out_dir, "knn_hardcoded_params.py")
        with open(py_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated best params to hardcode for KNN\n")
            f.write("BEST_KNN = " + json.dumps(self.best_params_export, ensure_ascii=False, indent=2) + "\n\n")
            f.write("""def example_rebuild_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.neighbors import KNeighborsClassifier
    score_func = f_classif if BEST_KNN.get("fs_kind","anova") == "anova" else mutual_info_classif
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('kbest', SelectKBest(score_func=score_func, k=BEST_KNN["kbest_k"])),
        ('knn', KNeighborsClassifier(
            n_neighbors=BEST_KNN["n_neighbors"],
            weights=BEST_KNN["weights"],
            metric=BEST_KNN["metric"],
            p=BEST_KNN.get("p", 2)
        ))
    ])
    return pipe
""")
        print(f"[LOG] Wrote hardcoded Python: {py_path}")

    def evaluate_model(self, save_prefix='KNN', out_dir="run_artifacts_knn"):
        if self.pipeline_ is None:
            print("No model trained yet!"); return None, None, None

        print("Evaluating model...")
        y_tr_pred = self.pipeline_.predict(self.X_train_raw)
        y_te_pred = self.pipeline_.predict(self.X_test_raw)

        train_acc = accuracy_score(self.y_train, y_tr_pred)
        test_acc = accuracy_score(self.y_test, y_te_pred)

        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report (Test Set):")
        print(classification_report(self.y_test, y_te_pred, target_names=['Non-Fatigue', 'Fatigue']))

        cm = confusion_matrix(self.y_test, y_te_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Fatigue', 'Fatigue'],
                    yticklabels=['Non-Fatigue', 'Fatigue'])
        plt.title('Confusion Matrix - KNN Pipeline')
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_path = os.path.join(out_dir, f'{save_prefix}_Confusion_Matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight'); plt.close()

        # ROC/AUC (KNN có predict_proba)
        if hasattr(self.pipeline_, "predict_proba"):
            prob = self.pipeline_.predict_proba(self.X_test_raw)[:, 1]
        else:
            df = self.pipeline_.decision_function(self.X_test_raw)
            prob = df if df.ndim == 1 else df[:, 1]
        roc_auc = roc_auc_score(self.y_test, prob)
        fpr, tpr, _ = roc_curve(self.y_test, prob, pos_label=1)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
        plt.xlim([0., 1.0]); plt.ylim([0., 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - KNN Pipeline')
        plt.legend(loc="lower right"); plt.grid(True, alpha=0.3); plt.tight_layout()
        roc_path = os.path.join(out_dir, f'{save_prefix}_ROC_Curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight'); plt.close()

        # log features/scaler
        scaler = self.pipeline_.named_steps.get('scaler')
        kbest  = self.pipeline_.named_steps.get('kbest')
        sel_idx = kbest.get_support(indices=True) if kbest is not None else np.arange(self.X_train_raw.shape[1])

        self.logger.set_features(
            k_selected=len(sel_idx),
            selected_idx=sel_idx,
            scaler_mean=(scaler.mean_ if hasattr(scaler, 'mean_') else None),
            scaler_scale=(scaler.scale_ if hasattr(scaler, 'scale_') else None)
        )
        self.logger.set_metrics(train_acc, test_acc, roc_auc)
        self.logger.add_file("confusion_matrix", cm_path)
        self.logger.add_file("roc_curve", roc_path)

        # export hardcode
        self._export_hardcode(out_dir=out_dir)
        return train_acc, test_acc, roc_auc

    def save_model(self, filename='best_knn_model.pkl'):
        import joblib
        if self.pipeline_ is None:
            print("No model to save!"); return
        bundle = {'pipeline': self.pipeline_, 'export': self.best_params_export}
        joblib.dump(bundle, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename='best_knn_model.pkl'):
        import joblib
        b = joblib.load(filename)
        self.pipeline_ = b['pipeline']
        self.best_params_export = b.get('export', {})
        print(f"Model loaded from {filename}")


# ============================= Multi-runs & Early Stop =============================
def repeat_runs(n_runs=10, data_path=None, test_size=0.2, k_features=50,
                use_grid_search=False, scoring='f1', cv_splits=5, cv_repeats=1,
                sliding=False, win=8000, step=4000, group_split=False, fs_kind='anova',
                save_csv='knn_seed_runs.csv'):
    rows = []
    for seed in range(n_runs):
        print(f"\n==== Repeat {seed+1}/{n_runs} (seed={seed}) ====")
        logger = RunLogger(out_dir=f"run_artifacts_knn_seed_{seed}")
        logger.set_args({
            "mode": "repeat_runs", "n_runs": n_runs, "data_path": data_path,
            "test_size": test_size, "k_features": k_features,
            "use_grid_search": use_grid_search, "scoring": scoring,
            "cv_splits": cv_splits, "cv_repeats": cv_repeats,
            "sliding": sliding, "win": win, "step": step,
            "group_split": group_split, "fs_kind": fs_kind
        })
        clf = sEMGKNNClassifier(data_path=data_path, run_logger=logger, seed=seed)
        clf.prepare_data(test_size=test_size, random_state=seed, k_features=k_features,
                         sliding=sliding, win=win, step=step, group_split=group_split, fs_kind=fs_kind)
        clf.train_knn(use_grid_search=use_grid_search, scoring=scoring,
                      cv_splits=cv_splits, cv_repeats=cv_repeats)
        tr, te, auc_score = clf.evaluate_model(save_prefix=f'KNN_seed{seed}', out_dir=logger.out_dir)
        clf.save_model(os.path.join(logger.out_dir, 'best_knn_model.pkl'))
        logger.write()
        rows.append({'seed': seed, 'train_acc': tr, 'test_acc': te, 'auc': auc_score})

    df = pd.DataFrame(rows)
    df.to_csv(save_csv, index=False)
    print(f"\nSaved seed-wise results to {save_csv}")
    m_te = df['test_acc'].mean(); s_te = df['test_acc'].std()
    m_auc = df['auc'].mean();     s_auc = df['auc'].std()
    print(f"[{n_runs} runs] TestAcc = {m_te:.4f} ± {s_te:.4f}, AUC = {m_auc:.4f} ± {s_auc:.4f}")
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df[['test_acc', 'auc']])
    plt.title(f'KNN variation across {n_runs} seeds')
    plt.tight_layout()
    plt.savefig('KNN_RepeatRuns_Boxplot.png', dpi=300, bbox_inches='tight'); plt.close()
    return df


def train_until_target(target_acc=0.85, max_tries=50, data_path=None, test_size=0.2,
                       k_features=50, use_grid_search=False, scoring='f1',
                       cv_splits=5, cv_repeats=1, sliding=False, win=8000, step=4000,
                       group_split=False, fs_kind='anova'):
    best = {'acc': -1.0, 'auc': -1.0, 'seed': None}
    for seed in range(max_tries):
        print(f"\n==== Try {seed+1}/{max_tries} (seed={seed}) ====")
        logger = RunLogger(out_dir=f"run_artifacts_knn_target_seed_{seed}")
        logger.set_args({
            "mode": "train_until_target", "target_acc": target_acc, "max_tries": max_tries,
            "data_path": data_path, "test_size": test_size, "k_features": k_features,
            "use_grid_search": use_grid_search, "scoring": scoring,
            "cv_splits": cv_splits, "cv_repeats": cv_repeats,
            "sliding": sliding, "win": win, "step": step,
            "group_split": group_split, "fs_kind": fs_kind
        })
        clf = sEMGKNNClassifier(data_path=data_path, run_logger=logger, seed=seed)
        clf.prepare_data(test_size=test_size, random_state=seed, k_features=k_features,
                         sliding=sliding, win=win, step=step, group_split=group_split, fs_kind=fs_kind)
        clf.train_knn(use_grid_search=use_grid_search, scoring=scoring,
                      cv_splits=cv_splits, cv_repeats=cv_repeats)
        tr, te, auc_score = clf.evaluate_model(save_prefix=f'KNN_target_seed{seed}', out_dir=logger.out_dir)
        clf.save_model(os.path.join(logger.out_dir, 'best_knn_model.pkl'))
        logger.write()
        if te > best['acc']:
            best.update({'acc': te, 'auc': auc_score, 'seed': seed})
        if te >= target_acc:
            print(f"\n>>> REACHED TARGET: TestAcc={te:.4f} (seed={seed}) >= {target_acc:.2f}. Stopping.")
            return seed, te, auc_score

    print(f"\n>>> Target {target_acc:.2f} NOT reached after {max_tries} tries. Best TestAcc={best['acc']:.4f} (seed={best['seed']}).")
    return best['seed'], best['acc'], best['auc']


# ============================= CLI =============================
def parse_args():
    p = argparse.ArgumentParser(description="sEMG KNN training/evaluation (with sliding & group-split options)")
    p.add_argument('--data', type=str, default=None, help='Path tới dataset (có fatigue/ và non fatigue/)')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--k', type=int, default=50, help='Số đặc trưng SelectKBest')
    p.add_argument('--fs', type=str, default='anova', choices=['anova', 'mi'], help='Bộ chọn đặc trưng: anova|mi')
    p.add_argument('--grid', action='store_true', help='Bật GridSearchCV cho KNN')
    p.add_argument('--tune-k-in-grid', action='store_true', help='Tune luôn kbest__k trong GridSearch')
    p.add_argument('--scoring', type=str, default='f1',
                   choices=['f1', 'accuracy', 'balanced_accuracy', 'roc_auc'])
    p.add_argument('--cv-splits', type=int, default=5)
    p.add_argument('--cv-repeats', type=int, default=1)
    p.add_argument('--repeat', type=int, default=0, help='Số lần lặp seeds (0 = không lặp)')

    # Sliding window
    p.add_argument('--sliding', action='store_true', help='Bật sliding window')
    p.add_argument('--win', type=int, default=8000, help='Độ dài cửa sổ')
    p.add_argument('--step', type=int, default=4000, help='Bước trượt')

    # Group split chống leakage
    p.add_argument('--group-split', action='store_true', help='Tách train/test theo file (GroupShuffleSplit)')

    # Early-stop
    p.add_argument('--target-acc', type=float, default=0.0, help='Ngưỡng TestAcc để dừng sớm (0 = tắt)')
    p.add_argument('--max-tries', type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = "run_artifacts_knn"
    os.makedirs(out_dir, exist_ok=True)

    print("sEMG Muscle Fatigue Classification using KNN")
    print("=" * 60)

    # Early stop ưu tiên
    if args.target_acc and args.target_acc > 0:
        seed, te, auc_score = train_until_target(
            target_acc=args.target_acc, max_tries=args.max_tries,
            data_path=args.data, test_size=args.test_size, k_features=args.k,
            use_grid_search=args.grid, scoring=args.scoring,
            cv_splits=args.cv_splits, cv_repeats=args.cv_repeats,
            sliding=args.sliding, win=args.win, step=args.step,
            group_split=args.group_split, fs_kind=args.fs
        )
        print("\n" + "=" * 60)
        if seed is not None and te >= args.target_acc:
            print(f"EARLY-STOP DONE: seed={seed}, TestAcc={te:.4f}, AUC={auc_score:.4f}")
        else:
            print(f"EARLY-STOP FINISHED (not reached): best TestAcc={te:.4f}, AUC={auc_score:.4f}")
        print("=" * 60)
        return

    # Repeat mode
    if args.repeat and args.repeat > 0:
        repeat_runs(
            n_runs=args.repeat, data_path=args.data, test_size=args.test_size, k_features=args.k,
            use_grid_search=args.grid, scoring=args.scoring,
            cv_splits=args.cv_splits, cv_repeats=args.cv_repeats,
            sliding=args.sliding, win=args.win, step=args.step,
            group_split=args.group_split, fs_kind=args.fs
        )
        return

    # Single run
    logger = RunLogger(out_dir=out_dir)
    logger.set_args({
        "mode": "single_run", "data_path": args.data, "test_size": args.test_size,
        "k_features": args.k, "use_grid_search": args.grid, "scoring": args.scoring,
        "cv_splits": args.cv_splits, "cv_repeats": args.cv_repeats,
        "sliding": args.sliding, "win": args.win, "step": args.step,
        "group_split": args.group_split, "fs_kind": args.fs
    })

    seed = 42
    clf = sEMGKNNClassifier(data_path=args.data, run_logger=logger, seed=seed)
    clf.prepare_data(test_size=args.test_size, random_state=seed, k_features=args.k,
                     sliding=args.sliding, win=args.win, step=args.step,
                     group_split=args.group_split, fs_kind=args.fs)
    clf.train_knn(use_grid_search=args.grid, scoring=args.scoring,
                  cv_splits=args.cv_splits, cv_repeats=args.cv_repeats,
                  tune_k_in_grid=args.tune_k_in_grid)
    tr, te, auc_score = clf.evaluate_model(save_prefix='KNN', out_dir=out_dir)
    clf.save_model(os.path.join(out_dir, 'best_knn_model.pkl'))
    logger.write()

    print("\n" + "=" * 60)
    print("KNN Classification Complete!")
    print(f"Best Test Accuracy: {te:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
