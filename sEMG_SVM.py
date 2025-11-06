#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sEMG Muscle Fatigue Classification using SVM
(Overfitting-safe + Sliding Window + Group Split + Hardcode export)

- Lọc: zero-mean -> notch 50 Hz (nếu valid) -> bandpass (10-100 Hz) -> RECTIFY -> z-score
- ZC/SSC tính trên tín hiệu đã lọc nhưng CHƯA rectify
- Welch PSD + MNF, MDF, bandpower 10-30/30-70/70-100 Hz
- Pipeline: StandardScaler -> SelectKBest(f_classif|mutual_info, k) -> SVC
- CV nghiêm ngặt: RepeatedStratifiedKFold (+ optional Nested CV)
- GridSearch dò k, C, gamma (ưu tiên giá trị nhỏ để giảm overfit)
- Sliding window để gia tăng số mẫu; Group-aware split chống leakage theo file
- Xuất:
  + best_svm_model.pkl (pipeline đã fit)
  + best_hardcode.json (tham số cần để hardcode)
  + hardcoded_params.py (file Python hằng số để copy/import)
"""
import os, sys, json, platform, warnings, argparse, datetime
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, iirnotch, welch

from sklearn import __version__ as sklearn_version
from sklearn.model_selection import (
    train_test_split, GridSearchCV,
    RepeatedStratifiedKFold, StratifiedKFold,
    GroupShuffleSplit, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_curve, auc, roc_auc_score
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


# ============================= Run Logger =============================
class RunLogger:
    def __init__(self, out_dir="run_artifacts"):
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
            "args": {},
            "seed": None,
            "data": {},
            "preprocess": {},
            "features": {},
            "model": {},
            "metrics": {},
            "files": {},
        }

    def set_args(self, args_dict): self.manifest["args"] = args_dict
    def set_seed(self, seed: int): self.manifest["seed"] = int(seed)

    def set_data_info(self, fat_files, non_files, train_idx, test_idx):
        self.manifest["data"] = {
            "fatigue_files_sorted": fat_files,
            "non_fatigue_files_sorted": non_files,
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
        }

    def set_preprocess(self, fs, lowcut, highcut, notch_hz, ranges, sliding, win, step, group_split, fs_kind):
        self.manifest["preprocess"] = {
            "fs": fs, "lowcut": lowcut, "highcut": highcut,
            "notch_hz": notch_hz, "ranges_to_keep": ranges,
            "sliding": bool(sliding), "win": win, "step": step,
            "group_split": bool(group_split), "feature_selector": fs_kind
        }

    def set_features(self, k_selected, selected_idx, scaler_mean, scaler_scale):
        self.manifest["features"] = {
            "k_selected": int(k_selected),
            "selected_feature_indices": selected_idx.tolist(),
            "scaler_mean": scaler_mean.tolist() if scaler_mean is not None else None,
            "scaler_scale": scaler_scale.tolist() if scaler_scale is not None else None,
        }

    def set_model(self, params: dict, best_params: dict | None = None):
        self.manifest["model"] = {
            "svm_params": params,
            "best_params": best_params,
        }

    def set_metrics(self, train_acc, test_acc, roc_auc):
        self.manifest["metrics"] = {
            "train_acc": float(train_acc) if train_acc is not None else None,
            "test_acc": float(test_acc) if test_acc is not None else None,
            "auc": float(roc_auc) if roc_auc is not None else None,
        }

    def add_file(self, key, path):
        self.manifest["files"][key] = os.path.abspath(path)

    def write(self, name="run_manifest.json"):
        out_path = os.path.join(self.out_dir, name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2, default=str)
        print(f"[LOG] Wrote manifest: {out_path}")
        return out_path

    def write_repro_shell(self, base_cmd: str, name="repro_run.sh"):
        sh = os.path.join(self.out_dir, name)
        with open(sh, "w", encoding="utf-8") as f:
            f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
            f.write(f"# Re-run command captured at {datetime.datetime.now().isoformat()}\n")
            f.write(base_cmd.strip() + "\n")
        os.chmod(sh, 0o755)
        print(f"[LOG] Wrote repro shell: {sh}")
        return sh


# ============================= Classifier =============================
class sEMGSVMClassifier:
    def __init__(self, data_path=None, run_logger: RunLogger | None = None, seed: int = 42):
        if data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(script_dir, 'dataset')
        self.data_path = data_path
        self.data_fatigue_path = os.path.join(data_path, 'fatigue')
        self.data_nonfatigue_path = os.path.join(data_path, 'non fatigue')

        self.pipeline_: Pipeline | None = None
        self.logger = run_logger or RunLogger()
        self.seed = int(seed)

        # raw matrices
        self.X_train_raw = self.X_test_raw = None
        self.y_train = self.y_test = None
        self.groups_train = self.groups_test = None
        self.train_idx = self.test_idx = None

        # signal params
        self.lowcut = 10.0
        self.highcut = 100.0
        self.fs = 1000.0
        self.notch_hz = 50.0

        # best export
        self.best_params_export = {}

        print("sEMG SVM Classifier initialized")
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
            np.sqrt(np.mean(x**2)),                 # RMS
            np.mean(np.abs(x - mu)),                # MAD
            np.sum(np.abs(np.diff(x)))              # WL / TV
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
            mask = (f >= f1) & (f <= f2)
            return np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0

        total = np.trapz(Pxx, f)
        bp_l = bandpower(10, 30)
        bp_m = bandpower(30, 70)
        bp_h = bandpower(70, 100)
        mnf = np.sum(f * Pxx) / np.sum(Pxx)
        csum = np.cumsum(Pxx)
        half = 0.5 * csum[-1]
        mdf = f[np.searchsorted(csum, half)]

        feats = [
            np.mean(Pxx), np.std(Pxx), np.max(Pxx), total,
            bp_l, bp_m, bp_h, mnf, mdf, mnf  # centroid ~ mnf
        ]
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
        """Trích nhiều cửa sổ (zero-mean -> lọc -> rectify -> z-score -> features)."""
        sig = np.asarray(sig, dtype=float)
        sig = sig - np.mean(sig)
        feats_all = []
        for start in range(0, max(0, len(sig) - win + 1), step):
            part = sig[start:start+win]
            if len(part) < win:
                break
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
        return feats_all  # list of vectors

    # ------------ DATA PIPELINE ------------
    def load_dataset(self, sliding=False, win=8000, step=4000):
        print("Loading dataset...")
        fat_files = sorted([f for f in os.listdir(self.data_fatigue_path) if f.endswith('.csv')])
        non_files = sorted([f for f in os.listdir(self.data_nonfatigue_path) if f.endswith('.csv')])
        print(f"Found {len(fat_files)} fatigue files and {len(non_files)} non-fatigue files")

        ranges_to_keep = [(15000, 25000), (30000, 35000)]
        X, y, groups = [], [], []  # groups = string id theo file

        # fatigue
        print("Processing fatigue files...")
        for idx, fn in enumerate(fat_files):
            path = os.path.join(self.data_fatigue_path, fn)
            try:
                sig = self._read_amplitudo(path)
                gid = f"fatigue_{fn}"  # NHÓM DẠNG CHUỖI
                if sliding:
                    feats_list = self._extract_windows(sig, win, step)
                    for fv in feats_list:
                        X.append(fv); y.append(1); groups.append(gid)
                else:
                    feats = self._extract_one(sig, ranges_to_keep)
                    if feats is not None:
                        X.append(feats); y.append(1); groups.append(gid)
            except Exception as ex:
                print(f"[WARN] {fn}: {ex}")

        # non-fatigue
        print("Processing non-fatigue files...")
        for idx, fn in enumerate(non_files):
            path = os.path.join(self.data_nonfatigue_path, fn)
            try:
                sig = self._read_amplitudo(path)
                gid = f"nonfatigue_{fn}"
                if sliding:
                    feats_list = self._extract_windows(sig, win, step)
                    for fv in feats_list:
                        X.append(fv); y.append(0); groups.append(gid)
                else:
                    feats = self._extract_one(sig, ranges_to_keep)
                    if feats is not None:
                        X.append(feats); y.append(0); groups.append(gid)
            except Exception as ex:
                print(f"[WARN] {fn}: {ex}")

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        groups = np.asarray(groups, dtype=str)  # QUAN TRỌNG: ép chuỗi đồng nhất

        print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")
        print(f"Class distribution: {np.sum(y==0)} non-fatigue, {np.sum(y==1)} fatigue")
        return X, y, groups, fat_files, non_files, ranges_to_keep

    def prepare_data(self, test_size=0.2, random_state=42, out_dir="run_artifacts",
                     sliding=False, win=8000, step=4000, group_split=False, fs_kind='anova'):
        print("Preparing data...")
        self.logger.set_seed(random_state)

        X, y, groups, fat_files, non_files, ranges_to_keep = self.load_dataset(sliding=sliding, win=win, step=step)
        groups = np.asarray(groups, dtype=str)

        if group_split:
            # đảm bảo các window của cùng 1 file không xuất hiện ở cả train & test
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            tr_idx, te_idx = next(gss.split(X, y, groups=groups))
            self.train_idx, self.test_idx = tr_idx, te_idx
            self.X_train_raw, self.X_test_raw = X[tr_idx], X[te_idx]
            self.y_train, self.y_test = y[tr_idx], y[te_idx]
            self.groups_train, self.groups_test = groups[tr_idx], groups[te_idx]
        else:
            all_idx = np.arange(len(y))
            tr_idx, te_idx, y_tr, y_te = train_test_split(
                all_idx, y, test_size=test_size, random_state=random_state, stratify=y
            )
            self.train_idx, self.test_idx = tr_idx, te_idx
            self.X_train_raw, self.X_test_raw = X[tr_idx], X[te_idx]
            self.y_train, self.y_test = y_tr, y_te
            self.groups_train = groups[tr_idx]; self.groups_test = groups[te_idx]

        self.logger.set_data_info(fat_files, non_files, self.train_idx, self.test_idx)
        self.logger.set_preprocess(self.fs, self.lowcut, self.highcut, self.notch_hz,
                                   ranges_to_keep, sliding, win, step, group_split, fs_kind)

    # ------------ MODEL (Pipeline + Repeated CV + optional Nested) ------------
    def _make_pipeline_and_grid(self, kernel: str, max_k: int, scoring: str, fs_kind: str):
        # k ứng viên: trải đều, tránh quá lớn để giảm overfit
        k_candidates = sorted(list({min(max_k, v) for v in [20, 30, 40, 50, 60, 70, max_k]}))
        score_func = f_classif if fs_kind == 'anova' else mutual_info_classif

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('kbest', SelectKBest(score_func=score_func, k=min(max_k, 50))),  # override bởi grid
            ('svc', SVC(class_weight='balanced', probability=True, random_state=self.seed))
        ])

        if kernel == 'rbf':
            param_grid = {
                'kbest__k': k_candidates,
                'svc__kernel': ['rbf'],
                'svc__C': [0.01, 0.1, 1, 10, 100],
                'svc__gamma': ['scale', 'auto', 1e-4, 1e-3, 1e-2]
            }
        elif kernel == 'linear':
            param_grid = {
                'kbest__k': k_candidates,
                'svc__kernel': ['linear'],
                'svc__C': [0.01, 0.1, 1, 10, 100],
            }
        elif kernel == 'poly':
            param_grid = {
                'kbest__k': k_candidates,
                'svc__kernel': ['poly'],
                'svc__degree': [2, 3],
                'svc__C': [0.01, 0.1, 1, 10],
                'svc__gamma': ['scale', 'auto', 1e-3, 1e-2]
            }
        else:  # sigmoid
            param_grid = {
                'kbest__k': k_candidates,
                'svc__kernel': ['sigmoid'],
                'svc__C': [0.01, 0.1, 1, 10],
                'svc__gamma': ['scale', 'auto', 1e-3, 1e-2]
            }
        return pipe, param_grid

    def train_svm(self, kernel='rbf', use_grid_search=True,
                  scoring='f1', cv_splits=5, cv_repeats=3, nested=False, fs_kind='anova'):
        """
        scoring: 'f1' (default), hoặc 'balanced_accuracy', 'roc_auc'
        fs_kind: 'anova' (f_classif) hoặc 'mi' (mutual_info_classif)
        """
        print("Training SVM model with Overfit-safe CV...")
        max_k = int(min(70, self.X_train_raw.shape[1]))
        pipe, param_grid = self._make_pipeline_and_grid(kernel, max_k, scoring, fs_kind)
        inner_cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=self.seed)

        if use_grid_search:
            gs = GridSearchCV(pipe, param_grid=param_grid, scoring=scoring, cv=inner_cv,
                              n_jobs=-1, verbose=1, refit=True)
        else:
            gs = pipe.set_params(**{
                'kbest__k': min(40, max_k),
                'svc__kernel': kernel,
                'svc__C': 1.0,
                'svc__gamma': 'scale'
            })

        if nested and use_grid_search:
            outer_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.seed)
            scores = cross_val_score(gs, self.X_train_raw, self.y_train,
                                     scoring=scoring, cv=outer_cv, n_jobs=-1)
            print(f"[Nested CV] {scoring} = {scores.mean():.4f} ± {scores.std():.4f}")

        if use_grid_search:
            gs.fit(self.X_train_raw, self.y_train)
            self.pipeline_ = gs.best_estimator_
            best_params = gs.best_params_
            print(f"Best parameters: {best_params}")
            print(f"Best inner-CV ({scoring}) score: {gs.best_score_:.4f}")

            svc = self.pipeline_['svc']; kbest = self.pipeline_['kbest']
            safe_params = {
                "steps": list(self.pipeline_.named_steps.keys()),
                "kbest_k": int(kbest.k),
                "svc": {
                    "kernel": svc.kernel,
                    "C": float(svc.C),
                    "gamma": (svc.gamma if isinstance(svc.gamma, str) else float(svc.gamma)),
                    "degree": getattr(svc, "degree", None),
                    "class_weight": svc.class_weight,
                    "probability": bool(svc.probability),
                    "random_state": self.seed,
                }
            }
            self.logger.set_model(params=safe_params, best_params=best_params)
            self.best_params_export = {
                "seed": self.seed,
                "train_idx": self.train_idx.tolist(),
                "test_idx": self.test_idx.tolist(),
                "kbest_k": int(kbest.k),
                "kernel": svc.kernel,
                "C": float(svc.C),
                "gamma": (svc.gamma if isinstance(svc.gamma, str) else float(svc.gamma)),
                "degree": getattr(svc, "degree", None),
                "class_weight": svc.class_weight,
                "probability": bool(svc.probability),
                "scoring": scoring,
                "cv_splits": cv_splits,
                "cv_repeats": cv_repeats,
                "feature_selector": fs_kind
            }
        else:
            self.pipeline_ = gs
            self.pipeline_.fit(self.X_train_raw, self.y_train)
            svc = self.pipeline_['svc']; kbest = self.pipeline_['kbest']
            safe_params = {
                "steps": list(self.pipeline_.named_steps.keys()),
                "kbest_k": int(kbest.k),
                "svc": {
                    "kernel": svc.kernel,
                    "C": float(svc.C),
                    "gamma": (svc.gamma if isinstance(svc.gamma, str) else float(svc.gamma)),
                    "degree": getattr(svc, "degree", None),
                    "class_weight": svc.class_weight,
                    "probability": bool(svc.probability),
                    "random_state": self.seed,
                }
            }
            self.logger.set_model(params=safe_params, best_params=None)
            self.best_params_export = {
                "seed": self.seed,
                "train_idx": self.train_idx.tolist(),
                "test_idx": self.test_idx.tolist(),
                "kbest_k": int(kbest.k),
                "kernel": svc.kernel,
                "C": float(svc.C),
                "gamma": (svc.gamma if isinstance(svc.gamma, str) else float(svc.gamma)),
                "degree": getattr(svc, "degree", None),
                "class_weight": svc.class_weight,
                "probability": bool(svc.probability),
                "scoring": scoring,
                "cv_splits": cv_splits,
                "cv_repeats": cv_repeats,
                "feature_selector": fs_kind
            }

        acc_scores = cross_val_score(self.pipeline_, self.X_train_raw, self.y_train,
                                     scoring='accuracy', cv=inner_cv, n_jobs=-1)
        print(f"Cross-validation accuracy (inner schema): {acc_scores.mean():.4f} (+/- {acc_scores.std()*2:.4f})")

    def _export_hardcode(self, out_dir="run_artifacts"):
        if not self.best_params_export:
            return
        json_path = os.path.join(out_dir, "best_hardcode.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.best_params_export, f, ensure_ascii=False, indent=2)
        print(f"[LOG] Wrote hardcode params: {json_path}")

        py_path = os.path.join(out_dir, "hardcoded_params.py")
        with open(py_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated best params to hardcode\n")
            f.write("BEST = " + json.dumps(self.best_params_export, ensure_ascii=False, indent=2) + "\n")
            f.write("""def example_rebuild_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.svm import SVC
    fs_kind = BEST.get("feature_selector", "anova")
    score_func = f_classif if fs_kind == "anova" else mutual_info_classif
    k = BEST["kbest_k"]
    svc = SVC(kernel=BEST["kernel"], C=BEST["C"],
              gamma=BEST["gamma"], degree=BEST["degree"] if BEST["degree"] is not None else 3,
              class_weight=BEST["class_weight"], probability=BEST["probability"],
              random_state=BEST["seed"])
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('kbest', SelectKBest(score_func=score_func, k=k)),
                     ('svc', svc)])
    return pipe
""")
        print(f"[LOG] Wrote hardcoded Python: {py_path}")

    def evaluate_model(self, save_prefix='SVM', out_dir="run_artifacts"):
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

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_te_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Fatigue', 'Fatigue'],
                    yticklabels=['Non-Fatigue', 'Fatigue'])
        plt.title('Confusion Matrix - SVM Pipeline')
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_path = os.path.join(out_dir, f'{save_prefix}_Confusion_Matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight'); plt.close()

        # ROC / AUC
        svc = self.pipeline_['svc']
        if hasattr(self.pipeline_, "predict_proba") and hasattr(svc, "predict_proba"):
            proba = self.pipeline_.predict_proba(self.X_test_raw)
            pos_idx = int(np.where(svc.classes_ == 1)[0][0])
            y_score = proba[:, pos_idx]
        else:
            df = self.pipeline_.decision_function(self.X_test_raw)
            if df.ndim == 1:
                y_score = df
            else:
                pos_idx = int(np.where(svc.classes_ == 1)[0][0])
                y_score = df[:, pos_idx]

        roc_auc = roc_auc_score(self.y_test, y_score)
        if roc_auc < 0.5:
            print("[WARN] AUC < 0.5 → có thể nhãn/chiều điểm bị đảo.")
        fpr, tpr, _ = roc_curve(self.y_test, y_score, pos_label=1)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
        plt.xlim([0., 1.0]); plt.ylim([0., 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - SVM Pipeline')
        plt.legend(loc="lower right"); plt.grid(True, alpha=0.3); plt.tight_layout()
        roc_path = os.path.join(out_dir, f'{save_prefix}_ROC_Curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight'); plt.close()

        # log features/scaler (lấy từ pipeline đã fit)
        scaler = self.pipeline_.named_steps.get('scaler', None)
        kbest  = self.pipeline_.named_steps.get('kbest', None)
        selected_idx = kbest.get_support(indices=True) if kbest is not None else np.arange(self.X_train_raw.shape[1])

        self.logger.set_features(
            k_selected=len(selected_idx),
            selected_idx=selected_idx,
            scaler_mean=(scaler.mean_ if hasattr(scaler, 'mean_') else None),
            scaler_scale=(scaler.scale_ if hasattr(scaler, 'scale_') else None)
        )
        self.logger.set_metrics(train_acc, test_acc, roc_auc)
        self.logger.add_file("confusion_matrix", cm_path)
        self.logger.add_file("roc_curve", roc_path)

        # xuất hardcode
        self._export_hardcode(out_dir=out_dir)
        return train_acc, test_acc, roc_auc

    def compare_kernels(self, out_dir="run_artifacts", scoring='f1', cv_splits=5, cv_repeats=3, fs_kind='anova'):
        print("Comparing different SVM kernels (Pipeline + CV)...")
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        results = {}
        inner_cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=self.seed)

        for k in kernels:
            pipe, param_grid = self._make_pipeline_and_grid(
                k, max_k=min(70, self.X_train_raw.shape[1]), scoring=scoring, fs_kind=fs_kind
            )
            gs = GridSearchCV(pipe, param_grid=param_grid, scoring=scoring, cv=inner_cv, n_jobs=-1, refit=True)
            gs.fit(self.X_train_raw, self.y_train)
            y_pred = gs.best_estimator_.predict(self.X_test_raw)
            acc = accuracy_score(self.y_test, y_pred)
            cv_mean = gs.best_score_
            results[k] = {'accuracy': acc, 'cv_mean': cv_mean}
            print(f"{k.capitalize()} - Test Accuracy: {acc:.4f}, Best {scoring}: {cv_mean:.4f}")

        ks = list(results.keys()); accs = [results[k]['accuracy'] for k in ks]
        cvm = [results[k]['cv_mean'] for k in ks]
        x = np.arange(len(ks)); w = 0.38

        plt.figure(figsize=(10, 6))
        r1 = plt.bar(x - w/2, accs, w, label='Test Accuracy', alpha=0.85)
        r2 = plt.bar(x + w/2, cvm, w, label=f'CV ({scoring})', alpha=0.85)
        plt.ylabel('Score'); plt.title('SVM Kernel Comparison (with Pipeline)')
        plt.xticks(x, ks); plt.legend(); plt.grid(True, alpha=0.3)
        for r in list(r1) + list(r2):
            h = r.get_height()
            plt.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom')
        plt.tight_layout()
        outp = os.path.join(out_dir, 'SVM_Kernel_Comparison.png')
        plt.savefig(outp, dpi=300, bbox_inches='tight'); plt.close()
        return results

    def save_model(self, filename='best_svm_model.pkl'):
        import joblib
        if self.pipeline_ is None:
            print("No model to save!"); return
        joblib.dump(self.pipeline_, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename='best_svm_model.pkl'):
        import joblib
        self.pipeline_ = joblib.load(filename)
        print(f"Model loaded from {filename}")


# ============================= Multi-runs & Early Stop =============================
def repeat_runs(n_runs=10, data_path=None, test_size=0.2, kernel='rbf',
                use_grid_search=True, scoring='f1', cv_splits=5, cv_repeats=3,
                save_csv='svm_seed_runs.csv', sliding=False, win=8000, step=4000,
                group_split=False, fs_kind='anova'):
    rows = []
    for seed in range(n_runs):
        print(f"\n==== Repeat {seed+1}/{n_runs} (seed={seed}) ====")
        logger = RunLogger(out_dir=f"run_artifacts_seed_{seed}")
        logger.set_args({
            "mode": "repeat_runs", "n_runs": n_runs, "data_path": data_path,
            "test_size": test_size, "kernel": kernel,
            "use_grid_search": use_grid_search, "scoring": scoring,
            "cv_splits": cv_splits, "cv_repeats": cv_repeats,
            "sliding": sliding, "win": win, "step": step,
            "group_split": group_split, "feature_selector": fs_kind
        })
        clf = sEMGSVMClassifier(data_path=data_path, run_logger=logger, seed=seed)
        clf.prepare_data(test_size=test_size, random_state=seed, out_dir=logger.out_dir,
                         sliding=sliding, win=win, step=step, group_split=group_split, fs_kind=fs_kind)
        clf.train_svm(kernel=kernel, use_grid_search=use_grid_search, scoring=scoring,
                      cv_splits=cv_splits, cv_repeats=cv_repeats, nested=False, fs_kind=fs_kind)
        tr, te, auc_score = clf.evaluate_model(save_prefix=f'SVM_seed{seed}', out_dir=logger.out_dir)
        logger.write()
        base_cmd = f"python {os.path.basename(__file__)} --data {data_path or './dataset'} --test-size {test_size} --kernel {kernel} --grid"
        logger.write_repro_shell(base_cmd)
        rows.append({'seed': seed, 'train_acc': tr, 'test_acc': te, 'auc': auc_score})

    df = pd.DataFrame(rows)
    df.to_csv(save_csv, index=False)
    print(f"\nSaved seed-wise results to {save_csv}")
    m_te = df['test_acc'].mean(); s_te = df['test_acc'].std()
    m_auc = df['auc'].mean();  s_auc = df['auc'].std()
    print(f"[{n_runs} runs] TestAcc = {m_te:.4f} ± {s_te:.4f}, AUC = {m_auc:.4f} ± {s_auc:.4f}")
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df[['test_acc', 'auc']])
    plt.title(f'Variation across {n_runs} seeds')
    plt.tight_layout()
    plt.savefig('SVM_RepeatRuns_Boxplot.png', dpi=300, bbox_inches='tight'); plt.close()
    return df


def train_until_target(target_acc=0.85, max_tries=50, data_path=None, test_size=0.2,
                       kernel='rbf', use_grid_search=True, scoring='f1',
                       cv_splits=5, cv_repeats=3, sliding=False, win=8000, step=4000,
                       group_split=False, fs_kind='anova'):
    best = {'acc': -1.0, 'auc': -1.0, 'seed': None}
    for seed in range(max_tries):
        print(f"\n==== Try {seed+1}/{max_tries} (seed={seed}) ====")
        logger = RunLogger(out_dir=f"run_artifacts_target_seed_{seed}")
        logger.set_args({
            "mode": "train_until_target", "target_acc": target_acc, "max_tries": max_tries,
            "data_path": data_path, "test_size": test_size,
            "kernel": kernel, "use_grid_search": use_grid_search,
            "scoring": scoring, "cv_splits": cv_splits, "cv_repeats": cv_repeats,
            "sliding": sliding, "win": win, "step": step,
            "group_split": group_split, "feature_selector": fs_kind
        })
        clf = sEMGSVMClassifier(data_path=data_path, run_logger=logger, seed=seed)
        clf.prepare_data(test_size=test_size, random_state=seed, out_dir=logger.out_dir,
                         sliding=sliding, win=win, step=step, group_split=group_split, fs_kind=fs_kind)
        clf.train_svm(kernel=kernel, use_grid_search=use_grid_search, scoring=scoring,
                      cv_splits=cv_splits, cv_repeats=cv_repeats, nested=False, fs_kind=fs_kind)
        tr, te, auc_score = clf.evaluate_model(save_prefix=f'SVM_target_seed{seed}', out_dir=logger.out_dir)
        logger.write()
        base_cmd = f"python {os.path.basename(__file__)} --data {data_path or './dataset'} --test-size {test_size} --kernel {kernel} --grid"
        logger.write_repro_shell(base_cmd)
        if te > best['acc']:
            best.update({'acc': te, 'auc': auc_score, 'seed': seed})
            clf.save_model(os.path.join(logger.out_dir, 'best_svm_model.pkl'))
        if te >= target_acc:
            print(f"\n>>> REACHED TARGET: TestAcc={te:.4f} (seed={seed}) >= {target_acc:.2f}. Stopping.")
            return seed, te, auc_score

    print(f"\n>>> Target {target_acc:.2f} NOT reached after {max_tries} tries. Best TestAcc={best['acc']:.4f} (seed={best['seed']}).")
    return best['seed'], best['acc'], best['auc']


# ============================= CLI =============================
def parse_args():
    p = argparse.ArgumentParser(description="sEMG SVM training/evaluation (overfitting-safe, sliding-window, group-split)")
    p.add_argument('--data', type=str, default=None, help='Path to dataset (folder có fatigue/ và non fatigue/)')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--kernel', type=str, default='rbf', choices=['rbf', 'linear', 'poly', 'sigmoid'])
    p.add_argument('--grid', action='store_true', help='Bật GridSearchCV (khuyến nghị)')
    p.add_argument('--repeat', type=int, default=0, help='Số lần lặp với seed khác nhau (0 = không lặp)')
    p.add_argument('--scoring', type=str, default='f1', choices=['f1', 'balanced_accuracy', 'roc_auc'],
                   help='Tiêu chí tối ưu trong GridSearchCV')
    p.add_argument('--cv-splits', type=int, default=5, help='Số fold cho CV')
    p.add_argument('--cv-repeats', type=int, default=3, help='Số lần lặp cho RepeatedStratifiedKFold')
    # Sliding window
    p.add_argument('--sliding', action='store_true', help='Bật sliding window augmentation')
    p.add_argument('--win', type=int, default=8000, help='Độ dài cửa sổ (samples)')
    p.add_argument('--step', type=int, default=4000, help='Bước trượt (samples)')
    # Group split theo file
    p.add_argument('--group-split', action='store_true', help='Tách train/test theo file (GroupShuffleSplit) chống leakage')
    # Feature selector
    p.add_argument('--fs', type=str, default='anova', choices=['anova', 'mi'], help='Chọn bộ chọn đặc trưng: anova|mi')
    # Nested CV
    p.add_argument('--nested', action='store_true', help='Bật Nested CV (đánh giá chặt hơn, chậm hơn)')
    # Early-stop theo ngưỡng
    p.add_argument('--target-acc', type=float, default=0.0, help='Ngưỡng Test Accuracy để dừng sớm (0 = tắt)')
    p.add_argument('--max-tries', type=int, default=50, help='Số seed tối đa để thử khi dùng --target-acc')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = "run_artifacts"
    os.makedirs(out_dir, exist_ok=True)

    base_cmd = (
        f"python {os.path.basename(__file__)}"
        f" --data {args.data or './dataset'} --test-size {args.test_size}"
        f" --kernel {args.kernel} {'--grid' if args.grid else ''}"
        f" --scoring {args.scoring} --cv-splits {args.cv_splits} --cv-repeats {args.cv_repeats}"
        f"{' --sliding' if args.sliding else ''} --win {args.win} --step {args.step}"
        f"{' --group-split' if args.group_split else ''}"
        f" --fs {args.fs}"
        f"{' --nested' if args.nested else ''}"
    )

    print("sEMG Muscle Fatigue Classification using SVM (Overfitting-safe + Sliding Window + Group Split)")
    print("=" * 60)

    # Early stop mode
    if args.target_acc and args.target_acc > 0:
        seed, te, auc_score = train_until_target(
            target_acc=args.target_acc,
            max_tries=args.max_tries,
            data_path=args.data,
            test_size=args.test_size,
            kernel=args.kernel,
            use_grid_search=args.grid,
            scoring=args.scoring,
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
            n_runs=args.repeat,
            data_path=args.data,
            test_size=args.test_size,
            kernel=args.kernel,
            use_grid_search=args.grid,
            scoring=args.scoring,
            cv_splits=args.cv_splits, cv_repeats=args.cv_repeats,
            sliding=args.sliding, win=args.win, step=args.step,
            group_split=args.group_split, fs_kind=args.fs
        )
        return

    # Single run
    logger = RunLogger(out_dir=out_dir)
    logger.set_args({
        "mode": "single_run",
        "data_path": args.data, "test_size": args.test_size,
        "kernel": args.kernel, "use_grid_search": args.grid,
        "scoring": args.scoring, "cv_splits": args.cv_splits, "cv_repeats": args.cv_repeats,
        "sliding": args.sliding, "win": args.win, "step": args.step,
        "group_split": args.group_split, "feature_selector": args.fs,
        "nested": args.nested
    })

    seed = 42
    clf = sEMGSVMClassifier(data_path=args.data, run_logger=logger, seed=seed)
    clf.prepare_data(test_size=args.test_size, random_state=seed, out_dir=out_dir,
                     sliding=args.sliding, win=args.win, step=args.step,
                     group_split=args.group_split, fs_kind=args.fs)
    clf.train_svm(kernel=args.kernel, use_grid_search=args.grid,
                  scoring=args.scoring, cv_splits=args.cv_splits,
                  cv_repeats=args.cv_repeats, nested=args.nested, fs_kind=args.fs)
    train_acc, test_acc, auc_score = clf.evaluate_model(save_prefix='SVM', out_dir=out_dir)

    # so sánh kernel (không ảnh hưởng model đã lưu)
    clf.compare_kernels(out_dir=out_dir, scoring=args.scoring,
                        cv_splits=args.cv_splits, cv_repeats=args.cv_repeats, fs_kind=args.fs)

    # lưu pipeline đã fit & logs
    clf.save_model(os.path.join(out_dir, 'best_svm_model.pkl'))
    logger.write()
    logger.write_repro_shell(base_cmd)

    print("\n" + "=" * 60)
    print("SVM Classification Complete!")
    print(f"Best Test Accuracy: {test_acc:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
