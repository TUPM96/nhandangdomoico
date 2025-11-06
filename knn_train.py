#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train & evaluate KNN for sEMG fatigue classification using the SAME pipeline as SVM.

- Reuses sEMGSVMClassifier for preprocessing/feature extraction.
- GridSearchCV over K, weights, metric (p=1/2).
- Saves model dict {'model','scaler','feature_selector'} => best_knn_model.pkl
- Saves Confusion Matrix (KNN_Confusion_Matrix.png) and ROC (KNN_ROC_Curve.png)
- (Optional) Saves CV curve over K (KNN_K_Comparison.png) if p=2, uniform vs distance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import joblib

# import your existing training class to reuse preprocessing/feature pipeline
from sEMG_SVM_Classification import sEMGSVMClassifier

def main():
    print("=== KNN Training for sEMG Fatigue Classification ===")

    # 1) Chuẩn bị dữ liệu y như SVM (giữ scaler + selector giống nhau)
    clf = sEMGSVMClassifier()
    clf.prepare_data()  # tạo clf.X_train_selected, clf.X_test_selected, clf.y_train, clf.y_test

    # 2) GridSearch cho KNN
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski"],
        "p": [1, 2],    # p=1 (Manhattan), p=2 (Euclidean)
    }
    base_knn = KNeighborsClassifier()
    grid = GridSearchCV(
        base_knn,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(clf.X_train_selected, clf.y_train)

    best_knn = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV acc: {grid.best_score_:.4f}")

    # 3) Cross-val (lại) trên train để báo cáo
    cv_scores = cross_val_score(best_knn, clf.X_train_selected, clf.y_train, cv=5)
    print(f"Cross-val accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # 4) Đánh giá trên train/test
    ytr_pred = best_knn.predict(clf.X_train_selected)
    yte_pred = best_knn.predict(clf.X_test_selected)
    acc_tr = accuracy_score(clf.y_train, ytr_pred)
    acc_te = accuracy_score(clf.y_test, yte_pred)
    print(f"Train Accuracy: {acc_tr:.4f}")
    print(f"Test  Accuracy: {acc_te:.4f}")
    print("\nClassification report (Test):")
    print(classification_report(clf.y_test, yte_pred, target_names=["Non-Fatigue","Fatigue"]))

    # 5) Confusion matrix
    cm = confusion_matrix(clf.y_test, yte_pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Fatigue","Fatigue"],
                yticklabels=["Non-Fatigue","Fatigue"])
    plt.title("Confusion Matrix - KNN")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("KNN_Confusion_Matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 6) ROC/AUC
    if hasattr(best_knn, "predict_proba"):
        y_score = best_knn.predict_proba(clf.X_test_selected)[:, 1]
    else:
        # KNN có predict_proba, nhưng để chắc chắn:
        y_score = best_knn.kneighbors(clf.X_test_selected, return_distance=True)[0]
        y_score = -y_score.mean(axis=1)  # proxy score; ít dùng
    fpr, tpr, _ = roc_curve(clf.y_test, y_score)
    auc_val = auc(fpr, tpr)

    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc_val:.3f})")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - KNN")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("KNN_ROC_Curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 7) Lưu model dict theo đúng format đang dùng
    joblib.dump({
        "model": best_knn,
        "scaler": clf.scaler,
        "feature_selector": clf.feature_selector
    }, "best_knn_model.pkl")
    print("Saved: best_knn_model.pkl, KNN_Confusion_Matrix.png, KNN_ROC_Curve.png")

    # 8) (Tuỳ chọn) Vẽ CV theo K để báo cáo (chỉ lọc p=2 cho gọn)
    try:
        results = grid.cv_results_
        # Lấy các hàng có p=2 (Euclidean)
        mask = [p == 2 for p in results["param_p"].data]
        Ks = np.array(results["param_n_neighbors"].data)[mask]
        Weights = np.array(results["param_weights"].data)[mask]
        Means = np.array(results["mean_test_score"].data)[mask]

        # Vẽ 2 đường: uniform vs distance
        plt.figure(figsize=(7,5))
        for w in ["uniform","distance"]:
            m = (Weights == w)
            # group by K
            pairs = sorted(set(Ks[m]))
            yvals = [Means[m & (Ks==k)].mean() for k in pairs]
            plt.plot(pairs, yvals, marker="o", label=w)
        plt.xlabel("n_neighbors (K)")
        plt.ylabel("CV accuracy")
        plt.title("KNN CV Accuracy vs K (p=2)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("KNN_K_Comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved: KNN_K_Comparison.png")
    except Exception:
        pass

    print("\n=== Done KNN training ===")

if __name__ == "__main__":
    main()
