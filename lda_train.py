#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train & evaluate LDA for sEMG fatigue classification using the SAME pipeline as SVM/KNN.

- Tái sử dụng sEMGSVMClassifier cho toàn bộ tiền xử lý/đặc trưng.
- GridSearchCV trên solver/shrinkage.
- Lưu model dict {'model','scaler','feature_selector'} => best_lda_model.pkl
- Xuất Confusion Matrix (LDA_Confusion_Matrix.png) và ROC (LDA_ROC_Curve.png)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc
)
import joblib

# dùng lại class train gốc để giữ đúng pipeline đặc trưng
from sEMG_SVM_Classification import sEMGSVMClassifier

def main():
    print("=== LDA Training for sEMG Fatigue Classification ===")

    # 1) Chuẩn bị dữ liệu như SVM/KNN (giữ scaler + selector giống nhau)
    clf = sEMGSVMClassifier()
    clf.prepare_data()  # tạo clf.X_train_selected, clf.X_test_selected, clf.y_train, clf.y_test

    # 2) GridSearch cho LDA
    # - 'svd' không hỗ trợ shrinkage
    # - 'lsqr' và 'eigen' hỗ trợ shrinkage ('auto' hoặc float)
    param_grid = [
        {"solver": ["svd"]},
        {"solver": ["lsqr"],  "shrinkage": ["auto", None]},
        {"solver": ["eigen"], "shrinkage": ["auto", None]},
    ]

    base_lda = LDA()
    grid = GridSearchCV(
        base_lda,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(clf.X_train_selected, clf.y_train)

    best_lda = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV acc: {grid.best_score_:.4f}")

    # 3) Cross-val trên train để báo cáo
    cv_scores = cross_val_score(best_lda, clf.X_train_selected, clf.y_train, cv=5)
    print(f"Cross-val accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # 4) Đánh giá train/test
    ytr_pred = best_lda.predict(clf.X_train_selected)
    yte_pred = best_lda.predict(clf.X_test_selected)
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
    plt.title("Confusion Matrix - LDA")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("LDA_Confusion_Matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 6) ROC/AUC
    if hasattr(best_lda, "predict_proba"):
        y_score = best_lda.predict_proba(clf.X_test_selected)[:, 1]
    else:
        # LDA chuẩn có predict_proba; phòng hờ:
        y_score = best_lda.decision_function(clf.X_test_selected)
    fpr, tpr, _ = roc_curve(clf.y_test, y_score)
    auc_val = auc(fpr, tpr)

    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc_val:.3f})")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - LDA")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("LDA_ROC_Curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 7) Lưu model dict theo đúng format đang dùng
    joblib.dump({
        "model": best_lda,
        "scaler": clf.scaler,
        "feature_selector": clf.feature_selector
    }, "best_lda_model.pkl")
    print("Saved: best_lda_model.pkl, LDA_Confusion_Matrix.png, LDA_ROC_Curve.png")

    print("\n=== Done LDA training ===")

if __name__ == "__main__":
    main()
