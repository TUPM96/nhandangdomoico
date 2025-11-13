# Quick Start Guide

## 🚀 Chạy ngay trong 3 bước

### Bước 1: Cài đặt dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib scipy
```

### Bước 2: Verify setup
```bash
python verify_setup.py
```

Bạn sẽ thấy:
```
✓ ALL CHECKS PASSED!
✓ System is ready to use
```

### Bước 3: Chạy demo
```bash
python demo_predict.py
```

Kết quả sẽ hiển thị predictions với **91.07% accuracy**!

---

## 📊 Xem kết quả ngay

### 1. Xem confusion matrices
```bash
cd plots_final
ls -la  # 3 confusion matrix images
```

### 2. Xem model comparison
```bash
cat models_final/model_comparison.csv
```

Output:
```
Model,Accuracy,Precision,Recall,F1-Score
SVM,91.07%,90.31%,92.00%,91.15%
LDA,90.27%,89.74%,90.93%,90.33%
KNN,86.93%,95.11%,77.87%,85.63%
```

### 3. Load và sử dụng model
```python
import joblib
import pandas as pd

# Load best model (SVM - 91.07%)
model = joblib.load('models_final/svm_model.pkl')

# Load test data
test_data = pd.read_csv('data_amplified_final/test_data.csv')
X_test = test_data.drop('label', axis=1)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(f"Accuracy on test set: {(predictions == test_data['label']).mean():.2%}")
```

---

## 🔄 Chạy lại toàn bộ pipeline

Nếu muốn train lại từ đầu:

```bash
python run_full_pipeline.py
```

Pipeline sẽ:
1. ✅ Generate 3000 synthetic samples từ 52 EMG files gốc
2. ✅ Train 3 models (LDA, KNN, SVM) với GridSearchCV
3. ✅ Test và evaluate
4. ✅ Lưu results vào `models_final/` và `plots_final/`

⏱️ Thời gian: ~5-10 phút (tùy CPU)

---

## 📖 Tài liệu chi tiết

- **README.md**: Hướng dẫn đầy đủ
- **SUCCESS_SUMMARY.md**: Chi tiết về solution
- **ANSWERS_QUESTIONS.md**: Trả lời câu hỏi kỹ thuật (53KB!)

---

## 🎯 Kết quả đạt được

| Metric | SVM (Best) | LDA | KNN |
|--------|-----------|-----|-----|
| Accuracy | **91.07%** | 90.27% | 86.93% |
| Precision | 90.31% | 89.74% | 95.11% |
| Recall | 92.00% | 90.93% | 77.87% |
| F1-Score | 91.15% | 90.33% | 85.63% |

✅ **Target: 85-95%** → Đạt 91.07% với SVM!

---

## ⚡ Troubleshooting nhanh

**Lỗi import?**
```bash
pip install -r requirements_new.txt
```

**File không tìm thấy?**
```bash
python verify_setup.py  # Kiểm tra setup
```

**Model không load?**
```bash
python train_models.py  # Train lại
```

---

**✅ Hệ thống sẵn sàng! Bắt đầu với `python demo_predict.py`**
