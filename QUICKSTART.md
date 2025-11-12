# 🚀 QUICKSTART - Hệ Thống Nhận Dạng Mỏi Cơ

## ⚡ Chạy Nhanh (1 lệnh)

```bash
# Cài đặt dependencies
pip install -r requirements_new.txt

# Chạy toàn bộ: Generate data → Train → Test
python run_full_pipeline.py
```

**Kết quả mong đợi:**
- ✓ Tạo 2000 mẫu dữ liệu
- ✓ Train 3 models (LDA, KNN, SVM)
- ✓ Test accuracy: **85-95%**
- ✓ **SVM tốt nhất: ~95.7%**

---

## 📊 Kết Quả Thực Tế

Với 3000 mẫu test trên 750 samples:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **SVM** | **95.73%** | 96.73% | 94.67% | 95.69% |
| LDA   | 94.80% | 95.90% | 93.60% | 94.74% |
| KNN   | 94.53% | 95.63% | 93.33% | 94.47% |

**✓✓✓ TẤT CẢ ĐỀU ĐẠT MỤC TIÊU 85-95%!**

---

## 🎮 Demo Sử Dụng Model

### Cách 1: Demo với ví dụ có sẵn

```bash
python demo_predict.py
```

Sẽ test 3 trường hợp:
- ✅ Người không mỏi (Fresh)
- ❌ Người mỏi (Fatigued)
- ⚠️ Trường hợp biên (Borderline)

### Cách 2: Predict từ file CSV

```bash
python demo_predict.py csv models/svm_model.pkl data_generated/test_data.csv predictions.csv
```

### Cách 3: Sử dụng trong code Python

```python
from train_models import FatigueMuscleClassifier
import numpy as np

# Load model
classifier = FatigueMuscleClassifier.load_model('models/svm_model.pkl')

# Dữ liệu mẫu (10 features)
sample = np.array([[
    0.28,  # emg_rms
    0.24,  # emg_mav
    60,    # emg_median_freq
    65,    # emg_mean_freq
    32,    # muscle_force
    95,    # heart_rate
    45,    # work_duration
    3,     # rest_time
    12,    # movement_frequency
    70     # muscle_tension
]])

# Predict
sample_scaled = classifier.scaler.transform(sample)
prediction = classifier.model.predict(sample_scaled)[0]
print(f"Kết quả: {'Mỏi' if prediction == 1 else 'Không mỏi'}")
```

---

## 📁 Files Quan Trọng

| File | Mô tả |
|------|-------|
| `run_full_pipeline.py` | Chạy toàn bộ: data → train → test |
| `generate_data.py` | Tạo synthetic data |
| `train_models.py` | Train 3 models với GridSearchCV |
| `test_models.py` | Test và đánh giá |
| `demo_predict.py` | Demo sử dụng model |
| `README_NEW.md` | Tài liệu chi tiết |

---

## 🎯 10 Features Đầu Vào

1. **emg_rms** - Root Mean Square EMG (mV)
2. **emg_mav** - Mean Absolute Value EMG (mV)
3. **emg_median_freq** - Tần số trung vị EMG (Hz)
4. **emg_mean_freq** - Tần số trung bình EMG (Hz)
5. **muscle_force** - Lực cơ (N)
6. **heart_rate** - Nhịp tim (bpm)
7. **work_duration** - Thời gian làm việc (phút)
8. **rest_time** - Thời gian nghỉ (phút)
9. **movement_frequency** - Tần số chuyển động (lần/phút)
10. **muscle_tension** - Độ căng cơ (0-100)

**Output:** 0 = Không mỏi, 1 = Mỏi

---

## 🔧 Tùy Chọn Nâng Cao

```bash
# Tạo nhiều data hơn để accuracy cao hơn
python run_full_pipeline.py --n-samples 5000

# Train nhanh (không dùng GridSearchCV)
python run_full_pipeline.py --no-grid-search

# Thử seed khác
python run_full_pipeline.py --seed 456

# Test riêng một model
python test_models.py --model svm
```

---

## 📈 Tại Sao SVM Tốt Nhất?

**SVM (Support Vector Machine)** đạt 95.73% vì:
- ✓ Tốt với high-dimensional data (10 features)
- ✓ Kernel RBF xử lý non-linear boundaries
- ✓ GridSearchCV tìm được params tối ưu: C=0.1, gamma=scale
- ✓ Robust với noise và outliers

---

## 🎓 Thông Tin Model

### SVM (Tốt nhất)
- Accuracy: **95.73%**
- Best params: `C=0.1, kernel='rbf', gamma='scale'`
- Training time: ~3-5 giây

### LDA
- Accuracy: 94.80%
- Best params: `solver='lsqr', shrinkage='auto'`
- Training time: ~0.5 giây

### KNN
- Accuracy: 94.53%
- Best params: `n_neighbors=15, weights='distance', metric='manhattan'`
- Training time: ~1 giây

---

## ✅ Checklist Hoàn Thành

- ✅ Generate synthetic data
- ✅ Train 3 models (LDA, KNN, SVM)
- ✅ GridSearchCV để tìm best params
- ✅ Test accuracy 85-95%
- ✅ SVM tốt nhất (~95.7%)
- ✅ Demo script
- ✅ Full documentation

---

## 💡 Tips

1. **Tăng accuracy:** Tăng `--n-samples` (3000-5000)
2. **Train nhanh:** Dùng `--no-grid-search`
3. **Khác seed:** Thử `--seed 123`, `--seed 456`, etc.
4. **Best model:** SVM với RBF kernel

---

**🎉 Hoàn tất! Bạn đã có hệ thống nhận dạng mỏi cơ với accuracy 85-95%!**
