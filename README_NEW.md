# Hệ Thống Nhận Dạng Mỏi Cơ - LDA, KNN, SVM

Hệ thống AI nhận dạng mỏi cơ sử dụng 3 phương pháp Machine Learning: **LDA** (Linear Discriminant Analysis), **KNN** (K-Nearest Neighbors), và **SVM** (Support Vector Machine).

## 🎯 Mục Tiêu

Xây dựng hệ thống nhận dạng mỏi cơ với độ chính xác **85-95%** trên test set.

## 📋 Yêu Cầu Hệ Thống

- Python 3.7+
- Các thư viện trong `requirements_new.txt`

## 🚀 Cài Đặt

### 1. Clone repository (hoặc tải về)

```bash
git clone <repository-url>
cd nhandangdomoico
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements_new.txt
```

## 📊 Dữ Liệu

Hệ thống sử dụng 10 features để nhận dạng mỏi cơ:

| Feature | Mô tả | Đơn vị |
|---------|-------|--------|
| `emg_rms` | Root Mean Square của tín hiệu EMG | mV |
| `emg_mav` | Mean Absolute Value của tín hiệu EMG | mV |
| `emg_median_freq` | Tần số trung vị của tín hiệu EMG | Hz |
| `emg_mean_freq` | Tần số trung bình của tín hiệu EMG | Hz |
| `muscle_force` | Lực cơ | N (Newton) |
| `heart_rate` | Nhịp tim | bpm |
| `work_duration` | Thời gian làm việc | phút |
| `rest_time` | Thời gian nghỉ ngơi | phút |
| `movement_frequency` | Tần số chuyển động | lần/phút |
| `muscle_tension` | Độ căng cơ | 0-100 |

**2 Classes:**
- `0`: Non-Fatigue (Không mỏi)
- `1`: Fatigue (Mỏi)

## 🔧 Cách Sử Dụng

### Phương Pháp 1: Chạy Toàn Bộ Pipeline (Khuyến Nghị)

Chạy từ generate data → train → test trong một lệnh duy nhất:

```bash
python run_full_pipeline.py
```

**Tùy chọn nâng cao:**

```bash
# Tạo 3000 mẫu, test size 30%, sử dụng GridSearchCV
python run_full_pipeline.py --n-samples 3000 --test-size 0.3

# Train nhanh (không dùng GridSearchCV)
python run_full_pipeline.py --no-grid-search

# Thay đổi random seed
python run_full_pipeline.py --seed 123
```

### Phương Pháp 2: Chạy Từng Bước

#### Bước 1: Generate Data

```bash
python generate_data.py
```

Tạo ra:
- `data_generated/train_data.csv` (1500 mẫu)
- `data_generated/test_data.csv` (500 mẫu)
- `data_generated/full_data.csv` (2000 mẫu)

#### Bước 2: Train Models

```bash
python train_models.py
```

Train cả 3 models (LDA, KNN, SVM) và lưu vào thư mục `models/`:
- `lda_model.pkl`
- `knn_model.pkl`
- `svm_model.pkl`
- `model_comparison.csv`
- `all_results.json`

Confusion matrices được lưu trong `plots/`.

#### Bước 3: Test Models

```bash
# Test tất cả models
python test_models.py

# Test một model cụ thể
python test_models.py --model svm
python test_models.py --model knn
python test_models.py --model lda

# Chỉ định path
python test_models.py --test-data data_generated/test_data.csv --models-dir models
```

## 📈 Kết Quả Mong Đợi

Sau khi chạy, bạn sẽ thấy:

```
SO SÁNH KẾT QUẢ TEST
============================================================
           Accuracy  Precision    Recall  F1-Score
lda        0.9120     0.9180    0.9050    0.9115
knn        0.8980     0.9020    0.8940    0.8980
svm        0.9340     0.9390    0.9290    0.9340

✓ Model tốt nhất: SVM
✓ Accuracy: 0.9340 (93.40%)

✓✓✓ ĐẠT MỤC TIÊU! Accuracy >= 85% ✓✓✓
```

## 📁 Cấu Trúc Thư Mục

```
.
├── generate_data.py          # Script tạo synthetic data
├── train_models.py            # Script train models
├── test_models.py             # Script test models
├── run_full_pipeline.py       # Script chạy toàn bộ pipeline
├── requirements_new.txt       # Dependencies
├── README_NEW.md              # Tài liệu này
│
├── data_generated/            # Dữ liệu được tạo
│   ├── train_data.csv
│   ├── test_data.csv
│   └── full_data.csv
│
├── models/                    # Models đã train
│   ├── lda_model.pkl
│   ├── knn_model.pkl
│   ├── svm_model.pkl
│   ├── model_comparison.csv
│   └── all_results.json
│
├── plots/                     # Confusion matrices
│   ├── lda_confusion_matrix.png
│   ├── knn_confusion_matrix.png
│   └── svm_confusion_matrix.png
│
└── test_results/              # Kết quả test
    ├── test_comparison.csv
    └── models_comparison.png
```

## 🔍 Chi Tiết Models

### 1. LDA (Linear Discriminant Analysis)

**Ưu điểm:**
- Nhanh, hiệu quả
- Tốt với dữ liệu tuyến tính
- Giảm chiều dữ liệu tự động

**Hyperparameters được tune:**
- `solver`: svd, lsqr, eigen
- `shrinkage`: None, auto, 0.1, 0.5, 0.9

### 2. KNN (K-Nearest Neighbors)

**Ưu điểm:**
- Đơn giản, dễ hiểu
- Không cần training phase
- Tốt với decision boundaries phức tạp

**Hyperparameters được tune:**
- `n_neighbors`: 3, 5, 7, 9, 11, 15
- `weights`: uniform, distance
- `metric`: euclidean, manhattan, minkowski

### 3. SVM (Support Vector Machine)

**Ưu điểm:**
- Hiệu quả với high-dimensional data
- Tốt với margin rõ ràng
- Sử dụng kernel trick cho non-linear problems

**Hyperparameters được tune:**
- `C`: 0.1, 1, 10, 100
- `kernel`: rbf, linear, poly
- `gamma`: scale, auto, 0.001, 0.01, 0.1, 1

## 🎛️ Tùy Chỉnh

### Tạo nhiều dữ liệu hơn

Sửa trong `generate_data.py`:

```python
train_df, test_df, full_df = save_train_test_data(
    output_dir='data_generated',
    n_samples=5000,  # Tăng lên 5000
    test_size=0.25,
    seed=42
)
```

### Thay đổi Hyperparameters

Sửa trong `train_models.py`, hàm `get_param_grid()`:

```python
def get_param_grid(self):
    if self.model_type == 'svm':
        return {
            'C': [0.1, 1, 10, 100, 1000],  # Thêm giá trị
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01]
        }
```

### Thêm Features Mới

Sửa trong `generate_data.py`, hàm `generate_fatigue_muscle_data()`:

```python
non_fatigue_data = {
    # ... features hiện có ...
    'new_feature': np.random.normal(50, 10, n_non_fatigue),
}
```

## 🐛 Troubleshooting

### Lỗi: Module not found

```bash
pip install -r requirements_new.txt
```

### Accuracy thấp hơn 85%

1. Tăng số lượng samples:
   ```bash
   python run_full_pipeline.py --n-samples 5000
   ```

2. Đảm bảo GridSearchCV được bật (mặc định)

3. Thử các random seeds khác:
   ```bash
   python run_full_pipeline.py --seed 123
   python run_full_pipeline.py --seed 456
   ```

### Training quá chậm

1. Tắt GridSearchCV:
   ```bash
   python run_full_pipeline.py --no-grid-search
   ```

2. Giảm số lượng samples:
   ```bash
   python run_full_pipeline.py --n-samples 1000
   ```

## 📊 Sử Dụng Model Đã Train

```python
from train_models import FatigueMuscleClassifier
import numpy as np

# Load model
classifier = FatigueMuscleClassifier.load_model('models/svm_model.pkl')

# Dữ liệu mẫu (1 sample với 10 features)
sample = np.array([[
    0.25,  # emg_rms
    0.20,  # emg_mav
    65,    # emg_median_freq
    70,    # emg_mean_freq
    35,    # muscle_force
    95,    # heart_rate
    40,    # work_duration
    3,     # rest_time
    12,    # movement_frequency
    70     # muscle_tension
]])

# Transform và predict
sample_scaled = classifier.scaler.transform(sample)
prediction = classifier.model.predict(sample_scaled)[0]

print(f"Prediction: {'Fatigue' if prediction == 1 else 'Non-Fatigue'}")
```

## 📝 License

MIT License

## 👥 Authors

Hệ thống nhận dạng mỏi cơ - AI/ML Project

## 🙏 Acknowledgments

- scikit-learn documentation
- EMG signal processing research papers
- Machine learning best practices

---

**Chúc bạn thành công! 🎉**

Nếu có vấn đề, hãy kiểm tra logs hoặc mở issue.
