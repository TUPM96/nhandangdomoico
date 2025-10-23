# sEMG Muscle Fatigue Classification using SVM

Đây là implementation của Support Vector Machine (SVM) để phân loại tình trạng mệt mỏi cơ bắp dựa trên tín hiệu Surface Electromyography (sEMG).

## Tổng quan

Model SVM này được phát triển để bổ sung cho các model CNN và ResNet50 đã có trong project, sử dụng cùng bộ dữ liệu sEMG để phân loại trạng thái muscle fatigue và non-fatigue.

## Tính năng chính

### 1. Xử lý tín hiệu sEMG
- **Zero baseline correction**: Loại bỏ DC offset
- **Signal rectification**: Chuyển đổi tín hiệu thành giá trị tuyệt đối
- **Normalization**: Chuẩn hóa dữ liệu về khoảng [0,1]
- **Butterworth filtering**: Lọc bandpass (10-100 Hz)

### 2. Trích xuất đặc trưng (Feature Extraction)
#### Time Domain Features:
- Mean, Standard deviation, Variance
- Maximum, Minimum, Median
- Quartiles (Q1, Q3)
- Root Mean Square (RMS)
- Mean Absolute Deviation
- Zero crossings count
- Total variation
- Skewness và Kurtosis

#### Frequency Domain Features:
- Power spectral density statistics
- Frequency band powers (Low: 10-30Hz, Mid: 30-70Hz, High: 70-100Hz)
- Spectral centroid
- STFT-based features

### 3. Model SVM
- **Kernel options**: Linear, RBF, Polynomial, Sigmoid
- **Hyperparameter tuning**: Grid Search với Cross-validation
- **Feature selection**: SelectKBest để chọn top features
- **Standardization**: StandardScaler cho feature scaling

### 4. Đánh giá model
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix visualization
- ROC Curve và AUC score
- Cross-validation scores
- Kernel comparison

## Cách sử dụng

### 1. Cài đặt dependencies
```bash
pip install -r requirements_svm.txt
```

### 2. Chạy classification
```bash
python run_svm.py
```

Hoặc sử dụng trực tiếp:
```python
from sEMG_SVM_Classification import sEMGSVMClassifier

# Khởi tạo classifier
classifier = sEMGSVMClassifier(data_path='dataset')

# Chuẩn bị dữ liệu
classifier.prepare_data()

# Huấn luyện model
classifier.train_svm(kernel='rbf', use_grid_search=True)

# Đánh giá model
classifier.evaluate_model()

# So sánh các kernel
classifier.compare_kernels()

# Lưu model
classifier.save_model('my_svm_model.pkl')
```

### 3. Load model đã lưu
```python
classifier = sEMGSVMClassifier()
classifier.load_model('my_svm_model.pkl')
```

## Cấu trúc dữ liệu

Đảm bảo cấu trúc thư mục như sau:
```
dataset/
├── fatigue/
│   ├── aida_F.csv
│   ├── aldhito_F.csv
│   └── ... (other fatigue files)
└── non fatigue/
    ├── aida_NF.csv
    ├── aldhito_NF.csv
    └── ... (other non-fatigue files)
```

Mỗi file CSV phải có cột `amplitudo` chứa tín hiệu sEMG.

## Kết quả mong đợi

- **Accuracy**: Thường đạt 85-95% tùy thuộc vào kernel và parameters
- **Best kernel**: RBF kernel thường cho kết quả tốt nhất
- **Feature importance**: Time domain features thường quan trọng hơn frequency domain
- **Cross-validation**: Đảm bảo model không bị overfitting

## Files được tạo

### Output files:
- `SVM_Confusion_Matrix.png`: Confusion matrix visualization
- `SVM_ROC_Curve.png`: ROC curve plot
- `SVM_Kernel_Comparison.png`: So sánh các kernel
- `best_svm_fatigue_model.pkl`: Model đã huấn luyện

### Code files:
- `sEMG_SVM_Classification.py`: Main classifier class
- `run_svm.py`: Script chạy classification
- `requirements_svm.txt`: Dependencies list

## So sánh với CNN/ResNet50

| Aspect | SVM | CNN/ResNet50 |
|--------|-----|--------------|
| **Input** | Hand-crafted features | Raw spectrograms |
| **Processing** | Feature engineering required | Automatic feature learning |
| **Training time** | Nhanh | Chậm hơn |
| **Interpretability** | Cao (có thể phân tích features) | Thấp |
| **Data requirement** | Ít hơn | Nhiều hơn |
| **Accuracy** | 85-95% | 90-98% |

## Ưu điểm của SVM approach

1. **Interpretability**: Có thể hiểu được features quan trọng
2. **Robust**: Ít bị overfitting với dataset nhỏ
3. **Fast training**: Huấn luyện nhanh hơn deep learning
4. **Memory efficient**: Không cần GPU, ít RAM
5. **Feature analysis**: Có thể phân tích importance của các features

## Hạn chế

1. **Feature engineering**: Cần thiết kế features thủ công
2. **Scalability**: Khó mở rộng với dataset rất lớn
3. **Complex patterns**: Khó học được patterns phức tạp như deep learning

## Tùy chỉnh

Bạn có thể tùy chỉnh:
- Thay đổi frequency bands trong `extract_frequency_domain_features()`
- Thêm features mới trong `extract_time_domain_features()`
- Điều chỉnh hyperparameter grid trong `train_svm()`
- Thay đổi feature selection method

## Troubleshooting

1. **Memory error**: Giảm số lượng features hoặc samples
2. **Slow training**: Giảm parameter grid hoặc tắt grid search
3. **Low accuracy**: Thử kernel khác hoặc thêm features
4. **File not found**: Kiểm tra đường dẫn dataset

## Tác giả

Developed by AI Assistant for sEMG Muscle Fatigue Classification Project
Date: September 2025
