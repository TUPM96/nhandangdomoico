# Hướng dẫn chạy 3 giải thuật: SVM, LDA, KNN (sử dụng CLI tương tự)

Tài liệu ngắn này mô tả cách chạy 3 thuật toán SVM, LDA và KNN cho bài toán phân lớp mệt mỏi cơ (sEMG) sử dụng cùng phong cách CLI như ví dụ bạn đã có (sEMG_KNN.py). Mục tiêu là:
- Dùng cùng pipeline (StandardScaler -> SelectKBest -> classifier)
- Hỗ trợ GridSearchCV (khi bật `--grid`)
- Hỗ trợ sliding windows, group-aware split (`--group-split`), feature selector (`--fs anova|mi`)
- Hỗ trợ early-stop theo `--target-acc` và `--max-tries`

---

## 1. Yêu cầu (cài đặt)
Cài các thư viện cần thiết (nếu chưa có):
```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib
```

---

## 2. Cách dùng (CLI mẫu)
Cú pháp ví dụ chung (sử dụng `sEMG_SVM.py` làm tên script ví dụ — có thể là `sEMG_KNN.py`/`sEMG_LDA.py` hoặc một script hợp nhất):
```bash
# SVM: chạy grid-search, sliding, group split, feature selector = mutual_info, dừng sớm khi TestAcc >= 0.85 (max 50 tries)
python sEMG_SVM.py  --data ./dataset --grid --sliding --win 8000 --step 4000 --group-split --fs mi --target-acc 0.85 --max-tries 50

# LDA: 
python sEMG_LDA.py --data ./dataset --grid --sliding --win 8000 --step 4000 --group-split --fs mi --target-acc 0.85 --max-tries 50

# KNN: 
python sEMG_KNN.py --data ./dataset --grid --sliding --win 8000 --step 4000 --group-split --fs mi --target-acc 0.85 --max-tries 50
```

Các tham số quan trọng:
- --data: thư mục chứa `fatigue/` và `non fatigue/`
- --grid: bật GridSearchCV
- --sliding --win --step: bật sliding window
- --group-split: tách theo file (GroupShuffleSplit) để tránh data leakage từ sliding windows
- --fs: feature selector (`anova` hoặc `mi`)
- --target-acc, --max-tries: chế độ early-stop (chạy nhiều seed đến khi đạt hoặc hết tries)

---


Ghi chú:
- Ở pipeline ta đặt classifier bước cuối là 'clf' để dễ map tới param grid (ví dụ `clf__C`).
- SVM dùng `probability=True` để có `predict_proba` (dùng cho ROC/AUC). Lưu ý: training chậm hơn.
- LDA không hỗ trợ `predict_proba` cho một vài solver; thường LDA có `predict_proba` nếu phù hợp.


## 5. Gợi ý grid & chiến lược
- SVM (RBF) thường hay đạt tốt nhưng chậm khi grid lớn. Bắt đầu với C in [0.1,1,10], kernel rbf/linear.
- LDA: ít tham số, nhanh, tốt khi features có phân biệt tuyến tính.
- KNN: tune n_neighbors, weights, metric; có thể tune kbest__k cùng GridSearch.

---

## 6. Chạy nhiều seed / early-stop
Script mẫu bạn cho sẵn đã có hai hàm hữu ích:
- `train_until_target(target_acc, max_tries, ...)` — chạy nhiều seed stop khi đạt target.
- `repeat_runs(n_runs, ...)` — chạy lặp và lưu CSV.

Sử dụng `--target-acc 0.85 --max-tries 50` để tự động dừng khi có seed đạt TestAcc >= 0.85.

---

## 7. Output mong đợi
Sau khi chạy thành công (ví dụ với `--grid`):
- Thư mục `run_artifacts_*` chứa manifest JSON, confusion matrix, ROC, model (pkl), file hardcode params (nếu export).
- File `best_*_model.pkl` (bundle pipeline + export)
- Báo cáo classification_report in console

---

## 8. Một vài lưu ý vận hành
- Nếu dataset nhỏ, GridSearchCV + repeated CV có thể rất chậm. Giảm `cv-splits` hoặc tập mẫu thông số trước.
- Khi dùng sliding window, luôn bật `--group-split` nếu bạn muốn tránh leakage (một file cung cấp nhiều cửa sổ).
- Đảm bảo `--fs mi` hoặc `--fs anova` phù hợp: MI (mutual_info) thường phù hợp với quan hệ không tuyến tính.
