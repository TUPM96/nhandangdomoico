# 🧹 HƯỚNG DẪN CLEAN SOURCE CODE

## 📁 CẤU TRÚC MỚI (SAU KHI CLEAN)

```
nhandangdomoico/
├── 📂 dataset/                      # Dataset gốc (GIỮ LẠI)
│   ├── fatigue/
│   └── non fatigue/
│
├── 📂 src/                          # Source code MỚI
│   ├── generate_data.py            # ⭐ Generate synthetic data
│   ├── train_models.py             # ⭐ Train 3 models
│   ├── test_models.py              # ⭐ Test models
│   ├── run_full_pipeline.py        # ⭐ Full pipeline
│   ├── demo_predict.py             # ⭐ Demo
│   └── extract_features.py         # ⭐ Extract từ dataset gốc
│
├── 📂 docs/                         # Documentation
│   ├── README_NEW.md               # ⭐ Hướng dẫn chính
│   ├── QUICKSTART.md               # ⭐ Quick start
│   └── ANSWERS_QUESTIONS.md        # ⭐ Trả lời câu hỏi
│
├── 📄 requirements_new.txt          # ⭐ Dependencies
├── 📄 .gitignore                    # Git ignore
└── 📄 CLEANUP_GUIDE.md              # File này
```

---

## ❌ CÁC FILE CẦN XÓA (CŨ, KHÔNG DÙNG)

### 1. **Python Scripts Cũ** (Thay bằng code mới)

```bash
# Xóa các file
sEMG_KNN.py                    # 32KB - Thay bằng train_models.py
sEMG_LDA.py                    # 31KB - Thay bằng train_models.py
sEMG_SVM.py                    # 40KB - Thay bằng train_models.py
sEMG_SVM_Classification.py     # 19KB - Thay bằng train_models.py
emg_classify_full.py           # 21KB - Thay bằng run_full_pipeline.py
knn_train.py                   # 5KB  - Thay bằng train_models.py
lda_train.py                   # 4KB  - Thay bằng train_models.py
run_svm.py                     # 3KB  - Thay bằng run_full_pipeline.py
predict_emg.py                 # 11KB - Thay bằng demo_predict.py
predict_improved.py            # 5KB  - Thay bằng demo_predict.py
predict_with_trained_pipeline.py # 7KB - Thay bằng demo_predict.py
improved_preprocessing.py      # 4KB  - Logic đã tích hợp vào code mới
```

**Lý do xóa:** Code cũ, không có structure, khó maintain. Code mới clean hơn, có GridSearchCV, docs đầy đủ.

### 2. **Models và Results Cũ**

```bash
best_model.joblib             # 8KB  - Model cũ
predictions.csv               # 3KB  - Results cũ
results.csv                   # 1KB  - Results cũ
summary.csv                   # 252B - Summary cũ
summary.tex                   # 438B - LaTeX cũ
```

**Lý do xóa:** Models và results từ code cũ. Code mới tạo ra models tốt hơn trong thư mục `models/`.

### 3. **Run Artifacts Cũ**

```bash
run_artifacts_target_seed_12/
run_artifacts_target_seed_13/
run_artifacts_target_seed_14/
run_artifacts_target_seed_15/
run_artifacts_target_seed_16/
run_artifacts_target_seed_17/
run_artifacts_target_seed_18/
```

**Lý do xóa:** Experiments cũ, không cần thiết.

### 4. **Documentation Cũ**

```bash
README.md                     # README cũ, thay bằng README_NEW.md
README_SVM.md                 # 5KB - Docs SVM cũ
readme.docx                   # 16KB - Word doc cũ
requirements_svm.txt          # 157B - Requirements cũ, thay bằng requirements_new.txt
```

**Lý do xóa:** Docs cũ, không cập nhật. Docs mới đầy đủ hơn trong `docs/`.

### 5. **Notebooks Cũ** (Tùy chọn - có thể giữ)

```bash
STFT_ASB_CNN.ipynb            # 9.2MB - Notebook CNN (nếu không dùng thì xóa)
```

**Lý do xóa:** Nếu không làm tiếp phần CNN. Nếu giữ thì move vào `notebooks/`.

---

## ✅ CÁC FILE GIỮ LẠI

### 📂 Dataset (QUAN TRỌNG!)
```bash
dataset/fatigue/*.csv         # ✓ Dataset gốc
dataset/non fatigue/*.csv     # ✓ Dataset gốc
```

### 📄 Code Mới (ĐÃ TẠO)
```bash
generate_data.py              # ✓ Generate synthetic data
train_models.py               # ✓ Train 3 models (LDA, KNN, SVM)
test_models.py                # ✓ Test và evaluate
run_full_pipeline.py          # ✓ Full pipeline
demo_predict.py               # ✓ Demo prediction
```

### 📄 Documentation Mới
```bash
README_NEW.md                 # ✓ Hướng dẫn chi tiết
QUICKSTART.md                 # ✓ Quick start guide
ANSWERS_QUESTIONS.md          # ✓ Trả lời 6 câu hỏi
CLEANUP_GUIDE.md              # ✓ File này
requirements_new.txt          # ✓ Dependencies mới
```

### 📄 Config
```bash
.gitignore                    # ✓ Git ignore đã update
.idea/                        # ✓ IDE settings (nếu dùng)
```

---

## 🚀 CÁCH CLEAN

### Phương pháp 1: Tự động (Khuyến nghị)

```bash
# Chạy script cleanup
bash cleanup.sh

# Hoặc
python cleanup.py
```

### Phương pháp 2: Thủ công

```bash
# 1. Xóa Python scripts cũ
rm sEMG_*.py emg_classify_full.py *_train.py run_svm.py predict*.py improved_preprocessing.py

# 2. Xóa models và results cũ
rm best_model.joblib predictions.csv results.csv summary.*

# 3. Xóa run artifacts
rm -rf run_artifacts_target_seed_*

# 4. Xóa docs cũ (BACKUP trước!)
rm README_SVM.md readme.docx requirements_svm.txt

# 5. Tùy chọn: Xóa notebook nếu không dùng
# rm STFT_ASB_CNN.ipynb

# 6. Tổ chức lại
mkdir -p src docs
mv generate_data.py train_models.py test_models.py run_full_pipeline.py demo_predict.py src/
mv README_NEW.md QUICKSTART.md ANSWERS_QUESTIONS.md docs/
mv requirements_new.txt requirements.txt
mv README_NEW.md README.md
```

---

## 📊 TRƯỚC VÀ SAU KHI CLEAN

### Trước:
```
82 files, ~10MB
├── 8 Python scripts cũ (không maintain)
├── 7 run_artifacts folders
├── 4 docs cũ
├── Nhiều files rời rạc
└── Khó tìm file cần thiết
```

### Sau:
```
~20 files, <2MB (không tính dataset)
├── 6 Python scripts mới (clean, documented)
├── 3 docs mới (đầy đủ)
├── Structure rõ ràng (src/, docs/)
└── Dễ maintain và sử dụng ✓
```

---

## ⚠️ CHÚ Ý QUAN TRỌNG

### 🛡️ BACKUP TRƯỚC KHI XÓA!

```bash
# Tạo backup
mkdir ../backup_nhandangdomoico
cp -r . ../backup_nhandangdomoico/
# Hoặc
git stash
```

### 📋 CHECKLIST

- [ ] Đã backup source code
- [ ] Đã review files cần xóa
- [ ] Dataset gốc (`dataset/`) KHÔNG bị xóa
- [ ] Code mới (`generate_data.py`, etc.) vẫn còn
- [ ] Docs mới (`README_NEW.md`, etc.) vẫn còn
- [ ] Git status clean
- [ ] Test chạy lại code sau khi clean:
  ```bash
  python run_full_pipeline.py --n-samples 1000 --no-grid-search
  ```

---

## 🎯 KẾT QUẢ MONG ĐỢI

Sau khi clean, bạn có:

✅ **Source code sạch sẽ**
- Code mới, có structure
- Dễ đọc, dễ maintain
- Documentation đầy đủ

✅ **Performance tốt**
- Models với 95.73% accuracy
- GridSearchCV đã optimize
- Đạt mục tiêu 85-95%

✅ **Sẵn sàng báo cáo**
- Docs đầy đủ
- Demo scripts
- Trả lời được câu hỏi

---

## 📞 HỖ TRỢ

Nếu có vấn đề sau khi clean:

1. **Code không chạy:**
   ```bash
   # Restore từ backup
   cp -r ../backup_nhandangdomoico/* .
   ```

2. **Thiếu file:**
   - Kiểm tra backup
   - Kiểm tra git history: `git log --all -- <filename>`

3. **Import error:**
   ```bash
   pip install -r requirements_new.txt
   ```

---

**Chúc bạn clean code thành công! 🎉**
