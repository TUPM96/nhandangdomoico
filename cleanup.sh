#!/bin/bash

# Script tự động clean source code
# Xóa các file cũ không cần thiết, giữ lại code mới

echo "========================================="
echo "  CLEAN SOURCE CODE - MUSCLE FATIGUE"
echo "========================================="
echo ""

# Màu sắc
RED='\033[0:31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Hỏi xác nhận
echo -e "${YELLOW}⚠️  WARNING: Script này sẽ xóa các file cũ!${NC}"
echo ""
echo "Các file sẽ bị xóa:"
echo "  - Python scripts cũ (sEMG_*.py, etc.)"
echo "  - Models cũ (best_model.joblib)"
echo "  - Results cũ (predictions.csv, etc.)"
echo "  - Run artifacts (run_artifacts_*)"
echo "  - Docs cũ (README_SVM.md, etc.)"
echo ""
echo "Dataset gốc (dataset/) SẼ ĐƯỢC GIỮ LẠI ✓"
echo ""
read -p "Bạn có chắc chắn muốn tiếp tục? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo -e "${RED}❌ Đã hủy!${NC}"
    exit 1
fi

echo ""
echo "========================================="
echo "  BẮT ĐẦU CLEAN..."
echo "========================================="
echo ""

# Đếm files
deleted_count=0

# 1. Xóa Python scripts cũ
echo "🗑️  Xóa Python scripts cũ..."
files_to_delete=(
    "sEMG_KNN.py"
    "sEMG_LDA.py"
    "sEMG_SVM.py"
    "sEMG_SVM_Classification.py"
    "emg_classify_full.py"
    "knn_train.py"
    "lda_train.py"
    "run_svm.py"
    "predict_emg.py"
    "predict_improved.py"
    "predict_with_trained_pipeline.py"
    "improved_preprocessing.py"
)

for file in "${files_to_delete[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "  ✓ Đã xóa: $file"
        ((deleted_count++))
    fi
done

# 2. Xóa models và results cũ
echo ""
echo "🗑️  Xóa models và results cũ..."
old_results=(
    "best_model.joblib"
    "predictions.csv"
    "results.csv"
    "summary.csv"
    "summary.tex"
)

for file in "${old_results[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "  ✓ Đã xóa: $file"
        ((deleted_count++))
    fi
done

# 3. Xóa run artifacts
echo ""
echo "🗑️  Xóa run artifacts..."
if ls run_artifacts_target_seed_* 1> /dev/null 2>&1; then
    rm -rf run_artifacts_target_seed_*
    echo "  ✓ Đã xóa: run_artifacts_target_seed_*"
    deleted_count=$((deleted_count + 7))
fi

# 4. Xóa docs cũ
echo ""
echo "🗑️  Xóa documentation cũ..."
old_docs=(
    "README_SVM.md"
    "readme.docx"
    "requirements_svm.txt"
)

for file in "${old_docs[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "  ✓ Đã xóa: $file"
        ((deleted_count++))
    fi
done

# 5. Tổ chức lại structure
echo ""
echo "📁 Tổ chức lại structure..."

# Tạo folders
mkdir -p src docs

# Di chuyển code mới vào src/
echo "  📂 Di chuyển code vào src/..."
code_files=(
    "generate_data.py"
    "train_models.py"
    "test_models.py"
    "run_full_pipeline.py"
    "demo_predict.py"
)

for file in "${code_files[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" "src/"
        echo "    ✓ $file → src/"
    fi
done

# Di chuyển docs vào docs/
echo "  📂 Di chuyển docs vào docs/..."
doc_files=(
    "README_NEW.md"
    "QUICKSTART.md"
    "ANSWERS_QUESTIONS.md"
    "CLEANUP_GUIDE.md"
)

for file in "${doc_files[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" "docs/"
        echo "    ✓ $file → docs/"
    fi
done

# Rename files
echo ""
echo "📝 Rename files..."
if [ -f "requirements_new.txt" ]; then
    mv requirements_new.txt requirements.txt
    echo "  ✓ requirements_new.txt → requirements.txt"
fi

# Tạo README.md mới (symlink hoặc copy)
if [ -f "docs/README_NEW.md" ]; then
    cp docs/README_NEW.md README.md
    echo "  ✓ Tạo README.md từ README_NEW.md"
fi

# 6. Update imports trong các file
echo ""
echo "🔧 Update imports..."

# Update imports to use src.module
if [ -f "src/run_full_pipeline.py" ]; then
    # Không cần update vì chạy từ root
    echo "  ⚠️  Lưu ý: Chạy scripts từ root directory!"
    echo "     cd /home/user/nhandangdomoico"
    echo "     python src/run_full_pipeline.py"
fi

# Hoàn tất
echo ""
echo "========================================="
echo "  ✅ HOÀN TẤT!"
echo "========================================="
echo ""
echo "📊 Thống kê:"
echo "  - Đã xóa: $deleted_count files"
echo "  - Đã tổ chức: src/, docs/"
echo ""
echo "📁 Cấu trúc mới:"
echo "  ."
echo "  ├── src/           (Code mới)"
echo "  ├── docs/          (Documentation)"
echo "  ├── dataset/       (Dataset gốc - GIỮ LẠI ✓)"
echo "  ├── README.md"
echo "  └── requirements.txt"
echo ""
echo "🚀 Kiểm tra:"
echo "  1. cd src"
echo "  2. python run_full_pipeline.py --n-samples 1000 --no-grid-search"
echo ""
echo -e "${GREEN}✓ Clean thành công!${NC}"
