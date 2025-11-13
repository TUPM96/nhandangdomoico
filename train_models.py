"""
Script train models cho nhận dạng mỏi cơ
Hỗ trợ 3 thuật toán: LDA, KNN, SVM
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

class FatigueMuscleClassifier:
    """
    Class để train và đánh giá các model nhận dạng mỏi cơ
    """

    def __init__(self, model_type='svm', random_state=42):
        """
        Khởi tạo classifier

        Parameters:
        -----------
        model_type : str
            Loại model: 'lda', 'knn', 'svm'
        random_state : int
            Random seed
        """
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        self.training_history = {}

    def get_model(self, **kwargs):
        """
        Tạo model theo loại được chọn
        """
        if self.model_type == 'lda':
            return LinearDiscriminantAnalysis(**kwargs)
        elif self.model_type == 'knn':
            if 'random_state' in kwargs:
                del kwargs['random_state']
            return KNeighborsClassifier(**kwargs)
        elif self.model_type == 'svm':
            return SVC(random_state=self.random_state, **kwargs)
        else:
            raise ValueError(f"Model type không hợp lệ: {self.model_type}")

    def get_param_grid(self):
        """
        Lấy param grid cho GridSearchCV
        """
        if self.model_type == 'lda':
            return {
                'solver': ['svd', 'lsqr', 'eigen'],
                'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
            }
        elif self.model_type == 'knn':
            return {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        elif self.model_type == 'svm':
            return {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }

    def train(self, X_train, y_train, use_grid_search=True, cv=5, verbose=True):
        """
        Train model với hoặc không có GridSearchCV

        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        use_grid_search : bool
            Sử dụng GridSearchCV để tìm best params
        cv : int
            Số folds cho cross-validation
        verbose : bool
            In thông tin training
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING MODEL: {self.model_type.upper()}")
            print('='*60)

        # Chuẩn hóa dữ liệu
        X_train_scaled = self.scaler.fit_transform(X_train)

        if use_grid_search:
            if verbose:
                print(f"\nĐang chạy GridSearchCV với {cv}-fold cross-validation...")

            # Tạo base model
            base_model = self.get_model()

            # GridSearchCV
            param_grid = self.get_param_grid()
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1 if verbose else 0
            )

            grid_search.fit(X_train_scaled, y_train)

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_

            if verbose:
                print(f"\n✓ Best parameters: {self.best_params}")
                print(f"✓ Best CV score: {grid_search.best_score_:.4f}")

        else:
            if verbose:
                print(f"\nĐang training model với default parameters...")

            # Train với default params
            self.model = self.get_model()
            self.model.fit(X_train_scaled, y_train)

            if verbose:
                print(f"✓ Training hoàn tất!")

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv, scoring='accuracy')

        if verbose:
            print(f"\nCross-validation scores:")
            print(f"  - Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  - Min: {cv_scores.min():.4f}")
            print(f"  - Max: {cv_scores.max():.4f}")

        # Lưu lại training history
        self.training_history = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'best_params': self.best_params if use_grid_search else None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return self

    def evaluate(self, X_test, y_test, verbose=True):
        """
        Đánh giá model trên test set

        Returns:
        --------
        dict : Dictionary chứa các metrics
        """
        if self.model is None:
            raise ValueError("Model chưa được train! Hãy gọi train() trước.")

        # Chuẩn hóa test data
        X_test_scaled = self.scaler.transform(X_test)

        # Predict
        y_pred = self.model.predict(X_test_scaled)

        # Tính các metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        report = classification_report(y_test, y_pred, target_names=['Non-Fatigue', 'Fatigue'])

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"EVALUATION RESULTS: {self.model_type.upper()}")
            print('='*60)
            print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"\nConfusion Matrix:")
            print(cm)
            print(f"\nClassification Report:")
            print(report)

        return metrics

    def save_model(self, output_dir='models', model_name=None):
        """
        Lưu model và scaler
        """
        os.makedirs(output_dir, exist_ok=True)

        if model_name is None:
            model_name = f"{self.model_type}_model"

        # Lưu model và scaler
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'best_params': self.best_params,
            'training_history': self.training_history
        }, model_path)

        print(f"\n✓ Model đã được lưu tại: {model_path}")
        return model_path

    @classmethod
    def load_model(cls, model_path):
        """
        Load model từ file
        """
        data = joblib.load(model_path)

        classifier = cls(model_type=data['model_type'])
        classifier.model = data['model']
        classifier.scaler = data['scaler']
        classifier.best_params = data.get('best_params')
        classifier.training_history = data.get('training_history', {})

        print(f"✓ Model đã được load từ: {model_path}")
        return classifier

    def plot_confusion_matrix(self, X_test, y_test, output_dir='plots', model_name=None):
        """
        Vẽ confusion matrix
        """
        if self.model is None:
            raise ValueError("Model chưa được train!")

        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Fatigue', 'Fatigue'],
                    yticklabels=['Non-Fatigue', 'Fatigue'])
        plt.title(f'Confusion Matrix - {self.model_type.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Lưu plot
        os.makedirs(output_dir, exist_ok=True)
        if model_name is None:
            model_name = self.model_type
        plot_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Confusion matrix đã được lưu tại: {plot_path}")
        return plot_path


def train_all_models(train_data_path, test_data_path=None, use_grid_search=True,
                     output_dir='models', plot_dir='plots'):
    """
    Train tất cả các models (LDA, KNN, SVM) và so sánh kết quả

    Parameters:
    -----------
    train_data_path : str
        Path đến file train data
    test_data_path : str, optional
        Path đến file test data. Nếu None, sẽ split từ train data
    use_grid_search : bool
        Sử dụng GridSearchCV
    output_dir : str
        Thư mục lưu models
    plot_dir : str
        Thư mục lưu plots
    """
    print(f"\n{'='*60}")
    print("TRAINING ALL MODELS - FATIGUE MUSCLE DETECTION")
    print('='*60)

    # Load dữ liệu
    print(f"\nĐang load dữ liệu từ {train_data_path}...")
    train_df = pd.read_csv(train_data_path)

    # Tách features và labels
    feature_columns = [col for col in train_df.columns if col not in ['label', 'class_name']]
    X_train = train_df[feature_columns]
    y_train = train_df['label']

    print(f"✓ Train set: {len(train_df)} mẫu")
    print(f"  - Non-Fatigue: {(y_train==0).sum()} mẫu")
    print(f"  - Fatigue: {(y_train==1).sum()} mẫu")

    if test_data_path:
        test_df = pd.read_csv(test_data_path)
        X_test = test_df[feature_columns]
        y_test = test_df['label']
        print(f"✓ Test set: {len(test_df)} mẫu")
        print(f"  - Non-Fatigue: {(y_test==0).sum()} mẫu")
        print(f"  - Fatigue: {(y_test==1).sum()} mẫu")
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )
        print("✓ Đã chia train/test từ dữ liệu gốc (75/25)")

    # Train từng model
    model_types = ['lda', 'knn', 'svm']
    results = {}

    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_type.upper()}")
        print('='*60)

        # Tạo và train model
        classifier = FatigueMuscleClassifier(model_type=model_type, random_state=42)
        classifier.train(X_train, y_train, use_grid_search=use_grid_search, cv=5)

        # Đánh giá
        metrics = classifier.evaluate(X_test, y_test)

        # Lưu model
        classifier.save_model(output_dir=output_dir)

        # Vẽ confusion matrix
        classifier.plot_confusion_matrix(X_test, y_test, output_dir=plot_dir)

        # Lưu kết quả
        results[model_type] = metrics

    # So sánh kết quả
    print(f"\n{'='*60}")
    print("SO SÁNH KẾT QUẢ CÁC MODELS")
    print('='*60)

    comparison_df = pd.DataFrame({
        model: {
            'Accuracy': results[model]['accuracy'],
            'Precision': results[model]['precision'],
            'Recall': results[model]['recall'],
            'F1-Score': results[model]['f1_score']
        }
        for model in model_types
    }).T

    print("\n", comparison_df)

    # Tìm model tốt nhất
    best_model = comparison_df['Accuracy'].idxmax()
    best_accuracy = comparison_df.loc[best_model, 'Accuracy']

    print(f"\n✓ Model tốt nhất: {best_model.upper()}")
    print(f"✓ Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

    # Lưu kết quả so sánh
    os.makedirs(output_dir, exist_ok=True)
    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path)
    print(f"\n✓ Kết quả so sánh đã được lưu tại: {comparison_path}")

    # Lưu tất cả metrics
    results_path = os.path.join(output_dir, 'all_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Chi tiết kết quả đã được lưu tại: {results_path}")

    return results, comparison_df


if __name__ == "__main__":
    # Train tất cả models
    results, comparison = train_all_models(
        train_data_path='data_amplified_final/train_data.csv',
        test_data_path='data_amplified_final/test_data.csv',
        use_grid_search=True,
        output_dir='models_final',
        plot_dir='plots_final'
    )

    print("\n✓ Hoàn tất training tất cả models!")
