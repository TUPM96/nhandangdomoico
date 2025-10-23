"""
sEMG Muscle Fatigue Classification using SVM
Author: AI Assistant
Date: September 2025

This script implements Support Vector Machine (SVM) classification for muscle fatigue detection
based on Surface Electromyography (sEMG) signals from handgrip strength measurements.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, stft
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class sEMGSVMClassifier:
    def __init__(self, data_path=None):
        """
        Initialize the sEMG SVM Classifier
        
        Args:
            data_path (str): Path to the dataset folder
        """
        # If no data_path provided, use the dataset folder in the same directory as this script
        if data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(script_dir, 'dataset')
        
        self.data_path = data_path
        self.data_fatigue_path = os.path.join(data_path, 'fatigue')
        self.data_nonfatigue_path = os.path.join(data_path, 'non fatigue')
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=50)  # Select top 50 features
        self.svm_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Signal processing parameters
        self.lowcut = 10    # Low cutoff frequency in Hz
        self.highcut = 100  # High cutoff frequency in Hz
        self.fs = 1000      # Sampling frequency in Hz
        
        print("sEMG SVM Classifier initialized")
        print(f"Fatigue data path: {self.data_fatigue_path}")
        print(f"Non-fatigue data path: {self.data_nonfatigue_path}")
    
    def zero_baseline_correction(self, data):
        """Apply zero baseline correction to the signal"""
        baseline = np.mean(data)
        return data - baseline
    
    def absolute_signal(self, data):
        """Convert signal to absolute values"""
        return np.abs(data)
    
    def normalisasi(self, data):
        """Normalize data to range [0, 1]"""
        nmin = min(data)
        nmax = max(data)
        if nmax == nmin:
            return [0] * len(data)
        return [(x - nmin) / (nmax - nmin) for x in data]
    
    def butterworth_filter(self, data, order=4, btype='band'):
        """Apply Butterworth bandpass filter"""
        nyquist_freq = 0.5 * self.fs
        low = self.lowcut / nyquist_freq
        high = self.highcut / nyquist_freq
        b, a = butter(order, [low, high], btype=btype, analog=False)
        return filtfilt(b, a, data)
    
    def extract_time_domain_features(self, signal):
        """Extract time domain features from the signal"""
        features = []
        
        # Basic statistical features
        features.append(np.mean(signal))           # Mean
        features.append(np.std(signal))            # Standard deviation
        features.append(np.var(signal))            # Variance
        features.append(np.max(signal))            # Maximum
        features.append(np.min(signal))            # Minimum
        features.append(np.median(signal))         # Median
        features.append(np.percentile(signal, 25)) # Q1
        features.append(np.percentile(signal, 75)) # Q3
        
        # Advanced features
        features.append(np.sqrt(np.mean(signal**2)))                    # RMS
        features.append(np.mean(np.abs(signal - np.mean(signal))))      # Mean Absolute Deviation
        features.append(len(np.where(np.diff(np.sign(signal)))[0]))     # Zero crossings
        features.append(np.sum(np.abs(np.diff(signal))))                # Total variation
        
        # Skewness and Kurtosis
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        if std_val > 0:
            features.append(np.mean(((signal - mean_val) / std_val) ** 3))  # Skewness
            features.append(np.mean(((signal - mean_val) / std_val) ** 4))  # Kurtosis
        else:
            features.append(0)
            features.append(0)
        
        return features
    
    def extract_frequency_domain_features(self, signal):
        """Extract frequency domain features using STFT"""
        f, t, Zxx = stft(signal, fs=self.fs, nperseg=256)
        magnitude = np.abs(Zxx)
        power_spectrum = np.mean(magnitude, axis=1)
        
        features = []
        
        # Power spectral density features
        features.append(np.mean(power_spectrum))      # Mean power
        features.append(np.std(power_spectrum))       # Power variation
        features.append(np.max(power_spectrum))       # Peak power
        features.append(np.sum(power_spectrum))       # Total power
        
        # Frequency band powers
        # Define frequency bands
        bands = {
            'low': (10, 30),
            'mid': (30, 70),
            'high': (70, 100)
        }
        
        for band_name, (low_f, high_f) in bands.items():
            band_indices = np.where((f >= low_f) & (f <= high_f))[0]
            if len(band_indices) > 0:
                band_power = np.sum(power_spectrum[band_indices])
                features.append(band_power)
            else:
                features.append(0)
        
        # Spectral centroid
        if np.sum(power_spectrum) > 0:
            spectral_centroid = np.sum(f * power_spectrum) / np.sum(power_spectrum)
            features.append(spectral_centroid)
        else:
            features.append(0)
        
        return features
    
    def load_and_preprocess_signal(self, file_path, ranges_to_keep):
        """Load and preprocess a single signal file"""
        try:
            data = pd.read_csv(file_path)['amplitudo'].values
            
            # Apply preprocessing steps
            data = self.zero_baseline_correction(data)
            data = self.absolute_signal(data)
            
            # Extract specified ranges and concatenate
            filtered_data = np.concatenate([data[start:end] for start, end in ranges_to_keep])
            
            # Normalize data
            data_normalized = self.normalisasi(filtered_data)
            
            # Apply Butterworth filter
            data_filtered = self.butterworth_filter(data_normalized)
            
            return data_filtered
            
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return None
    
    def extract_features_from_signal(self, signal):
        """Extract comprehensive features from a signal"""
        if signal is None:
            return None
        
        # Split signal into segments for feature extraction
        segment_length = len(signal) // 2
        segment1 = signal[:segment_length]
        segment2 = signal[segment_length:]
        
        features = []
        
        # Extract features from each segment
        for segment in [segment1, segment2]:
            time_features = self.extract_time_domain_features(segment)
            freq_features = self.extract_frequency_domain_features(segment)
            features.extend(time_features)
            features.extend(freq_features)
        
        # Extract features from the full signal
        full_time_features = self.extract_time_domain_features(signal)
        full_freq_features = self.extract_frequency_domain_features(signal)
        features.extend(full_time_features)
        features.extend(full_freq_features)
        
        return features
    
    def load_dataset(self):
        """Load and preprocess the entire dataset"""
        print("Loading dataset...")
        
        # File lists
        fatigue_files = [f for f in os.listdir(self.data_fatigue_path) if f.endswith('.csv')]
        nonfatigue_files = [f for f in os.listdir(self.data_nonfatigue_path) if f.endswith('.csv')]
        
        print(f"Found {len(fatigue_files)} fatigue files and {len(nonfatigue_files)} non-fatigue files")
        
        # Define ranges to extract (same as in original notebook)
        ranges_to_keep = [(15000, 25000), (30000, 35000)]
        
        X = []
        y = []
        
        # Process fatigue files
        print("Processing fatigue files...")
        for filename in fatigue_files:
            file_path = os.path.join(self.data_fatigue_path, filename)
            signal = self.load_and_preprocess_signal(file_path, ranges_to_keep)
            
            if signal is not None:
                features = self.extract_features_from_signal(signal)
                if features is not None:
                    X.append(features)
                    y.append(1)  # Label 1 for fatigue
        
        # Process non-fatigue files
        print("Processing non-fatigue files...")
        for filename in nonfatigue_files:
            file_path = os.path.join(self.data_nonfatigue_path, filename)
            signal = self.load_and_preprocess_signal(file_path, ranges_to_keep)
            
            if signal is not None:
                features = self.extract_features_from_signal(signal)
                if features is not None:
                    X.append(features)
                    y.append(0)  # Label 0 for non-fatigue
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")
        print(f"Class distribution: {np.sum(y == 0)} non-fatigue, {np.sum(y == 1)} fatigue")
        
        return X, y
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for training"""
        print("Preparing data...")
        
        # Load dataset
        X, y = self.load_dataset()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Feature selection
        self.X_train_selected = self.feature_selector.fit_transform(self.X_train_scaled, self.y_train)
        self.X_test_selected = self.feature_selector.transform(self.X_test_scaled)
        
        print(f"Training set: {self.X_train_selected.shape}")
        print(f"Test set: {self.X_test_selected.shape}")
        print(f"Selected {self.X_train_selected.shape[1]} features out of {self.X_train_scaled.shape[1]}")
    
    def train_svm(self, kernel='rbf', use_grid_search=True):
        """Train SVM model with optional hyperparameter tuning"""
        print("Training SVM model...")
        
        if use_grid_search:
            print("Performing grid search for hyperparameter tuning...")
            
            # Define parameter grid
            if kernel == 'rbf':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf']
                }
            elif kernel == 'linear':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear']
                }
            elif kernel == 'poly':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'degree': [2, 3, 4],
                    'kernel': ['poly']
                }
            else:
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'kernel': [kernel]
                }
            
            # Perform grid search
            svm = SVC(random_state=42)
            grid_search = GridSearchCV(
                svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train_selected, self.y_train)
            
            self.svm_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
        else:
            # Train with default or specified parameters
            self.svm_model = SVC(kernel=kernel, random_state=42)
            self.svm_model.fit(self.X_train_selected, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.svm_model, self.X_train_selected, self.y_train, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if self.svm_model is None:
            print("No model trained yet!")
            return
        
        print("Evaluating model...")
        
        # Make predictions
        y_train_pred = self.svm_model.predict(self.X_train_selected)
        y_test_pred = self.svm_model.predict(self.X_test_selected)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report (Test Set):")
        print(classification_report(self.y_test, y_test_pred, 
                                  target_names=['Non-Fatigue', 'Fatigue']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_test_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Fatigue', 'Fatigue'],
                   yticklabels=['Non-Fatigue', 'Fatigue'])
        plt.title('Confusion Matrix - SVM Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('SVM_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # ROC curve (if model supports probability prediction)
        if hasattr(self.svm_model, 'predict_proba'):
            y_prob = self.svm_model.predict_proba(self.X_test_selected)[:, 1]
        else:
            # Use decision function for SVM
            y_prob = self.svm_model.decision_function(self.X_test_selected)
        
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - SVM Model')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('SVM_ROC_Curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return train_accuracy, test_accuracy, roc_auc
    
    def compare_kernels(self):
        """Compare different SVM kernels"""
        print("Comparing different SVM kernels...")
        
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        results = {}
        
        for kernel in kernels:
            print(f"\nTraining SVM with {kernel} kernel...")
            
            # Train model with current kernel
            svm = SVC(kernel=kernel, random_state=42)
            svm.fit(self.X_train_selected, self.y_train)
            
            # Evaluate
            y_pred = svm.predict(self.X_test_selected)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(svm, self.X_train_selected, self.y_train, cv=5)
            
            results[kernel] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"{kernel.capitalize()} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Plot comparison
        kernels_list = list(results.keys())
        accuracies = [results[k]['accuracy'] for k in kernels_list]
        cv_means = [results[k]['cv_mean'] for k in kernels_list]
        cv_stds = [results[k]['cv_std'] for k in kernels_list]
        
        x = np.arange(len(kernels_list))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
        rects2 = ax.bar(x + width/2, cv_means, width, label='CV Mean', alpha=0.8)
        
        ax.set_ylabel('Accuracy')
        ax.set_title('SVM Kernel Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(kernels_list)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.savefig('SVM_Kernel_Comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    
    def save_model(self, filename='svm_model.pkl'):
        """Save the trained model"""
        import joblib
        
        if self.svm_model is None:
            print("No model to save!")
            return
        
        # Save model and preprocessing objects
        model_data = {
            'model': self.svm_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='svm_model.pkl'):
        """Load a saved model"""
        import joblib
        
        try:
            model_data = joblib.load(filename)
            self.svm_model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            print(f"Model loaded from {filename}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")

def main():
    """Main function to run the SVM classification"""
    print("sEMG Muscle Fatigue Classification using SVM")
    print("=" * 50)
    
    # Initialize classifier
    classifier = sEMGSVMClassifier()
    
    # Prepare data
    classifier.prepare_data()
    
    # Train model with grid search
    classifier.train_svm(kernel='rbf', use_grid_search=True)
    
    # Evaluate model
    train_acc, test_acc, auc_score = classifier.evaluate_model()
    
    # Compare different kernels
    kernel_results = classifier.compare_kernels()
    
    # Save the best model
    classifier.save_model('best_svm_model.pkl')
    
    print("\n" + "=" * 50)
    print("SVM Classification Complete!")
    print(f"Best Test Accuracy: {test_acc:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
