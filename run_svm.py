"""
Simple script to run SVM classification for sEMG muscle fatigue detection
"""

from sEMG_SVM_Classification import sEMGSVMClassifier
import matplotlib.pyplot as plt

def run_svm_classification():
    """Run the complete SVM classification pipeline"""
    
    print("Starting sEMG Muscle Fatigue Classification with SVM")
    print("=" * 60)
    
    try:
        # Initialize the classifier
        classifier = sEMGSVMClassifier()
        
        # Prepare the data
        print("\n1. Preparing data...")
        classifier.prepare_data(test_size=0.2, random_state=42)
        
        # Train SVM with RBF kernel and grid search
        print("\n2. Training SVM model...")
        classifier.train_svm(kernel='rbf', use_grid_search=True)
        
        # Evaluate the model
        print("\n3. Evaluating model...")
        train_acc, test_acc, auc_score = classifier.evaluate_model()
        
        # Compare different kernels
        print("\n4. Comparing different kernels...")
        kernel_results = classifier.compare_kernels()
        
        # Print summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        print(f"Best Model Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"AUC Score: {auc_score:.4f}")
        
        print("\nKernel Comparison Results:")
        for kernel, results in kernel_results.items():
            print(f"  {kernel.capitalize():8}: {results['accuracy']:.4f} (CV: {results['cv_mean']:.4f})")
        
        # Save the model
        print("\n5. Saving model...")
        classifier.save_model('best_svm_fatigue_model.pkl')
        
        print("\nClassification complete! Check the generated plots and saved model.")
        
        return classifier, kernel_results
        
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        print("Please make sure the dataset folder structure is correct:")
        print("  dataset/")
        print("    fatigue/")
        print("      *.csv files")
        print("    non fatigue/")
        print("      *.csv files")
        return None, None

def quick_test():
    """Quick test with limited grid search for faster execution"""
    
    print("Running Quick Test (Limited Grid Search)")
    print("=" * 50)
    
    try:
        classifier = sEMGSVMClassifier()
        classifier.prepare_data(test_size=0.3, random_state=42)
        
        # Train with simple parameters (no grid search for speed)
        classifier.train_svm(kernel='rbf', use_grid_search=False)
        
        # Quick evaluation
        train_acc, test_acc, auc_score = classifier.evaluate_model()
        
        print(f"\nQuick Test Results:")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"AUC Score: {auc_score:.4f}")
        
        return classifier
        
    except Exception as e:
        print(f"Error during quick test: {str(e)}")
        return None

if __name__ == "__main__":
    import sys
    
    print("sEMG Muscle Fatigue SVM Classification")
    print("Choose an option:")
    print("1. Full classification with grid search (recommended)")
    print("2. Quick test (faster, less accurate)")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        classifier, results = run_svm_classification()
    elif choice == "2":
        classifier = quick_test()
    else:
        print("Invalid choice. Running full classification...")
        classifier, results = run_svm_classification()
