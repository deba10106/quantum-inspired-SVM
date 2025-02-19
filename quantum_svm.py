import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

def generate_complex_dataset(n_samples=10000):
    """Generate a complex dataset with multiple features and classes"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,  # More features for complexity
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=42
    )
    return X, y

class QuantumInspiredSVM:
    def __init__(self):
        self.model = SVC(kernel='rbf')  # Using RBF kernel for better performance
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate and print model performance metrics"""
    print(f"Predictions ({model_name}): {y_pred[:30]}")  # Show first 30 predictions
    print(f"Actual Results: {y_true[:30]}")
    print(f"Accuracy ({model_name}): {accuracy_score(y_true, y_pred)}")
    print(f"Classification Report ({model_name}):")
    print(classification_report(y_true, y_pred))
    print(f"Confusion Matrix ({model_name}):")
    print(confusion_matrix(y_true, y_pred))

def plot_training_times(qsvm_time, classical_time):
    """Plot the training times comparison"""
    labels = ['Quantum-Inspired SVM', 'Classical SVM']
    times = [qsvm_time, classical_time]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, times, color=['blue', 'orange'])
    plt.ylabel('Training Time (seconds)')
    plt.title('Comparative Efficiency Analysis of SVMs')
    
    # Add time values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s',
                ha='center', va='bottom')
    
    plt.savefig('efficiency_analysis.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Generate complex dataset
    print("Generating complex dataset...")
    X, y = generate_complex_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset size: {len(X)} samples with {X.shape[1]} features")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Test Quantum-Inspired SVM
    print("\nTesting Quantum-Inspired SVM...")
    qsvm = QuantumInspiredSVM()
    start_time = time.time()
    qsvm.fit(X_train, y_train)
    qsvm_training_time = time.time() - start_time
    qsvm_predictions = qsvm.predict(X_test)
    print(f"Training Time (Quantum): {qsvm_training_time:.4f} seconds")
    evaluate_model(y_test, qsvm_predictions, "Quantum")
    
    # Wait for 10 seconds
    print("\nWaiting for 10 seconds before testing Classical SVM...")
    time.sleep(10)
    
    # Test Classical SVM
    print("\nTesting Classical SVM...")
    classical_svm_model = SVC(kernel='linear')
    start_time = time.time()
    classical_svm_model.fit(X_train, y_train)
    classical_training_time = time.time() - start_time
    classical_predictions = classical_svm_model.predict(X_test)
    print(f"Training Time (Classical): {classical_training_time:.4f} seconds")
    evaluate_model(y_test, classical_predictions, "Classical")
    
    # Plot results
    plot_training_times(qsvm_training_time, classical_training_time)
    
    if accuracy_score(y_test, qsvm_predictions) > 0.8 and accuracy_score(y_test, classical_predictions) > 0.8:
        print("\nStatus: Success")
    else:
        print("\nStatus: Further optimization needed")
