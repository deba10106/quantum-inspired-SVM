import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt

class QuantumInspiredSVM:
    def __init__(self, kernel='linear'):
        self.kernel = kernel
        self.model = SVC(kernel=self.kernel)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

def classical_svm(X_train, y_train, X_test):
    start_time = time.time()
    classical_svm_model = SVC(kernel='linear')  # Using a linear kernel for classical SVM
    classical_svm_model.fit(X_train, y_train)
    predictions_classical = classical_svm_model.predict(X_test)
    end_time = time.time()
    training_time = end_time - start_time
    return predictions_classical, training_time

if __name__ == '__main__':
    # Load dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Quantum Inspired SVM
    qsvm = QuantumInspiredSVM(kernel='rbf')  # Using RBF kernel as an example
    start_time = time.time()
    qsvm.fit(X_train, y_train)
    end_time = time.time()
    qsvm_training_time = end_time - start_time

    # Make predictions
    predictions_qsvm = qsvm.predict(X_test)

    # Print predictions and actual results
    print(f'Predictions (Quantum): {predictions_qsvm}')
    print(f'Actual Results: {y_test}')
    print(f'Accuracy (Quantum): {accuracy_score(y_test, predictions_qsvm)}')
    print(f'Classification Report (Quantum):\n{classification_report(y_test, predictions_qsvm)}')
    print(f'Confusion Matrix (Quantum):\n{confusion_matrix(y_test, predictions_qsvm)}')

    # Compare predictions and actual results
    if np.array_equal(predictions_qsvm, y_test):
        print('Status: Success')
    else:
        print('Status: Failure')

    # Run Classical SVM
    predictions_classical, classical_training_time = classical_svm(X_train, y_train, X_test)
    print(f'Predictions (Classical): {predictions_classical}')
    print(f'Accuracy (Classical): {accuracy_score(y_test, predictions_classical)}')
    print(f'Classification Report (Classical):\n{classification_report(y_test, predictions_classical)}')
    print(f'Confusion Matrix (Classical):\n{confusion_matrix(y_test, predictions_classical)}')

    # Measure performance
    print(f'Training Time (Quantum): {qsvm_training_time}')
    print(f'Training Time (Classical): {classical_training_time}')

    # Efficiency Analysis
    labels = ['Quantum SVM', 'Classical SVM']
    times = [qsvm_training_time, classical_training_time]

    plt.bar(labels, times, color=['blue', 'orange'])
    plt.ylabel('Training Time (seconds)')
    plt.title('Comparative Efficiency Analysis of SVMs')
    plt.savefig('efficiency_analysis.png', bbox_inches='tight')  # Save the plot as an image file
    plt.close()  # Close the plot to free up memory
