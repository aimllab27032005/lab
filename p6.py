import numpy as np  # Import NumPy for numerical operations
from sklearn.datasets import load_iris  # Import Iris dataset from scikit-learn
from sklearn.model_selection import train_test_split  # Import function to split dataset
from sklearn.metrics import confusion_matrix, classification_report  # Import metrics for evaluation

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target  # Extract features (X) and labels (y)
class_names = iris.target_names  # Get class names (setosa, versicolor, virginica)

# Define a Naïve Bayes classifier class
class NaiveBayes:
    def fit(self, X, y):
        """Train the Naïve Bayes classifier."""
        self._classes = np.unique(y)  # Get unique class labels
        # Compute mean for each class
        self._mean = np.array([X[y == c].mean(axis=0) for c in self._classes])
        # Compute variance for each class
        self._var = np.array([X[y == c].var(axis=0) for c in self._classes])
        # Compute prior probabilities for each class
        self._priors = np.array([X[y == c].shape[0] / len(y) for c in self._classes])

    def predict(self, X):
        """Predict class labels for a given dataset."""
        return np.array([self._predict(x) for x in X])  # Predict for each sample

    def _predict(self, x):
        """Compute posterior probability for each class and return the class with the highest probability."""
        posteriors = [np.log(prior) + np.sum(np.log(self._pdf(idx, x)))
                      for idx, prior in enumerate(self._priors)]
        return self._classes[np.argmax(posteriors)]  # Return class with max posterior probability

    def _pdf(self, class_idx, x):
        """Compute the probability density function for a given class."""
        mean, var = self._mean[class_idx], self._var[class_idx]  # Get mean and variance for the class
        numerator = np.exp(- (x - mean)**2 / (2 * var))  # Compute Gaussian formula numerator
        denominator = np.sqrt(2 * np.pi * var)  # Compute Gaussian formula denominator
        return numerator / denominator  # Return probability density function value

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create and train the Naïve Bayes model
nb = NaiveBayes()
nb.fit(X_train, y_train)  # Train model using training data

# Make predictions on test data
y_pred = nb.predict(X_test)

# Calculate and print accuracy
print('Accuracy: %.4f' % np.mean(y_pred == y_test))

# Print predicted class names
print("Predictions:", iris.target_names[y_pred])

# Compute and print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print classification report (precision, recall, f1-score for each class)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))
