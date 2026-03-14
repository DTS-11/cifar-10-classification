# CIFAR-10 classification using KNN

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.datasets import cifar10  # type: ignore

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test  = x_test.reshape(x_test.shape[0], -1) / 255.0
y_train = y_train.ravel()
y_test  = y_test.ravel()

# Use subset for faster execution
x_train_small = x_train[:10000]
y_train_small = y_train[:10000]
x_test_small  = x_test[:2000]
y_test_small  = y_test[:2000]

# Train once at import time so main.py can reuse the fitted model
knn_model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn_model.fit(x_train_small, y_train_small)


def run_knn():
    y_pred = knn_model.predict(x_test_small)
    return accuracy_score(y_test_small, y_pred)


if __name__ == "__main__":
    print("Accuracy:", run_knn())
