import threading
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score

from knn import run_knn, knn_model, x_test_small, y_test_small
from nb import run_nb, nb_model


def wrapper(func, name, results):
    print(f"Starting {name}...")
    start_time = time.time()
    acc = func()
    end_time = time.time()
    results[name] = acc
    print(f"{name} finished in {end_time - start_time:.2f} seconds.")


def compute_metrics(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "F1 Score":  f1_score(y_test, y_pred, average="macro", zero_division=0),
    }


def plot_metrics(knn_metrics, nb_metrics):
    metric_names = list(knn_metrics.keys())
    knn_values   = [knn_metrics[m] * 100 for m in metric_names]
    nb_values    = [nb_metrics[m]  * 100 for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0d0d1a")

    bars1 = ax.bar(x - width / 2, knn_values, width, label="KNN",         color="#00e5ff", alpha=0.85)
    bars2 = ax.bar(x + width / 2, nb_values,  width, label="Naive Bayes", color="#ff6b9d", alpha=0.85)

    ax.set_xlabel("Metric", color="#7788bb")
    ax.set_ylabel("Score (%)", color="#7788bb")
    ax.set_title("KNN vs Naive Bayes — CIFAR-10 Performance", color="#e8e8f0", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, color="#7788bb")
    ax.tick_params(colors="#5566aa")
    ax.spines[:].set_color("#1e1e3f")
    ax.yaxis.label.set_color("#7788bb")
    ax.set_ylim(0, 50)
    ax.legend(facecolor="#0d0d1a", edgecolor="#2a2a5a", labelcolor="#e8e8f0")
    ax.grid(axis="y", color="#1e1e3f", linestyle="--", linewidth=0.8)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8, color="#00e5ff")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8, color="#ff6b9d")

    plt.tight_layout()
    plt.savefig("metrics_chart.png", dpi=150, facecolor=fig.get_facecolor())
    plt.show()
    print("Chart saved to metrics_chart.png")


def predict_image(image_path, knn, nb):
    """
    Load a single image, preprocess it, and predict using both models.
    Supports any image format readable by PIL.
    """
    from PIL import Image

    CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                       "dog","frog","horse","ship","truck"]

    img = Image.open(image_path).convert("RGB").resize((32, 32))
    img_array = np.array(img).reshape(1, -1) / 255.0

    knn_pred = knn.predict(img_array)[0]
    nb_pred  = nb.predict(img_array)[0]

    print(f"\n{'='*40}")
    print(f"  Image: {image_path}")
    print(f"  KNN prediction        → {CIFAR10_CLASSES[knn_pred]}")
    print(f"  Naive Bayes prediction → {CIFAR10_CLASSES[nb_pred]}")
    print(f"{'='*40}\n")
    return CIFAR10_CLASSES[knn_pred], CIFAR10_CLASSES[nb_pred]


if __name__ == "__main__":
    results = {}

    t1 = threading.Thread(target=wrapper, args=(run_knn, "KNN", results))
    t2 = threading.Thread(target=wrapper, args=(run_nb, "Naive Bayes", results))

    print("Starting classification tasks simultaneously...")
    start_total = time.time()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    end_total = time.time()

    print(f"\nAll tasks completed in {end_total - start_total:.2f} seconds.")
    print("-" * 40)
    print(f"KNN Accuracy:         {results.get('KNN', 0) * 100:.2f}%")
    print(f"Naive Bayes Accuracy: {results.get('Naive Bayes', 0) * 100:.2f}%")
    print("-" * 40)

    # Compute full metrics and plot
    print("\nComputing detailed metrics...")
    knn_metrics = compute_metrics(knn_model, x_test_small, y_test_small)
    nb_metrics  = compute_metrics(nb_model,  x_test_small, y_test_small)
    plot_metrics(knn_metrics, nb_metrics)

    # Image prediction
    predict_image("img1.jpeg", knn_model, nb_model)
    predict_image("img2.jpeg", knn_model, nb_model)
