import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from utils.logger import setup_logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, silhouette_score

logger = setup_logger()
results_dir = Path("resources/results")

def train_knn(df: pd.DataFrame, target_col: str = "condition", n_neighbors: int = 5) -> KNeighborsClassifier:
    """
    Trains a KNN classifier using ROM, angle, and angular_velocity.
    Splits data into train/test sets, evaluates performance, and logs metrics.
    """
    logger.info("Preparing data for KNN")

    # Features and target
    feature_cols = ["ROM", "mean_angle", "std_angle", "mean_velocity", "std_velocity", "max_velocity", "ROM_diff", "angle_corr"]
    X = df[feature_cols] 
    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(f"KNN test accuracy: {acc:.3f}")
    logger.info("Classification report:")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            logger.info(f"  {label}: precision={metrics['precision']:.2f}, recall={metrics['recall']:.2f}, f1={metrics['f1-score']:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix")
    plt.xlabel("Cluster")
    plt.ylabel("Condition")
    plt.tight_layout()
    plt.savefig(results_dir / "knn_cf.png", dpi=300, bbox_inches="tight")
    logger.info("knn cf saved")
    # plt.show()


    return model