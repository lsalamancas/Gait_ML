import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from utils.logger import logger

def train_knn(df: pd.DataFrame, target_col: str = "condition", n_neighbors: int = 5) -> KNeighborsClassifier:
    """
    Trains a KNN classifier using ROM, angle, and angular_velocity.
    Splits data into train/test sets, evaluates performance, and logs metrics.
    """
    logger.info("Preparing data for KNN")

    # Features and target
    feature_cols = ["ROM", "angle", "angular_velocity"]
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

    return model