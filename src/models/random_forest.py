import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from utils.logger import setup_logger

logger = setup_logger()

def train_random_forest(df: pd.DataFrame, target_col: str = "condition") -> RandomForestClassifier:
    """
    Trains a Random Forest classifier using ROM, angle, and angular_velocity.
    Splits data into train/test sets, evaluates performance, and logs metrics.
    """
    logger.info("Preparing data for Random Forest")

    # Features and target
    feature_cols = ["ROM", "angle", "angular_velocity"]
    X = df[feature_cols]
    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(f"Random Forest test accuracy: {acc:.3f}")
    logger.info("Classification report:")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            logger.info(f"  {label}: precision={metrics['precision']:.2f}, recall={metrics['recall']:.2f}, f1={metrics['f1-score']:.2f}")

    return model
