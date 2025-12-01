import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from utils.logger import setup_logger

logger = setup_logger()
results_dir = Path("resources/results")

def train_random_forest(df: pd.DataFrame, target_col: str = "condition") -> RandomForestClassifier:
    """
    Trains a Random Forest classifier using aggregated gait features.
    Splits data into train/test sets, evaluates performance, logs metrics,
    and generates plots (feature importances + confusion matrix).
    """
    logger.info("Preparing data for Random Forest")

    # Features y target (usar las columnas de df_feat)
    feature_cols = ["ROM", "mean_angle", "std_angle", "mean_velocity", "std_velocity", "max_velocity", "ROM_diff", "angle_corr"]
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

    # --- Plot 1: Feature importances ---
    importances = model.feature_importances_
    logger.info(importances)
    plt.figure(figsize=(8,6))
    sns.barplot(x=importances, y=feature_cols, palette="viridis")
    plt.title("Feature Importance in Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(results_dir / "importance_features_rf.png", dpi=300, bbox_inches="tight")
    logger.info("random Forest matrix saved")
    # plt.show()

    # --- Plot 2: Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=sorted(df[target_col].unique()))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(df[target_col].unique()),
                yticklabels=sorted(df[target_col].unique()))
    plt.title("Matriz de confusión (Random Forest)")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix_rf.png", dpi=300, bbox_inches="tight")
    logger.info("random Forest matrix 2 saved")
    # plt.show()

    return model