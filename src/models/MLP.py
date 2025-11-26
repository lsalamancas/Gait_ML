import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.logger import setup_logger

logger = setup_logger()

def train_mlp(df: pd.DataFrame, target_col: str = "condition") -> MLPClassifier:
    """
    Train a Multi-Layer Perceptron (MLP) neural network classifier
    using angle and velocity features only.
    Performs GridSearchCV for hyperparameter tuning, evaluates performance,
    and plots confusion matrix.
    """
    logger.info("Preparing data for MLP")

    # --- Features: only angles and velocities ---
    feature_cols = ["mean_angle", "std_angle", "mean_velocity", "std_velocity", "max_velocity"]
    X = df[feature_cols]
    y = df[target_col]

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # --- GridSearch for hidden layers ---
    param_grid = {
        "hidden_layer_sizes": [(16,), (32,), (32,16), (64,32)],
        "activation": ["relu", "tanh"],
        "solver": ["adam"],
        "alpha": [0.0001, 0.001],
        "max_iter": [500]
    }

    grid = GridSearchCV(
        MLPClassifier(random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    # --- Best model ---
    model = grid.best_estimator_
    logger.info(f"Best MLP params: {grid.best_params_}")

    # --- Predictions and evaluation ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(f"MLP test accuracy: {acc:.3f}")
    logger.info("Classification report:")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            logger.info(
                f"  {label}: precision={metrics['precision']:.2f}, "
                f"recall={metrics['recall']:.2f}, f1={metrics['f1-score']:.2f}"
            )

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title("Confusion Matrix (MLP - Angle & Velocity Features)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    return model