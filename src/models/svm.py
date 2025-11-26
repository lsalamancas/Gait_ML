import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from utils.logger import setup_logger

logger = setup_logger()

def train_svm(df: pd.DataFrame, target_col: str = "condition") -> SVC:
    """
    Train a Support Vector Machine classifier using aggregated gait features.
    Uses RBF kernel and GridSearchCV for hyperparameter tuning.
    Splits data into train/test sets, evaluates performance, logs metrics,
    and generates plots (confusion matrix, PCA 2D, PCA 3D).
    """
    logger.info("Preparing data for SVM")

    # --- Features and target ---
    feature_cols = [
    "ROM", "mean_angle", "std_angle",
    "mean_velocity", "std_velocity", "max_velocity",
    "ROM_diff", "angle_corr"
    ]
    X = df[feature_cols]
    y = df[target_col]

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # --- GridSearch with RBF kernel ---
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.01, 0.1, 1]
    }
    grid = GridSearchCV(
        SVC(kernel="rbf", probability=False, random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    # --- Best model ---
    model = grid.best_estimator_
    logger.info(f"Best SVM params (original features): {grid.best_params_}")

    # --- Predictions and evaluation ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(f"SVM test accuracy (original features): {acc:.3f}")
    logger.info("Classification report (original features):")
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
    plt.title("Confusion Matrix (SVM RBF - Original Features)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # --- PCA 2D visualization ---
    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(X)
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca2[:,0], X_pca2[:,1], c=y, cmap="coolwarm", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D of Aggregated Features")
    plt.tight_layout()
    plt.show()

    # --- PCA 3D visualization ---
    pca3 = PCA(n_components=3)
    X_pca3 = pca3.fit_transform(X)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2],
                         c=y, cmap="coolwarm", s=40, alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA 3D of Aggregated Features")
    legend_labels = sorted(df[target_col].unique())
    legend = ax.legend(handles=scatter.legend_elements()[0],
                       labels=[f"Condition {c}" for c in legend_labels],
                       title="Condition", loc="upper right")
    ax.add_artist(legend)
    plt.tight_layout()
    plt.show()

    # --- Train/evaluate SVM on PCA features ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca3, y, test_size=0.2, stratify=y, random_state=42
    )
    grid_pca = GridSearchCV(
        SVC(kernel="rbf", random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_pca.fit(X_train, y_train)

    model_pca = grid_pca.best_estimator_
    logger.info(f"Best SVM params (PCA features): {grid_pca.best_params_}")

    y_pred = model_pca.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(f"SVM test accuracy (PCA features): {acc:.3f}")
    logger.info("Classification report (PCA features):")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            logger.info(
                f"  {label}: precision={metrics['precision']:.2f}, "
                f"recall={metrics['recall']:.2f}, f1={metrics['f1-score']:.2f}"
            )

    cm = confusion_matrix(y_test, y_pred, labels=model_pca.classes_)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model_pca.classes_,
                yticklabels=model_pca.classes_)
    plt.title("Confusion Matrix (SVM RBF - PCA Features)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    return model


def train_svm_by_joint(df: pd.DataFrame, target_col: str = "condition") -> dict:
    """
    Train and evaluate SVM models separately for each joint.
    Uses RBF kernel with GridSearchCV.
    Generates confusion matrix plots per joint.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe including 'joint' column.
    target_col : str
        Target column name (default = "condition").
    
    Returns
    -------
    results : dict
        Dictionary with trained models and metrics per joint.
    """
    results = {}
    feature_cols = ["ROM", "mean_angle", "std_angle", "mean_velocity", "std_velocity", "max_velocity"]

    for joint_id in sorted(df["joint"].unique()):
        logger.info(f"=== Training SVM for joint {joint_id} ===")
        
        # Filter by joint
        df_joint = df[df["joint"] == joint_id]
        X = df_joint[feature_cols]
        y = df_joint[target_col]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # GridSearch with RBF kernel
        param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1, 1]}
        grid = GridSearchCV(
            SVC(kernel="rbf", random_state=42),
            param_grid,
            cv=3,
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        logger.info(f"Joint {joint_id} - Best params: {grid.best_params_}")
        logger.info(f"Joint {joint_id} - Accuracy: {acc:.3f}")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                logger.info(
                    f"  {label}: precision={metrics['precision']:.2f}, "
                    f"recall={metrics['recall']:.2f}, f1={metrics['f1-score']:.2f}"
                )

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=model.classes_,
                    yticklabels=model.classes_)
        plt.title(f"Confusion Matrix (Joint {joint_id})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

        # Save results
        results[joint_id] = {
            "model": model,
            "accuracy": acc,
            "report": report,
            "best_params": grid.best_params_
        }

    return results
