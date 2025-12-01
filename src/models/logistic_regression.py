import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from utils import setup_logger

results_dir = Path("resources/results")
logger = setup_logger()


def train_multiclass_logreg(df: pd.DataFrame,
                            target_col: str = "condition",
                            test_size: float = 0.2,
                            random_state: int = 42):
    feature_cols = ["ROM", "mean_angle", "std_angle", "mean_velocity", "std_velocity", "max_velocity", "ROM_diff", "angle_corr"]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    logger.info(f"Accuracy: {acc:.3f} | Macro-F1: {f1:.3f}")
    logger.info(classification_report(y_test, y_pred))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion matrix (LogReg multinomial)")
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(results_dir / "lgreg_cf.png", dpi=300, bbox_inches="tight")
    logger.info("lgred cf saved")
    # plt.show()

    return {"model": model, "scaler": scaler, "accuracy": acc, "f1": f1}


def train_per_joint_classifier(df_features: pd.DataFrame,
                               joint_id: int,
                               target_col="condition",
                               n_splits=5,
                               scale=True):
    """
    Train and evaluate a multi-class classifier ONLY for one joint (joint_id).
    Use subject-stratified cross-validation to prevent information leakage.
    """
    data = df_features[df_features["joint"] == joint_id].copy()
    feature_cols = ["ROM","mean_angle","std_angle","mean_velocity","std_velocity","max_velocity", "ROM_diff", "angle_corr"] 
    X = data[feature_cols].values
    y = data[target_col].values
    groups = data["subject"].values

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    cms = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)

    for train_idx, test_idx in cv.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if scale:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro"))
        cms += confusion_matrix(y_te, y_pred, labels=np.unique(y))

    logger.info(f"Joint {joint_id}: Accuracy CV mean={np.mean(accs):.3f} | Macro-F1 CV mean={np.mean(f1s):.3f}")

    # Plot matriz de confusión acumulada
    plt.figure(figsize=(6,5))
    sns.heatmap(cms, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Cumulative confusion matrix (joint {joint_id})")
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(results_dir / f"lgreg_cf_byjoint{joint_id}.png", dpi=300, bbox_inches="tight")
    logger.info("lgred cf by joint saved")
    # plt.show()
    return {"acc_mean": np.mean(accs), "f1_mean": np.mean(f1s), "cm": cms}