import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.decomposition import PCA
from utils.logger import setup_logger

logger = setup_logger()
results_dir = Path("resources/results")


def run_kmeans(df: pd.DataFrame, target_col: str = "condition", n_clusters: int = 3):
    """
    Run KMeans clustering on aggregated gait features.
    Scales features, fits KMeans, evaluates inertia and silhouette score,
    compares clusters vs true labels, and generates PCA visualizations.
    """
    feature_cols = ["angle_corr"]
    X = df[feature_cols].values
    y = df[target_col].values

    # --- Scale features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Fit KMeans ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # --- Evaluation metrics ---
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, clusters)
    logger.info(f"KMeans inertia: {inertia:.2f}")
    logger.info(f"KMeans silhouette score: {silhouette:.3f}")

    # --- Confusion matrix (true labels vs clusters) ---
    cm = confusion_matrix(y, clusters)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix: True Condition vs KMeans Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Condition")
    plt.tight_layout()
    plt.savefig(results_dir / "kmeans_cf.png", dpi=300, bbox_inches="tight")
    logger.info("kmeans cf saved")
    # plt.show()

    # --- PCA 2D visualization of clusters ---
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X_scaled)
    # plt.figure(figsize=(8,6))
    # plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="Set1", alpha=0.7)
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.title("KMeans Clusters (PCA 2D)")
    # plt.tight_layout()
    # plt.savefig(results_dir / "kmeans_clusters.png", dpi=300, bbox_inches="tight")
    # logger.info("kmeans clusters saved")
    # plt.show()

    return kmeans, clusters

def run_kmeans_by_joint(df: pd.DataFrame, target_col: str = "condition", n_clusters: int = 3):
    """
    Run KMeans clustering separately for each joint.
    Scales features, fits KMeans, evaluates inertia and silhouette score,
    compares clusters vs true labels, and generates PCA visualizations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe including 'joint' column.
    target_col : str
        Target column name (default = "condition").
    n_clusters : int
        Number of clusters for KMeans (default = 3).
    
    Returns
    -------
    results : dict
        Dictionary with trained KMeans models and metrics per joint.
    """
    results = {}
    feature_cols = ["angle_corr"]

    for joint_id in sorted(df["joint"].unique()):
        logger.info(f"=== Running KMeans for joint {joint_id} ===")

        # Filter by joint
        df_joint = df[df["joint"] == joint_id]
        X = df_joint[feature_cols].values
        y = df_joint[target_col].values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Evaluation metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_scaled, clusters)
        logger.info(f"Joint {joint_id} - KMeans inertia: {inertia:.2f}")
        logger.info(f"Joint {joint_id} - KMeans silhouette score: {silhouette:.3f}")

        # Confusion matrix (true labels vs clusters)
        cm = confusion_matrix(y, clusters)
        plt.figure(figsize=(6,5))
        contingency = pd.crosstab(y, clusters)
        sns.heatmap(contingency, annot=True, cmap="Blues")

        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix: Condition vs Cluster (Joint {joint_id})")
        plt.xlabel("Cluster")
        plt.ylabel("Condition")
        plt.tight_layout()
        plt.savefig(results_dir /f"kmeans_cf_byjoint{joint_id}.png", dpi=300, bbox_inches="tight")
        logger.info("kmeans cf by joint saved")
        plt.show()

        # PCA 2D visualization of clusters
        # pca = PCA(n_components=2)
        # X_pca = pca.fit_transform(X_scaled)
        # plt.figure(figsize=(8,6))
        # plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="Set1", alpha=0.7)
        # plt.xlabel("PC1")
        # plt.ylabel("PC2")
        # plt.title(f"KMeans Clusters (PCA 2D - Joint {joint_id})")
        # plt.tight_layout()
        # plt.show()

        # Save results
        results[joint_id] = {
            "model": kmeans,
            "clusters": clusters,
            "inertia": inertia,
            "silhouette": silhouette
        }

    return results
