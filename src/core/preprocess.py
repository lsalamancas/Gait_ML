from utils.logger import setup_logger
import pandas as pd
import numpy as np 
 
logger = setup_logger()

def compute_ROM(df:pd.DataFrame) -> pd.DataFrame:
    """
    This function finds the ROM of each joint of each leg from each subject
    """
    logger.info("Finding ROM from each joint")
    rom_df = (
        df
        .groupby(["subject", "condition", "replication", "leg", "joint"])["angle"]
        .agg(["min", "max"])
        .assign(ROM=lambda df: df["max"] - df["min"])
        .reset_index()
    )

    rom_df["ROM"] = rom_df["ROM"].round().astype(int)

    df = df.merge(
        rom_df[["subject", "condition", "replication", "leg", "joint", "ROM"]],
        on=["subject", "condition", "replication", "leg", "joint"],
        how="left"
    )

    return df

def compute_angular_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes angular velocity (Δθ/Δt) for each joint per subject, condition, replication, leg.
    Adds a new column 'angular_velocity' to the original DataFrame.
    """
    df["angular_velocity"] = (
        df.groupby(["subject", "condition", "replication", "leg", "joint"])
        .apply(lambda g: g.assign(
            angular_velocity=g["angle"].diff() / g["time"].diff()
        ))
        .reset_index(drop=True)["angular_velocity"]
    )

    df["angular_velocity"] = df["angular_velocity"].fillna(0)

    return df

def compute_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary metrics (mean, std, min, max) for angle and angular_velocity
    per subject, condition, replication, leg, joint.
    Adds these metrics as new columns merged into the original DataFrame.
    """
    logger.info("Computing summary metrics for angle and angular velocity")

    metrics_df = (
        df.groupby(["subject", "condition", "replication", "leg", "joint"])
        .agg(
            mean_angle=("angle", "mean"),
            std_angle=("angle", "std"),
            min_angle=("angle", "min"),
            max_angle=("angle", "max"),
            mean_velocity=("angular_velocity", "mean"),
            std_velocity=("angular_velocity", "std"),
            min_velocity=("angular_velocity", "min"),
            max_velocity=("angular_velocity", "max"),
        )
        .reset_index()
    )

    # Merge back into original df
    df = df.merge(metrics_df,
                  on=["subject", "condition", "replication", "leg", "joint"],
                  how="left")

    return df


def build_cycle_features(df: pd.DataFrame,
                         time_col="time", angle_col="angle", vel_col="angular_velocity") -> pd.DataFrame:
    """
    Builds a cycle-level feature DataFrame with aggregated metrics.
    Keeps unilateral features (per leg) and merges bilateral features (ROM_diff, angle_corr).
    """
    logger.info("Building cycle-level features")

    # --- Unilateral features (per leg) ---
    group_cols = ["subject", "condition", "replication", "leg", "joint"]
    agg_leg = df.groupby(group_cols).agg(
        min_angle=(angle_col, "min"),
        max_angle=(angle_col, "max"),
        mean_angle=(angle_col, "mean"),
        std_angle=(angle_col, "std"),
        mean_velocity=(vel_col, "mean"),
        std_velocity=(vel_col, "std"),
        max_velocity=(vel_col, "max"),
        ROM=("ROM", "first")
    ).reset_index()

    # --- Bilateral features (per joint, no leg) ---
    group_cols_bilat = ["subject", "condition", "replication", "joint"]
    agg_bilat = df.groupby(group_cols_bilat).agg(
        ROM_diff=("ROM_diff", "first"),
        angle_corr=("angle_corr", "first")
    ).reset_index()

    # --- Merge both ---
    df_feat = agg_leg.merge(agg_bilat, on=["subject", "condition", "replication", "joint"], how="left")

    return df_feat


def compute_bilateral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds bilateral features (e.g. ROM_diff) to df_feat by comparing left and right legs
    for each subject, condition, replication, joint.
    Returns df_feat with new columns like ROM_diff.
    """
    logger.info("Computing bilateral features (left vs right)")

    # Pivot to get left/right ROM side by side
    rom_pivot = (
        df.pivot_table(index=["subject", "condition", "replication", "joint"],
                            columns="leg", values="ROM")
        .rename(columns={1: "ROM_left", 2: "ROM_right"})
        .reset_index()
    )

    # Compute difference
    rom_pivot["ROM_diff"] = rom_pivot["ROM_left"] - rom_pivot["ROM_right"]

    # Merge back into df_feat
    df = df.merge(rom_pivot[["subject", "condition", "replication", "joint", "ROM_diff"]],
                            on=["subject", "condition", "replication", "joint"],
                            how="left")

    return df


def compute_leg_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Pearson correlation between left and right leg angle curves
    for each subject, condition, replication, and joint.
    Returns a DataFrame with one row per cycle and a new column 'angle_corr'.
    """
    logger.info("Computing left-right leg correlation per cycle")

    # Agrupar por ciclo
    group_cols = ["subject", "condition", "replication", "joint"]
    corr_rows = []

    for keys, group in df.groupby(group_cols):
        left = group[group["leg"] == 1].sort_values("time")["angle"].values
        right = group[group["leg"] == 2].sort_values("time")["angle"].values

        # Validar que ambas piernas tengan misma longitud
        if len(left) == len(right) and len(left) > 0:
            corr = np.corrcoef(left, right)[0, 1]
        else:
            corr = np.nan  # No se puede calcular

        corr_rows.append({
            "subject": keys[0],
            "condition": keys[1],
            "replication": keys[2],
            "joint": keys[3],
            "angle_corr": corr
        })

    # Merge directly into df_feat
    df = df.merge(pd.DataFrame(corr_rows), on=["subject", "condition", "replication", "joint"], how="left")

    return df
