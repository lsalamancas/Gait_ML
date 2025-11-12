from utils.logger import setup_logger
import pandas as pd
 
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
