import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from utils import setup_logger

logger = setup_logger()


results_dir = Path("resources/results")

def plot_rom_vs_condition(df: pd.DataFrame, condition_col: str = "condition"):
    """
    Boxplot de ROM por condición.
    """
    plt.figure(figsize=(8,6))
    sns.boxplot(x=condition_col, y="ROM", data=df)
    plt.title("ROM distribution by condition")
    plt.xlabel("Condition")
    plt.ylabel("ROM")
    plt.savefig(results_dir / "ROM_condition.png", dpi=300, bbox_inches="tight")
    logger.info("kinematics saved")
    # plt.show()


def plot_rom_vs_velocity(df: pd.DataFrame, condition_col: str = "condition"):
    """
    Scatterplot de ROM vs velocidad angular, coloreado por condición.
    """
    plt.figure(figsize=(8,6))
    sns.scatterplot(x="condition", y="ROM", hue=condition_col, data=df, alpha=0.7)
    plt.title("ROM vs Velocidad Angular por condición")
    plt.xlabel("Condition")
    plt.ylabel("ROM")
    plt.legend(title="Condición")
    plt.savefig(results_dir / "ROM_velocity.png", dpi=300, bbox_inches="tight")
    logger.info("Rom vs velocity saved")
    # plt.show()

def plot_anglecorr_vs_condition(df: pd.DataFrame): 
    "Scatterplot of the angle correlation vs condition"

    plt.figure(figsize=(8,6))
    sns.boxplot(x="condition", y="angle_corr", data=df, showfliers=False)
    plt.title("Angle correlation vs condition")
    plt.xlabel("Condition")
    plt.ylabel("Angle correlation")
    # plt.legend(title="Condition")
    plt.savefig(results_dir / "anglecorr_condition.png", dpi=300, bbox_inches="tight")
    logger.info("angle corr vs condition saved")
    # plt.show()

def plot_kinematics_mean_std(df: pd.DataFrame, graph: str = "angle", condition_col: str = "condition"):
    """
    Subplots of kinematic curves (angular position vs time),
    showing mean ± std for each condition and joint.
    Left leg (leg=1) in red, right leg (leg=2) in green.
    Mean curve is darker, std band is lighter.
    """
    joints = sorted(df['joint'].unique())
    conditions = sorted(df[condition_col].unique())
    fig, axes = plt.subplots(len(joints), len(conditions),
                             figsize=(6*len(conditions), 4*len(joints)),
                             sharey=True)

    # Color mapping for legs
    leg_colors = {1: "red", 2: "green"}
    leg_names = {1: "Left", 2: "Right"}

    # Joint names
    joint_names = {1: "Ankle", 2: "Knee", 3: "Hip"}

    for i, joint in enumerate(joints):
        for j, cond in enumerate(conditions):
            ax = axes[i, j]
            for leg in sorted(df['leg'].unique()):
                subset = df[(df['joint'] == joint) & (df[condition_col] == cond) & (df['leg'] == leg)]
                grouped = subset.groupby("time")[graph].agg(["mean", "std"])

                color = leg_colors.get(leg, "blue")

                # Plot mean (darker line)
                ax.plot(grouped.index, grouped["mean"],
                        label=f"{leg_names[leg]} Mean", color=color, linewidth=2)

                # Plot std band (lighter fill)
                ax.fill_between(grouped.index,
                                grouped["mean"] - grouped["std"],
                                grouped["mean"] + grouped["std"],
                                color=color, alpha=0.3, label=f"{leg_names[leg]} ±1 std")

            ax.set_title(f"{joint_names.get(joint, joint)} - Condition {cond}")
            ax.set_xlabel("Time")
            if j == 0:  # only first column shows Y axis
                ax.set_ylabel("Angle")
            ax.legend()

    plt.suptitle("Kinematics: mean ± std by joint, condition, and leg")
    plt.tight_layout()
    plt.savefig(results_dir / f"kinematics_{graph}.png", dpi=300, bbox_inches="tight")
    logger.info("kinematics saved")
    # plt.show()

