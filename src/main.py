from core.read_data import read_data
from core.preprocess import *
from utils.logger import setup_logger
from models.linear_regression import train_linear_regression
from models.random_forest import train_random_forest
from models.knn import train_knn
from models.svm import train_svm, train_svm_by_joint
from models.logistic_regression import train_multiclass_logreg, train_per_joint_classifier
from models.kmeans import run_kmeans, run_kmeans_by_joint
from models.MLP import train_mlp
from utils import setup_logger, plot_kinematics_mean_std, plot_rom_vs_condition, plot_rom_vs_velocity, plot_anglecorr_vs_condition

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

logger = setup_logger()

def main():
    logger.info("Starting gait analysis pipeline")

    # 1. Load data
    df = read_data()  # ajusta el path si es necesario
    logger.info(f"Data loaded with shape: {df.shape}")

    # 2. Compute ROM and merge into df
    df = compute_ROM(df)
    logger.info("ROM added to DataFrame")

    # 3. Compute angular velocity and add to df
    df = compute_angular_velocity(df)
    logger.info("Angular velocity added to DataFrame")

    df = compute_summary_metrics(df)
    logger.info("Metrics added to DataFrame")
    # 4. (Optional) Save enriched DataFrame for inspection
    df.to_csv("resources/enriched_gait.csv", index=False)
    logger.info("Enriched DataFrame saved")


    #5. compute bilateral features
    df_feat = compute_bilateral_features(df)
    df_feat = compute_leg_correlation(df_feat)
    df_feat = build_cycle_features(df_feat)

    #6. (Optional) Save features df 
    df_feat.to_csv("resources/features_gait.csv", index=False)
    logger.info("Enriched DataFrame saved")
    logger.info(df_feat.head())

    # # #7. run models 
    # # lr_model = train_linear_regression(df_feat)
    # logreg_model = train_multiclass_logreg(df_feat)
    # res_ankle = train_per_joint_classifier(df_feat, joint_id=3)  # ankle
    # res_knee  = train_per_joint_classifier(df_feat, joint_id=2)  # knee
    # res_hip   = train_per_joint_classifier(df_feat, joint_id=1)  # hip

    # knn_model = train_knn(df_feat, target_col="condition", n_neighbors=5)
    # kmeans_model, clusters = run_kmeans(df_feat, target_col="condition", n_clusters=3)
    # results = run_kmeans_by_joint(df_feat, target_col="condition", n_clusters=3)

    rf_model = train_random_forest(df_feat)

    # svm_model = train_svm(df_feat, target_col="condition") 
    # svm_model_joint = train_svm_by_joint(df_feat, target_col="condition") 

    # MLP = train_mlp(df)

    # # #plot_data
    # plot_rom_vs_condition(df)
    # plot_rom_vs_velocity(df)
    # plot_kinematics_mean_std(df, graph="angle")
    # plot_kinematics_mean_std(df, "angular_velocity")

    # plot_anglecorr_vs_condition(df_feat)


if __name__ == "__main__":
    main()
