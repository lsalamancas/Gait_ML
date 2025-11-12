from core.read_data import read_data
from core.preprocess import compute_ROM, compute_angular_velocity
from utils.logger import setup_logger
from models.linear_regression import train_linear_regression
from models.random_forest import train_random_forest
from models.knn import train_knn
import matplotlib.pyplot as plt
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

    # 4. (Opcional) Save enriched DataFrame for inspection
    df.to_csv("resources/enriched_gait.csv", index=False)
    logger.info("Enriched DataFrame saved")

    lr_model = train_linear_regression(df)
    rf_model = train_random_forest(df)
    knn_model = train_knn(df, target_col="condition_code", n_neighbors=5)





if __name__ == "__main__":
    main()
