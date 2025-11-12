import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utils.logger import setup_logger

logger = setup_logger()

def train_linear_regression(df: pd.DataFrame) -> LinearRegression:
    """
    Trains a linear regression model using enriched gait data.
    Features include ROM, angular_velocity, joint, leg, and replication.
    Assumes target_col is already numeric.
    """
    logger.info("Preparing data for linear regression")

    # Select features
    feature_cols = ["ROM", "angular_velocity", "angle"]
    X = df[feature_cols]
    y = df['condition']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predict and evaluate
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    logger.info(f"Linear regression trained. MSE: {mse:.3f}, R²: {r2:.3f}")
    logger.info(f"Coefficients: {dict(zip(feature_cols, model.coef_))}")

    return model
