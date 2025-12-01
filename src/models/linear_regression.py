import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from utils.logger import setup_logger
import matplotlib.pyplot as plt

logger = setup_logger()

def train_linear_regression(df: pd.DataFrame,
                            target_col: str = "condition",
                            test_size: float = 0.2,
                            random_state: int = 42):
    """
    Trains a linear regression model using enriched gait data.
    Features include ROM, mean_angle, std_angle, mean_velocity, max_velocity.
    Splits data into train/test sets, evaluates performance, and plots results.
    """

    logger.info("Preparing data for linear regression")

    # Features (mejor usar métricas agregadas en lugar de ángulo crudo)
    feature_cols = ["ROM_diff", "angle_corr"]
    X = df[feature_cols]
    y = df[target_col]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Linear regression test MSE: {mse:.3f}, R²: {r2:.3f}")
    logger.info(f"Coefficients: {dict(zip(feature_cols, model.coef_))}")
    
    # Plot resultados
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "r--", lw=2, label="Perfect fit")
    plt.xlabel("Valores reales")
    plt.ylabel("Predicciones")
    plt.title("Regresión lineal: reales vs predichos (test set)")
    plt.legend()
    plt.show()


    return {"model": model, "mse": mse, "r2": r2}