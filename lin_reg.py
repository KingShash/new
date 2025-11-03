"""
MLOps Style Linear Regression Pipeline
Includes:
- Data loading / splitting
- Model training & evaluation
- Model artifact saving
- Logging for CI/CD pipelines
"""

import os
import joblib
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.datasets import make_regression

# =======================
# Logging Setup
# =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.info("Pipeline started...")

# =======================
# Data Generation / Loading
# =======================
logging.info("Generating dataset...")
X, y = make_regression(
    n_samples=200,
    n_features=2,
    noise=15,
    random_state=42
)

# Split data
logging.info("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# =======================
# Model Training
# =======================
logging.info("Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

# =======================
# Evaluation
# =======================
logging.info("Evaluating model...")
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

logging.info(f"R2 Score: {r2:.4f}")
logging.info(f"MAE: {mae:.4f}")

# =======================
# Save Model for Deployment
# =======================
os.makedirs("models", exist_ok=True)
model_path = "models/linear_model.pkl"
joblib.dump(model, model_path)

logging.info(f"Model saved to: {model_path}")
logging.info("Pipeline completed successfully ✅")
