import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

DATA_PATH = "data/insurance.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

def main():
    data = pd.read_csv(DATA_PATH)
    target = 'charges'
    X = data.drop(columns=[target])
    y = data[target]

    # Ensure categorical types
    X['sex'] = X['sex'].astype('object')
    X['smoker'] = X['smoker'].astype('object')
    X['region'] = X['region'].astype('object')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing
    num_features = ['age', 'bmi', 'children']
    cat_features = ['sex', 'smoker', 'region']

    num_pipeline = Pipeline([('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    # Define RidgeCV with alpha tuning
    ridgecv = RidgeCV(
        alphas=np.logspace(-3, 3, 20),  # from 0.001 to 1000
        cv=5,
        scoring='r2'
    )

    pipeline = Pipeline([
        ('pre', preprocessor),
        ('ridge', ridgecv)
    ])

    # Fit model
    pipeline.fit(X_train, y_train)

    best_alpha = pipeline.named_steps['ridge'].alpha_
    print(f"Best alpha from RidgeCV: {best_alpha:.5f}")

    # Evaluate
    y_pred_test = pipeline.predict(X_test)
    metrics = regression_metrics(y_test, y_pred_test)
    print("Test metrics:", metrics)

    # Save model
    joblib.dump(pipeline, f"{MODELS_DIR}/ridgecv_best_model.joblib")
    print(f"✅ Saved RidgeCV best model to {MODELS_DIR}/ridgecv_best_model.joblib")

if __name__ == "__main__":
    main()