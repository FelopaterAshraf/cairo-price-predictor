import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.utils import load_data, clean_and_engineer, split_features_target

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    df = load_data()
    df = clean_and_engineer(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    }

    results = []

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        r2_pct = r2 * 100

        filename = f"{name}_{r2_pct:.2f}.joblib"
        joblib.dump(pipe, os.path.join(MODELS_DIR, filename))

        print(f"{name}: MAE={mae:,.0f}, RMSE={rmse:,.0f}, R2={r2:.4f} -> saved {filename}")
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

    pd.DataFrame(results).to_csv(os.path.join(MODELS_DIR, "model_summary.csv"), index=False)
    print("Training complete — models saved in 'models/'")

if __name__ == "__main__":
    main()
