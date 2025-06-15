# scripts/model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import pickle
def train_model(df, feature_cols, target_col='price', model_path="model.pkl"):
    """
    Train a Random Forest Regressor and save the model to disk.
    """
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_path)
    #save features used for training
    # During training
    features = X.columns.tolist()
    with open("data/model_features.pkl", "wb") as f:
        pickle.dump(features, f)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Model trained. RMSE: {rmse:.2f}, R²: {r2:.2f}")
    return model, X_train

def predict_and_save_undervalued(df, model, feature_cols, output_path="undervalued_listings.csv"):
    """
    Predict prices, compute undervaluation, and save undervalued listings.
    """
    df['predicted_price'] = model.predict(df[feature_cols])
    df['undervaluation'] = df['predicted_price'] - df['price']
    undervalued = df[df['undervaluation'] > 0].sort_values(by='undervaluation', ascending=False)

    undervalued.to_csv(output_path, index=False)
    return undervalued
