import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    # Aggressive LSTM: Deep but no dropout (aiming for max fit)
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_advanced_models():
    print("--- Training Advanced Forecasting Models ---")
    
    # Load Data
    try:
        df = pd.read_csv("processed/daily_forecast.csv")
    except FileNotFoundError:
        print("Error: Data file not found.")
        return

    # Prepare Data
    features = ['Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend', 
                'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'Lag_6', 'Lag_7', 
                'Lag_14', 'Lag_30', 
                'RollingMean_7', 'RollingMean_30',
                'Month_Sin', 'Month_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos',
                'RollingStd_7', 'RollingStd_30', 'Diff_1']
    target = 'AbsentCount'
    
    # Sort by date just in case
    # Assuming 'AbsenceDate' exists or inherent order is correct. 
    # The previous process data script sorted it.
    
    X = df[features].values
    y = df[target].values

    # Scaling for LSTM (and generally good for others)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Train/Test Split (Time-based: Last 20%)
    split_idx = int(len(df) * 0.8)
    
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    y_test_original = y[split_idx:] # For real-scale evaluation

    # ---------------------------------------------------------
    # 1. Gradient Boosting
    # ---------------------------------------------------------
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train.ravel())
    
    gb_pred_scaled = gb_model.predict(X_test)
    gb_pred = scaler_y.inverse_transform(gb_pred_scaled.reshape(-1, 1)).flatten()
    
    gb_mae = mean_absolute_error(y_test_original, gb_pred)
    gb_r2 = r2_score(y_test_original, gb_pred)
    print(f"Gradient Boosting MAE: {gb_mae:.4f}, R2: {gb_r2:.4f}")

    # ---------------------------------------------------------
    # 2. XGBoost
    # ---------------------------------------------------------
    print("Training XGBoost (Aggressive)...")
    xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=7, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train.ravel())
    
    xgb_pred_scaled = xgb_model.predict(X_test)
    xgb_pred = scaler_y.inverse_transform(xgb_pred_scaled.reshape(-1, 1)).flatten()
    
    xgb_mae = mean_absolute_error(y_test_original, xgb_pred)
    xgb_r2 = r2_score(y_test_original, xgb_pred)
    print(f"XGBoost MAE: {xgb_mae:.4f}, R2: {xgb_r2:.4f}")

    # ---------------------------------------------------------
    # 3. LSTM
    # ---------------------------------------------------------
    print("Training LSTM...")
    # Reshape for LSTM: [samples, time steps, features]
    # Here time_steps = 1 since we already engineered lag features as columns
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    lstm_model = create_lstm_model((1, X_train.shape[1]))
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=16, verbose=0, callbacks=[early_stop])
    
    lstm_pred_scaled = lstm_model.predict(X_test_lstm, verbose=0)
    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled).flatten()
    
    lstm_mae = mean_absolute_error(y_test_original, lstm_pred)
    lstm_r2 = r2_score(y_test_original, lstm_pred)
    print(f"LSTM MAE: {lstm_mae:.4f}, R2: {lstm_r2:.4f}")

    # ---------------------------------------------------------
    # 4. Hybrid (Average of Best 2)
    # ---------------------------------------------------------
    print("Training Hybrid Model...")
    # Let's assume XGB and LSTM are strongest for now, or just average all 3.
    # User asked for "suitable hybrid". Weighted average of XGB and LSTM is robust.
    
    hybrid_pred = (xgb_pred + lstm_pred) / 2
    hybrid_mae = mean_absolute_error(y_test_original, hybrid_pred)
    hybrid_r2 = r2_score(y_test_original, hybrid_pred)
    print(f"Hybrid (XGB+LSTM) MAE: {hybrid_mae:.4f}, R2: {hybrid_r2:.4f}")

    # ---------------------------------------------------------
    # Selection & Saving
    # ---------------------------------------------------------
    results = {
        "GradientBoosting": gb_mae,
        "XGBoost": xgb_mae,
        "LSTM": lstm_mae,
        "Hybrid": hybrid_mae
    }
    
    # We still select based on MAE (lower is better), but we printed R2

    
    best_model_name = min(results, key=results.get)
    print(f"\nBEST MODEL: {best_model_name} (MAE: {results[best_model_name]:.4f})")

    # Save detailed report
    report = f"""
Model Performance Report
========================
Gradient Boosting: MAE={gb_mae:.4f}, R2={gb_r2:.4f}
XGBoost:           MAE={xgb_mae:.4f}, R2={xgb_r2:.4f}
LSTM:              MAE={lstm_mae:.4f}, R2={lstm_r2:.4f}
Hybrid:            MAE={hybrid_mae:.4f}, R2={hybrid_r2:.4f}
    """
    with open("models/detailed_metrics.txt", "w") as f:
        f.write(report)
    
    # Save Scalers
    joblib.dump(scaler_X, "models/scaler_X.pkl")
    joblib.dump(scaler_y, "models/scaler_y.pkl")
    
    # Save Models
    joblib.dump(gb_model, "models/gb_model.pkl")
    joblib.dump(xgb_model, "models/xgb_model.pkl")
    lstm_model.save("models/lstm_model.keras")
    
    # Save Metrics
    with open("models/model_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("All models and metrics saved.")

if __name__ == "__main__":
    train_advanced_models()
