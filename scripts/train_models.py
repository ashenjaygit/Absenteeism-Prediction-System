import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def train_models():
    # Ensure models directory exists
    if not os.path.exists("models"):
        os.makedirs("models")

    # ==========================================
    # 1. Employee Risk Classification Model
    # ==========================================
    print("\n--- Training Employee Risk Classification Model ---")
    try:
        df_emp = pd.read_csv("processed/employee_risk.csv")
    except FileNotFoundError:
        print("Error: processed/employee_risk.csv not found.")
        return

    # Features and Target
    # Using Demographics and Current Leave Balances to predict Risk Category
    # Note: In a real temporal setup, we'd use past balances to predict future risk.
    # Here we are mapping attributes to the risk level derived from history.
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    df_emp['Gender_Code'] = le_gender.fit_transform(df_emp['Gender'])
    
    le_marital = LabelEncoder()
    df_emp['Marital_Code'] = le_marital.fit_transform(df_emp['MaritalStatus'])
    
    le_shift = LabelEncoder()
    df_emp['Shift_Code'] = le_shift.fit_transform(df_emp['Shift'])

    features = [
        'Age', 'Gender_Code', 'Marital_Code', 'NumberOfDependents', 
        'Shift_Code', 'Tenure_Years', 
        'AnnualLeaveBalance', 'SickLeaveBalance', 'MaternityLeaveBalance', 'PaternityLeaveBalance'
    ]
    target = 'RiskCategory'

    X = df_emp[features]
    y = df_emp[target]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save Model and Encoders
    joblib.dump(clf, "models/classifier.pkl")
    joblib.dump(le_gender, "models/le_gender.pkl")
    joblib.dump(le_marital, "models/le_marital.pkl")
    joblib.dump(le_shift, "models/le_shift.pkl")
    print("Saved classification models.")

    # ==========================================
    # 2. Daily Absence Forecasting Model
    # ==========================================
    print("\n--- Training Daily Absence Forecasting Model ---")
    try:
        df_daily = pd.read_csv("processed/daily_forecast.csv")
    except FileNotFoundError:
        print("Error: processed/daily_forecast.csv not found.")
        return

    # Features and Target
    features_ts = ['Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend', 'Lag_1', 'Lag_7', 'RollingMean_7']
    target_ts = 'AbsentCount'

    X_ts = df_daily[features_ts]
    y_ts = df_daily[target_ts]

    # Time-based split (don't shuffle time series)
    split_point = int(len(df_daily) * 0.8)
    X_train_ts, X_test_ts = X_ts.iloc[:split_point], X_ts.iloc[split_point:]
    y_train_ts, y_test_ts = y_ts.iloc[:split_point], y_ts.iloc[split_point:]

    # Train Model
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train_ts, y_train_ts)

    # Evaluate
    y_pred_ts = reg.predict(X_test_ts)
    mae = mean_absolute_error(y_test_ts, y_pred_ts)
    rmse = np.sqrt(mean_squared_error(y_test_ts, y_pred_ts))
    
    print(f"Forecasting MAE: {mae:.2f}")
    print(f"Forecasting RMSE: {rmse:.2f}")
    print(f"Mean Daily Absences (Test Set): {y_test_ts.mean():.2f}")

    # Save Model
    joblib.dump(reg, "models/forecaster.pkl")
    print("Saved forecasting model.")

if __name__ == "__main__":
    train_models()
