import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
# Import tensorflow only if needed to avoid lag, but app needs it for LSTM loading
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Apparel Industry Absenteeism Predictor", layout="wide")

# Load Models & Metrics
@st.cache_resource
def load_all_models():
    try:
        clf = joblib.load("models/classifier.pkl")
        le_gender = joblib.load("models/le_gender.pkl")
        le_marital = joblib.load("models/le_marital.pkl")
        le_shift = joblib.load("models/le_shift.pkl")
        
        # Forecasting Models
        try:
            scaler_X = joblib.load("models/scaler_X.pkl")
            scaler_y = joblib.load("models/scaler_y.pkl")
            gb_model = joblib.load("models/gb_model.pkl")
            xgb_model = joblib.load("models/xgb_model.pkl")
            lstm_model = load_model("models/lstm_model.keras")
            
            with open("models/model_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception as e:
            st.error(f"Error loading forecasting models: {e}")
            return clf, le_gender, le_marital, le_shift, None, None, None, None, None, None

        return clf, le_gender, le_marital, le_shift, scaler_X, scaler_y, gb_model, xgb_model, lstm_model, metrics
    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        return None, None, None, None, None, None, None, None, None, None

clf, le_gender, le_marital, le_shift, scaler_X, scaler_y, gb_model, xgb_model, lstm_model, metrics = load_all_models()

# Load Data for Visualization
@st.cache_data
def load_data():
    try:
        df_emp = pd.read_csv("processed/employee_risk.csv")
        df_daily = pd.read_csv("processed/daily_forecast.csv")
        # Ensure date format
        df_daily['AbsenceDate'] = pd.to_datetime(df_daily['AbsenceDate'])
        return df_emp, df_daily
    except:
        return pd.DataFrame(), pd.DataFrame()

df_emp, df_daily = load_data()

st.title("👕 Apparel Industry Absenteeism Prediction Dashboard")

if clf is None:
    st.error("Models not found! Please run training scripts first.")
else:
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "👤 Employee Risk Prediction", "📅 Smart Forecasting"])

    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.header("Workforce Absenteeism Overview")
        if not df_emp.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Employees", len(df_emp))
            col2.metric("High Risk Employees", len(df_emp[df_emp['RiskCategory'] == 'High']))
            col3.metric("Avg Tenure", f"{df_emp['Tenure_Years'].mean():.1f} Years")

            st.subheader("Risk Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df_emp, x='RiskCategory', order=['Low', 'Moderate', 'High'], palette='viridis', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No employee data found.")

    # --- TAB 2: EMPLOYEE RISK PREDICTION ---
    with tab2:
        st.header("Predict Employee Risk")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 65, 30)
            gender = st.selectbox("Gender", le_gender.classes_)
            marital = st.selectbox("Marital Status", le_marital.classes_)
            dependents = st.number_input("Number of Dependents", 0, 10, 1)
            shift = st.selectbox("Shift", le_shift.classes_)
        
        with col2:
            tenure = st.number_input("Tenure (Years)", 0, 40, 5)
            annual_bal = st.number_input("Annual Leave Balance", 0.0, 30.0, 10.0)
            sick_bal = st.number_input("Sick Leave Balance", 0.0, 30.0, 5.0)
            maternity_bal = st.number_input("Maternity Leave Balance", 0, 100, 0)
            paternity_bal = st.number_input("Paternity Leave Balance", 0, 100, 0)

        if st.button("Predict Risk"):
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender_Code': [le_gender.transform([gender])[0]],
                'Marital_Code': [le_marital.transform([marital])[0]],
                'NumberOfDependents': [dependents],
                'Shift_Code': [le_shift.transform([shift])[0]],
                'Tenure_Years': [tenure],
                'AnnualLeaveBalance': [annual_bal],
                'SickLeaveBalance': [sick_bal],
                'MaternityLeaveBalance': [maternity_bal],
                'PaternityLeaveBalance': [paternity_bal]
            })
            
            prediction = clf.predict(input_data)[0]
            probs = clf.predict_proba(input_data)[0]
            
            st.subheader(f"Risk Category: **{prediction}**")
            st.write(f"Confidence: {max(probs):.2%}")
            
            if prediction == 'High':
                st.error("⚠️ HIGH RISK: Proactive engagement recommended.")
            elif prediction == 'Moderate':
                st.warning("⚠️ MODERATE RISK: Monitor trends.")
            else:
                st.success("✅ LOW RISK: Normal activity.")

    # --- TAB 3: SMART FORECASTING ---
    with tab3:
        st.header("Advanced Forecasting System")
        
        # Display Model Comparison
        if metrics:
            st.subheader("Model Performance (MAE - Lower is Better)")
            st.info("We compared multiple advanced models to find the most accurate predictor.")
            
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Model', 'MAE']).sort_values('MAE')
            best_model_name = metrics_df.iloc[0]['Model']
            
            col_m1, col_m2 = st.columns([1, 2])
            with col_m1:
                st.table(metrics_df)
                st.success(f"Best Model Selected: **{best_model_name}**")
            
            with col_m2:
                fig_perf, ax_perf = plt.subplots(figsize=(8, 4))
                sns.barplot(data=metrics_df, x='MAE', y='Model', palette='magma', ax=ax_perf)
                ax_perf.set_title("Model Error Comparison")
                st.pyplot(fig_perf)
        
        # Forecasting Execution
        st.divider()
        st.subheader("Future Absence Forecast")
        
        if not df_daily.empty and scaler_X:
            days_to_predict = st.slider("Days to Forecast", 7, 30, 14)
            last_date = df_daily['AbsenceDate'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
            
            # Recursive Forecasting Logic using the Best Model (or Hybrid Logic)
            future_preds = []
            temp_history = df_daily['AbsentCount'].tolist()
            
            for date in future_dates:
                # Construct features
                day_of_week = date.dayofweek
                is_weekend = 1 if day_of_week >= 5 else 0
                month = date.month
                day = date.day
                year = date.year
                
                lag_1 = temp_history[-1]
                lag_7 = temp_history[-7]
                roll_7 = np.mean(temp_history[-7:])
                
                ft_vector = np.array([[year, month, day, day_of_week, is_weekend, lag_1, lag_7, roll_7]])
                ft_scaled = scaler_X.transform(ft_vector)
                
                # Predict based on selected best model
                if best_model_name == 'Hybrid':
                    # Hybrid = (XGB + LSTM) / 2
                    xgb_p_scaled = xgb_model.predict(ft_scaled)
                    lstm_p_scaled = lstm_model.predict(ft_scaled.reshape(1, 1, 8), verbose=0)
                    pred_scaled = (xgb_p_scaled + lstm_p_scaled) / 2
                elif best_model_name == 'XGBoost':
                    pred_scaled = xgb_model.predict(ft_scaled)
                elif best_model_name == 'GradientBoosting':
                    pred_scaled = gb_model.predict(ft_scaled)
                elif best_model_name == 'LSTM':
                    pred_scaled = lstm_model.predict(ft_scaled.reshape(1, 1, 8), verbose=0)
                else: 
                    # Default fallback
                    pred_scaled = xgb_model.predict(ft_scaled)
                
                # Inverse scale
                pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                
                temp_history.append(pred)
                future_preds.append({'Date': date, 'PredictedAbsences': max(0, int(round(pred)))})
            
            df_future = pd.DataFrame(future_preds)
            
            st.line_chart(df_future.set_index('Date')['PredictedAbsences'])
            st.dataframe(df_future)
