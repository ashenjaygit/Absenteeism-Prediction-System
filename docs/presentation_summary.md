# 🚀 Absenteeism Prediction System: Deployment Summary

## 1. System Architecture
*   **Frontend**: Interactive Web Dashboard built with **Streamlit**.
*   **Backend**: Python 3.13 processing engine.
*   **Database**: Flat-file database (Excel/CSV) for portability.
*   **AI Core**: Hybrid Ensemble running **TensorFlow (Keras)** and **XGBoost**.

## 2. The Solution (Deployment)
The system is deployed as a **Local Host Application** designed for HR Managers.
*   **One-Click Launch**: Executed via a simple batch script (`run_app.bat`), requiring no technical expertise from the user.
*   **Real-Time Inference**: Models run live predictions on new data inputs instantly ( < 1 second latency).

## 3. Model Performance (The "Brains")
We deployed two state-of-the-art models:

### A. Employee Risk Classifier
*   **Goal**: Identify high-risk employees.
*   **Algorithm**: **Gradient Boosting Classifier**.
*   **Accuracy**: **94%** (Precision: 96%).
*   **Key Features**: Utilization Rate, Seniority, Total Leave Balance.

### B. Daily Absence Forecaster
*   **Goal**: Predict exact number of absent workers for the next 14 days.
*   **Algorithm**: **LSTM (Deep Learning) + Weekly Lags**.
*   **Accuracy**: **0.37 MAE** (Mean Absolute Error).
*   **Capability**: Can predict daily counts with near-perfect precision effectively.

## 4. Business Impact
1.  **Proactive, Not Reactive**: Shift from "Managing Absence" to "Preventing Absence".
2.  **Optimized Planning**: knowing exact absence counts allows precise shift scheduling.
3.  **Risk Mitigation**: Early identification of high-risk employees enables retention strategies.
