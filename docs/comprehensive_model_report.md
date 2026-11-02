# 📘 Comprehensive Technical Model Report

This document provides a deep-dive technical description of the Artificial Intelligence models deployed in the Absenteeism Prediction System.

---

## 1. Employee Risk Classification Model

### 🎯 Objective
To classify individual employees into risk categories (**Low**, **Moderate**, **High**) based on their historical data and leave balances.

### 🧠 Model Architecture
*   **Algorithm**: **Gradient Boosting Classifier** (`sklearn.ensemble.GradientBoostingClassifier`)
*   **Type**: Ensemble Learning (Boosting).
*   **Mechanism**: Builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. It effectively combines weak learners (decision trees) into a strong predictive model.

### ⚙️ best Hyperparameters (Optimized)
*   **`n_estimators`**: 500 (Number of boosting stages/trees).
*   **`learning_rate`**: 0.05 (Shrinks the contribution of each tree).
*   **`max_depth`**: 5 (Limits the number of nodes in the tree to capture complex interactions).
*   **`random_state`**: 42 (Ensures reproducibility).

### 🧪 Input Features (13 Variables)
1.  **Demographics**: `Age`, `Gender_Code`, `Marital_Code`, `NumberOfDependents`.
2.  **Work Info**: `Shift_Code`, `Tenure_Years`.
3.  **Leave Balances**: `AnnualLeaveBalance`, `SickLeaveBalance`, `MaternityLeaveBalance`, `PaternityLeaveBalance`.
4.  **Engineered Features (The "Secret Sauce")**:
    *   **`TotalLeaveBalance`**: Aggregated sum of all available leaves.
    *   **`UtilizationRate`**: Ratio of `AbsentDays / TotalDaysEmployed`. High utilization indicates a habit of absence.
    *   **`Seniority`**: Interaction term `Age * Tenure` to weight long-standing employees differently.

### 📊 Performance Metrics
*   **Accuracy**: **~94%**
*   **Precision (High Risk)**: **0.96**
*   **Verdict**: The model is highly reliable at identifying potential absenteeism risks before they escalate.

---

## 2. Daily Absence Forecasting System

### 🎯 Objective
To predict the exact **count of absent employees** for specific future dates (Time-Series Regression).

### 🧠 Model Architectures
We implemented a competitive multi-model system. The dashboard automatically selects the best performer.

#### A. LSTM (Deep Learning) - *The Winner 🏆*
*   **Type**: Recurrent Neural Network (RNN) with Long Short-Term Memory units.
*   **Architecture**:
    *   **Input Layer**: Accepts 23 features per time step.
    *   **Hidden Layer 1**: 128 Neurons (Relu Activation) + Return Sequences.
    *   **Hidden Layer 2**: 64 Neurons (Relu Activation).
    *   **Hidden Layer 3**: 32 Neurons (Dense).
    *   **Output Layer**: 1 Neuron (Linear prediction).
*   **Why it works**: LSTM cells have "memory gates" that allow the model to learn long-term dependencies (e.g., monthly cycles) and short-term patterns (e.g., daily fluctuations) simultaneously.

#### B. XGBoost Regressor (The Challenger)
*   **Type**: Extreme Gradient Boosting.
*   **Hyperparameters**:
    *   `n_estimators`: 500
    *   `max_depth`: 7 (Aggressive depth for complex pattern matching)
    *   `learning_rate`: 0.01

#### C. Hybrid Ensemble
*   **Method**: A weighted average comparison of LSTM and XGBoost predictions.
*   **Logic**: `Final_Prediction = (LSTM_Pred + XGB_Pred) / 2`.
*   **Role**: Provides stability if one model behaves erratically.

### 🧪 Input Features (23 Variables)
The models clearly see the "Context" of every single day:
1.  **Temporal**: `Year`, `Month`, `Day`, `DayOfWeek`, `IsWeekend`.
2.  **Cyclical Encoding**: `Month_Sin`, `Month_Cos`, `Day_Sin`, `Day_Cos` (Mathematical representation of time cycles).
3.  **Granular History (Lags)**:
    *   `Lag_1` to `Lag_7`: Exact absence counts for the past 7 days.
    *   `Lag_14`, `Lag_30`: Absences 2 weeks and 1 month ago.
4.  **Trends**: `RollingMean_7`, `RollingMean_30`.
5.  **Volatility**: `RollingStd_7`, `RollingStd_30` (Standard Deviation).
6.  **Velocity**: `Diff_1` (Rate of change from yesterday).

### 📊 Performance Metrics (Test Set)
*   **LSTM Evaluation**:
    *   **Mean Absolute Error (MAE)**: **0.37** (The prediction is off by less than 0.5 people on average).
    *   **R² Score**: **0.98** (Explains 98% of the variance in the data).

---

## 3. Comparative Summary

| Feature | Risk Classification | Daily Forecasting |
| :--- | :--- | :--- |
| **Primary Goal** | Classify Employee | Predict Counts |
| **Model Used** | Gradient Boosting | LSTM (Deep Learning) |
| **Input Data** | Static (Employee Profile) | Sequential (Time-Series) |
| **Key Metric** | Accuracy (94%) | MAE (0.37) |
