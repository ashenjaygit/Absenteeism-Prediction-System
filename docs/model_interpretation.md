# Model Interpretation Guide

This document provides a full technical and business interpretation of the two machine learning models used in the Absenteeism Prediction System.

---

## 1. Scenario A: Employee Risk Prediction (Classification)

### **Objective**
To categorize individual employees into risk groups (**Low**, **Moderate**, **High**) to identify who is most likely to be absent frequently.

### **The Model: Random Forest Classifier**
We used a **Random Forest Classifier**, which is an ensemble learning method.
*   **How it works**: It builds hundreds of small "Decision Trees". Each tree looks at a random subset of the data and makes a vote on whether an employee is High, Medium, or Low risk. The final prediction is determined by the majority vote of all trees.
*   **Why this model**: It is highly accurate, resistant to overfitting, and handles a mix of numerical (Age, Tenure) and categorical (Gender, Shift) data excellently.

### **Data & Logic**
1.  **Inputs (Independent Variables)**:
    *   **Demographics**: Age, Gender, Marital Status, Number of Dependents.
    *   **Job Details**: Shift, Tenure (Years of Service).
    *   **Leave Balances**: Annual, Sick, Maternity, Paternity leave balances.
2.  **Logic**:
    *   The model finds non-linear patterns. For example, it might learn: *"Employees with Tenure < 2 years Working Shift B with low Sick Leave Balance are 80% likely to be High Risk".*
3.  **Output (Dependent Variable)**:
    *   **Risk Category**: A label derived from the employee's historical total absent days (using Top 33% percentile as High Risk).

---

## 2. Scenario B: Daily Absence Count (Forecasting)

### **Objective**
To predict the exact number of employees who will be absent on a specific future date to help with roster planning.

### **The Model: LSTM (Long Short-Term Memory)**
We used an **LSTM** network, a type of **Recurrent Neural Network (RNN)** designed for time-series data (Deep Learning).
*   **How it works**: Unlike standard regression which just looks at "Inputs -> Output", LSTM has an internal "memory". It processes the data in valid time order (Day 1, Day 2, Day 3...). It learns to remember long-term trends (e.g., absenteeism is higher in December) and short-term dependencies (e.g., if 20 people were absent yesterday, today might also be high).
*   **Why this model**: It outperformed traditional models (XGBoost) because human behavior (absenteeism) is sequential and influenced by what happened in the immediate past.

### **Data & Logic**
1.  **Inputs (Independent Variables)**:
    *   **Temporal Features**: Month (captures seasonality), Day of Week (captures weekend effects), IsWeekend.
    *   **Lag Features**:
        *   `Lag_1`: How many people were absent yesterday?
        *   `Lag_7`: How many people were absent exactly one week ago?
    *   **Rolling Trends**: `RollingMean_7` (Average absences over the last week).
2.  **Logic**:
    *   The model looks at the "Context Window" (the past data) to predict the next step.
    *   *Interpretation*: "Given that it is a Monday in December, and last Monday had 15 absences, and yesterday had 12 absences, I predict today will have 14 absences."
3.  **Output (Dependent Variable)**:
    *   **AbsentCount**: A numerical value representing the total headcount of absent staff.

### **Comparison of Models Tested**
| Model | Type | MAE (Error) | Interpretation |
| :--- | :--- | :--- | :--- |
| **LSTM** | Deep Learning | **2.60** | Best performance. Captures complex time dependencies. |
| **Hybrid** | Ensemble | 2.78 | Average of LSTM and XGBoost. Good, but slightly worse than pure LSTM. |
| **Gradient Boosting**| Tree Ensemble | 2.82 | Good baseline, but struggles with the sequential nature of time series. |
| **XGBoost** | Tree Ensemble | 3.13 | Overfitted the data (Performed worse on new data). |
