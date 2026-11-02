# Model Technical Specifications

This document provides a detailed breakdown of the **Dependent Variables (Target)** and **Independent Variables (Features)** for every model used in the project.

---

## 1. Classification Model (Employee Risk)
**Algorithm**: `RandomForestClassifier`
**Goal**: Predict the risk level of an individual employee.

### A. Dependent Variable (Target)
*   **`RiskCategory`**: The output class.
    *   Values: `Low`, `Moderate`, `High`
    *   *Derived from*: The 33rd and 66th percentiles of total absence days.

### B. Independent Variables (Features)
These are the inputs the model uses to make the prediction:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Age` | Numerical | Employee's age. |
| `Gender_Code` | Categorical | Encoded Gender (e.g., 0=Male, 1=Female). |
| `Marital_Code` | Categorical | Encoded Marital Status (Single/Married). |
| `Shift_Code` | Categorical | The work shift pattern. |
| `NumberOfDependents`| Numerical | How many children/dependents they have. |
| `Tenure_Years` | Numerical | How long they have worked at the company. |
| `AnnualLeaveBalance`| Numerical | Remaining annual leave days. |
| `SickLeaveBalance` | Numerical | Remaining sick leave days. |
| `Maternity` / `Paternity` Balances | Numerical | Specific leave balances. |

---

## 2. Advanced Forecasting Models (Daily Count)
**Algorithms**: `LSTM`, `XGBoost`, `GradientBoosting`, `Hybrid`
**Goal**: Predict the total number of absentees for a specific date.

### A. Dependent Variable (Target)
*   **`AbsentCount`**: The total count of employees absent on that day.
    *   Type: Continuous (Numerical)

### B. Independent Variables (Features)
We optimized these models by adding "Long-Term Memory" features.

| Feature Group | Feature Name | Description |
| :--- | :--- | :--- |
| **Calendar** | `Year`, `Month`, `Day` | Captures yearly growth and seasonal monthly trends. |
| **Weekly Cycles** | `DayOfWeek` | (0-6) Captures "Monday Blues" or "Friday" effects. |
| **Weekend Flag** | `IsWeekend` | (0 or 1) Explicitly marks non-working/low-activity days. |
| **Short-Term Memory** | `Lag_1` | **"Yesterday's Absences"**. The strongest predictor. |
| | `Lag_7` | **"Last Week same day"**. Captures weekly cycles. |
| **Long-Term Memory** | `Lag_14` | Absences 2 weeks ago. |
| | `Lag_30` | Absences 1 month ago. |
| **Trends** | `RollingMean_7` | Avg absences over the last week (Is it trending up?). |
| | `RollingMean_30` | Avg absences over the last month (Long term trend). |

---

## 3. Basic Regression Model (Baseline)
**Algorithm**: `RandomForestRegressor`
**Goal**: A simple baseline to compare against the advanced models.

### A. Dependent Variable (Target)
*   **`AbsentCount`**

### B. Independent Variables (Features)
*The basic model uses a simpler feature set than the advanced ones.*
*   `Year`, `Month`, `Day`
*   `DayOfWeek`, `IsWeekend`
*   `Lag_1` (Yesterday)
*   `Lag_7` (Last Week)
*   `RollingMean_7` (7-Day Trend)
*   *(Note: It does NOT use the 14/30 day long-term memory features)*
