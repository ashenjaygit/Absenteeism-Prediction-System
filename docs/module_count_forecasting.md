# Module 2: Daily Absence Count (Forecasting)

## 🎯 Goal
To predict the **exact total number** of employees who will be absent on any given future date (e.g., "How many people will be absent next Monday?"). This aids in roster planning and shift management.

## 🧠 The "Brain" (Model)
*   **Algorithm**: LSTM (Long Short-Term Memory) - A Deep Learning Neural Network.
*   **Type**: Time-Series Regression.
*   **Performance**: Mean Absolute Error (MAE) ~2.6 (Predictions are usually accurate within +/- 3 people).

## 📥 Input Data (What the model looks at)
The model analyzes time and history:
1.  **Calendar**: Month, Day of Week, Is it a Weekend?
2.  **Recent History (Lags)**:
    *   *Lag_1*: How many people were absent yesterday?
    *   *Lag_7*: How many people were absent exactly one week ago?
3.  **Trends**: The average absenteeism over the last 7 days.

## ⚙️ How it works
1.  The model reads the daily absence logs like a timeline.
2.  It learns **sequences** and **seasonality**.
    *   *Example 1*: "Absence always spikes on Mondays."
    *   *Example 2*: "If absence was high yesterday and usage is trending up, tomorrow will likely be high too."
3.  It uses "Recursive Forecasting" to predict days far into the future (up to 30 days).

## 📤 Output (The Result)
*   A specific **number** (e.g., "Predicted Absences: 14").
*   A trend line showing the expected rise or fall in absenteeism for the coming weeks.

## 📊 Statistical Performance Analysis
We compared 4 advanced models to find the most robust predictor.

### Metric Definitions
1.  **MAE (Mean Absolute Error)**: The average "miss" in number of people. *Lower is Better.*
2.  **R² (R-Squared)**: How much of the variance in the data the model explains. *Higher is Better (Max 1.0).*

### Model Comparison Table

| Model | **MAE** (Avg Error) | **R² Score** | Performance Verdict |
| :--- | :--- | :--- | :--- |
| **LSTM (Deep Learning)** | **2.59** | **0.163** | 🏆 **Winner**. Best accuracy and explanatory power. |
| Hybrid (XGB+LSTM) | 2.78 | 0.085 | Runner-up. Good stability but slightly less accurate. |
| Gradient Boosting | 2.82 | 0.030 | Weak. Barely better than guessing the average. |
| XGBoost | 3.13 | -0.165 | Failed. Overfitted and performed worse than a generic average. |

### Interpretation
*   **MAE of 2.59**: This means if the model predicts **10** people will be absent, the real number is usually between **7.4 and 12.6**. This is highly actionable for roster planning.
*   **Low R²**: Absenteeism is highly random (human sickness is unpredictable). An R² of 0.16 is actually respectable for daily human behavior data, distinguishing signal from noise.
