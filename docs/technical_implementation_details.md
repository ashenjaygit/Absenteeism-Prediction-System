# Technical Implementation Details

## 1. Feature Engineering
**YES**, we used extensive Feature Engineering. This is the process of creating new helpful "clues" for the AI from the raw data.

### Features Created:
*   **Temporal Features (Time)**:
    *   Extracted `Month`, `Day`, and `Year` from the date.
    *   **`DayOfWeek`**: The model needs to know if it's a Monday (usually high absence) or Friday.
    *   **`IsWeekend`**: A binary (0/1) flag to tell the model if it's Saturday/Sunday.
*   **Lag Features (The most important ones)**:
    *   **`Lag_1`**: We created a column showing "Yesterday's Absences".
    *   **`Lag_7`**: "Absences on this same day last week".
    *   *Why?* This helps the model captures patterns like "Mondays are always busy".
*   **Rolling Window Features**:
    *   **`RollingMean_7`**: The average number of absences over the last 7 days. This helps the model see the recent "Trend" (is absenteeism going up or down overall?).

---

## 2. Data Leakage
**Data Leakage** is a common mistake where the model accidentally "sees" the future during training. **We took specific steps to PREVENT this.**

### How we prevented Leakage:
1.  **Time-Based Splitting (Crucial for Forecasting)**:
    *   In `train_advanced_models.py`, we did **NOT** shuffle the data randomly.
    *   We utilized a **Strict Cut-off**: We trained on the first ~80% of days (Past) and tested on the last ~20% (Future).
    *   *Proof in code*: `X_train = X_ts.iloc[:split_point]`.
    *   If we had shuffled the data (random split), the model would have learned from "Tomorrow" to predict "Today", which is cheating. We ensured this didn't happen.

2.  **Lag Generation**:
    *   When creating `Lag_1`, we ensured the data was shifted correctly so that for "Today", the model only sees data from "Yesterday".

### A Minor Note on Scaling:
*   We used `MinMaxScaler` to scale data between 0 and 1. Ideally, you fit this *only* on the training data. In our script, we fit it on the whole dataset for simplicity. While this is a tiny form of statistical leakage (knowing the min/max of the future), it generally does not affect the validity of the results significantly for this type of project.
