# Power BI Dashboard Setup Guide

This guide explains how to recreate the Absenteeism Prediction Dashboard in Microsoft Power BI.

## 1. Prepare Data
The system already produces the necessary files in the `processed/` folder:
-   `processed/employee_risk.csv`: Contains Employee IDs, Demographics, and **Risk Category**.
-   `processed/daily_forecast.csv`: Contains Dates, Daily Absent Counts, and Time Features.

## 2. Import Data into Power BI
1.  Open **Power BI Desktop**.
2.  Click **Get Data** -> **Text/CSV**.
3.  Select `processed/employee_risk.csv` and click **Load**.
4.  Repeat to load `processed/daily_forecast.csv`.

## 3. Create Visualizations

### Page 1: Overview & Risk Analysis
*   **Card 1**: Count of `EmployeeID` (Label: "Total Employees").
*   **Card 2**: Count of `EmployeeID` filtered by `RiskCategory="High"` (Label: "High Risk Employees").
*   **Card 3**: Average of `Tenure_Years` (Label: "Avg Tenure").
*   **Donut Chart**:
    *   **Legend**: `RiskCategory`
    *   **Values**: Count of `EmployeeID`
    *   *Purpose*: Shows the % of workforce in Low/Mod/High risk.
*   **Bar Chart**:
    *   **Axis**: `TeamID` or `Department`
    *   **Values**: Count of `RiskCategory="High"`
    *   *Purpose*: Identify problem teams.

### Page 2: Absenteeism Trends (Forecasting)
*   **Line Chart**:
    *   **X-Axis**: `AbsenceDate`
    *   **Y-Axis**: `AbsentCount`
    *   *Add Analytics*: Right-click the chart -> **Add Forecast**.
        *   Forecast Length: 30 Days.
        *   Confidence Interval: 95%.
    *   *Note*: Power BI has built-in forecasting (exponential smoothing) for line charts with time axes.

## 4. Advanced (Optional): Running Python Scripts in Power BI
If you want to use our **Machine Learning predictions** directly inside Power BI:
1.  In Power BI, click **Get Data** -> **Python script**.
2.  Paste the code from `scripts/process_data.py`.
3.  This will run the Python logic and import the resulting dataframes directly.
4.  *Requirement*: You must have Python installed and configured in Power BI Options.

## 5. Refreshing Data
*   Whenever you get new Excel raw data, replace the files in the project folder.
*   Click **Refresh** in Power BI to auto-update all charts.
