import pandas as pd
import numpy as np
import os

def process_data():
    print("Loading data...")
    # Load raw data
    try:
        df_emp = pd.read_excel("Absenteeism 1.xlsx")
        df_daily = pd.read_excel("daily_absence_summary.xlsx")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # --- Process Employee Data for Classification ---
    print("Processing Employee Data...")
    
    # 1. Aggregate absenteeism per employee
    # We want to predict risk based on history. 
    # For a real scenario we might split by time (train on past, predict future).
    # Here we will define 'Risk' based on their total absenteeism in the dataset.
    
    # Calculate total absent days per employee
    emp_agg = df_emp.groupby('EmployeeID').agg({
        'AbsenceDuration_Days': 'sum',
        'AbsentHours': 'sum',
        'Age': 'first',
        'Gender': 'first',
        'MaritalStatus': 'first',
        'NumberOfDependents': 'first',
        'Shift': 'first',
        'Tenure_Years': 'first',
        'AnnualLeaveBalance': 'last', # Take latest balance
        'SickLeaveBalance': 'last',
        'MaternityLeaveBalance': 'last',
        'PaternityLeaveBalance': 'last'
    }).reset_index()

    # 2. Define Risk Categories
    # Use quartiles to define Low, Moderate, High
    q1 = emp_agg['AbsenceDuration_Days'].quantile(0.33)
    q2 = emp_agg['AbsenceDuration_Days'].quantile(0.66)
    
    def classify_risk(days):
        if days <= q1:
            return 'Low'
        elif days <= q2:
            return 'Moderate'
        else:
            return 'High'

    emp_agg['RiskCategory'] = emp_agg['AbsenceDuration_Days'].apply(classify_risk)
    
    print(f"Risk Thresholds: Low <= {q1}, Moderate <= {q2}, High > {q2}")
    print(emp_agg['RiskCategory'].value_counts())

    # Save processed employee data
    emp_agg.to_csv("processed/employee_risk.csv", index=False)
    print("Saved processed/employee_risk.csv")

    # --- Process Daily Data for Forecasting ---
    print("Processing Daily Data...")
    
    # Ensure date format
    df_daily['AbsenceDate'] = pd.to_datetime(df_daily['AbsenceDate'])
    df_daily = df_daily.sort_values('AbsenceDate')
    
    # Feature Engineering
    df_daily['Year'] = df_daily['AbsenceDate'].dt.year
    df_daily['Month'] = df_daily['AbsenceDate'].dt.month
    df_daily['Day'] = df_daily['AbsenceDate'].dt.day
    df_daily['DayOfWeek'] = df_daily['AbsenceDate'].dt.dayofweek
    df_daily['IsWeekend'] = df_daily['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Lag Features (Previous day's absence)
    df_daily['Lag_1'] = df_daily['AbsentCount'].shift(1)
    df_daily['Lag_7'] = df_daily['AbsentCount'].shift(7)
    
    # Rolling Averages
    df_daily['RollingMean_7'] = df_daily['AbsentCount'].rolling(window=7).mean()
    
    # Drop NaNs created by lagging
    df_daily = df_daily.dropna()
    
    # Save processed daily data
    df_daily.to_csv("processed/daily_forecast.csv", index=False)
    print("Saved processed/daily_forecast.csv")

if __name__ == "__main__":
    if not os.path.exists("processed"):
        os.makedirs("processed")
    process_data()
