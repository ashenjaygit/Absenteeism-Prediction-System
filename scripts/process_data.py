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

    # 1.5 Feature Engineering for Classification
    # Total Leave Balance available
    emp_agg['TotalLeaveBalance'] = (emp_agg['AnnualLeaveBalance'] + 
                                    emp_agg['SickLeaveBalance'] + 
                                    emp_agg['MaternityLeaveBalance'] + 
                                    emp_agg['PaternityLeaveBalance'])
    
    # Utilization Potential: (Total Absent Days) / (Tenure in Days + 1)
    # This normalizes absenteeism by how long they've been here.
    emp_agg['UtilizationRate'] = emp_agg['AbsenceDuration_Days'] / (emp_agg['Tenure_Years'] * 365 + 1)
    
    # Interaction: Age * Tenure (Seniority Factor)
    emp_agg['Seniority'] = emp_agg['Age'] * emp_agg['Tenure_Years']

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
    df_daily['Lag_2'] = df_daily['AbsentCount'].shift(2)
    df_daily['Lag_3'] = df_daily['AbsentCount'].shift(3)
    df_daily['Lag_4'] = df_daily['AbsentCount'].shift(4)
    df_daily['Lag_5'] = df_daily['AbsentCount'].shift(5)
    df_daily['Lag_6'] = df_daily['AbsentCount'].shift(6)
    df_daily['Lag_7'] = df_daily['AbsentCount'].shift(7)
    df_daily['Lag_14'] = df_daily['AbsentCount'].shift(14)
    df_daily['Lag_30'] = df_daily['AbsentCount'].shift(30)
    
    
    # 3. Advanced Feature Engineering (Cyclical Features)
    # Month (1-12)
    df_daily['Month_Sin'] = np.sin(2 * np.pi * df_daily['Month']/12)
    df_daily['Month_Cos'] = np.cos(2 * np.pi * df_daily['Month']/12)
    
    # DayOfWeek (0-6)
    df_daily['DayOfWeek_Sin'] = np.sin(2 * np.pi * df_daily['DayOfWeek']/7)
    df_daily['DayOfWeek_Cos'] = np.cos(2 * np.pi * df_daily['DayOfWeek']/7)

    # Rolling Statistics (Volatility)
    df_daily['RollingMean_7'] = df_daily['AbsentCount'].rolling(window=7).mean()
    df_daily['RollingMean_30'] = df_daily['AbsentCount'].rolling(window=30).mean()
    df_daily['RollingStd_7'] = df_daily['AbsentCount'].rolling(window=7).std()
    df_daily['RollingStd_30'] = df_daily['AbsentCount'].rolling(window=30).std()
    
    # Differencing (Velocity of change)
    # Lag_1 - Lag_2 (We can approximate this using shifts)
    df_daily['Diff_1'] = df_daily['AbsentCount'].shift(1) - df_daily['AbsentCount'].shift(2)
    
    # Drop NaNs created by lagging
    df_daily = df_daily.dropna()
    
    # Save processed daily data
    df_daily.to_csv("processed/daily_forecast.csv", index=False)
    print("Saved processed/daily_forecast.csv")

if __name__ == "__main__":
    if not os.path.exists("processed"):
        os.makedirs("processed")
    process_data()
