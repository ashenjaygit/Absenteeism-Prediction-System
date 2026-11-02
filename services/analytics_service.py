import pandas as pd
import numpy as np

class HRAnalyticsService:
    def __init__(self, raw_data_path="Absenteeism 1.xlsx", processed_data_path="processed/employee_risk.csv"):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self._load_data()

    def _load_data(self):
        try:
            self.df_raw = pd.read_excel(self.raw_data_path)
            if 'AbsenceDate' in self.df_raw.columns:
                self.df_raw['AbsenceDate'] = pd.to_datetime(self.df_raw['AbsenceDate'])
        except Exception as e:
            self.df_raw = pd.DataFrame()
            print(f"Warning: Could not load raw dataset: {e}")

        try:
            self.df_emp = pd.read_csv(self.processed_data_path)
        except Exception as e:
            self.df_emp = pd.DataFrame()
            print(f"Warning: Could not load employee_risk dataset: {e}")

    def get_kpis(self):
        """Returns high-level workforce absenteeism KPIs."""
        if self.df_emp.empty:
            return {}
        
        total_employees = len(self.df_emp)
        risk_counts = self.df_emp['RiskCategory'].value_counts().to_dict()
        high_risk_count = risk_counts.get('High', 0)
        mod_risk_count = risk_counts.get('Moderate', 0)
        low_risk_count = risk_counts.get('Low', 0)

        unauth_count = 0
        if not self.df_raw.empty and 'LeaveType' in self.df_raw.columns:
            unauth_count = int((self.df_raw['LeaveType'] == 'Unauthorized Leave').sum())

        avg_days = float(self.df_emp['AbsenceDuration_Days'].mean())
        avg_tenure = float(self.df_emp['Tenure_Years'].mean())

        return {
            "total_employees": total_employees,
            "high_risk_employees": high_risk_count,
            "moderate_risk_employees": mod_risk_count,
            "low_risk_employees": low_risk_count,
            "high_risk_percentage": round((high_risk_count / total_employees) * 100, 1) if total_employees else 0,
            "avg_absence_days": round(avg_days, 1),
            "avg_tenure_years": round(avg_tenure, 1),
            "total_unauthorized_incidents": unauth_count
        }

    def get_team_analysis(self):
        """Analyzes absenteeism, leave types, and risk across apparel teams/lines."""
        if self.df_raw.empty:
            return pd.DataFrame()

        # Aggregate raw data by TeamID
        team_grp = self.df_raw.groupby('TeamID').agg(
            total_records=('AbsenceDuration_Days', 'count'),
            total_absent_days=('AbsenceDuration_Days', 'sum'),
            total_hours=('AbsentHours', 'sum'),
            unauthorized_count=('LeaveType', lambda x: (x == 'Unauthorized Leave').sum()),
            sick_count=('LeaveType', lambda x: (x == 'Sick Leave').sum()),
            unique_employees=('EmployeeID', 'nunique')
        ).reset_index()

        team_grp['avg_days_per_emp'] = (team_grp['total_absent_days'] / team_grp['unique_employees']).round(2)
        team_grp['unauthorized_rate_%'] = ((team_grp['unauthorized_count'] / team_grp['total_records']) * 100).round(1)

        return team_grp.sort_values(by='total_absent_days', ascending=False)

    def get_leave_type_distribution(self):
        """Returns distribution and breakdown of leave types."""
        if self.df_raw.empty or 'LeaveType' not in self.df_raw.columns:
            return pd.DataFrame()

        dist = self.df_raw['LeaveType'].value_counts().reset_index()
        dist.columns = ['LeaveType', 'Count']
        dist['Percentage'] = ((dist['Count'] / dist['Count'].sum()) * 100).round(1)
        return dist

    def get_shift_analysis(self):
        """Compares Day vs Night shift absenteeism characteristics."""
        if self.df_raw.empty:
            return pd.DataFrame()

        shift_grp = self.df_raw.groupby('Shift').agg(
            total_absences=('AbsenceDuration_Days', 'count'),
            total_days=('AbsenceDuration_Days', 'sum'),
            avg_duration=('AbsenceDuration_Days', 'mean'),
            unauthorized=('LeaveType', lambda x: (x == 'Unauthorized Leave').sum()),
            unique_workers=('EmployeeID', 'nunique')
        ).reset_index()
        return shift_grp

    def get_employee_history(self, emp_id):
        """Retrieves individual absence incident history for a given employee."""
        if self.df_raw.empty:
            return pd.DataFrame()
        return self.df_raw[self.df_raw['EmployeeID'] == emp_id].sort_values(by='AbsenceDate', ascending=False)
