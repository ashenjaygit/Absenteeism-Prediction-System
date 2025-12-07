import pandas as pd
import os

files = ["Absenteeism 1.xlsx", "daily_absence_summary.xlsx"]

for f in files:
    print(f"--- Inspecting {f} ---")
    if os.path.exists(f):
        try:
            df = pd.read_excel(f)
            print("Columns:", df.columns.tolist())
            print("\nHead:")
            print(df.head())
            print("\nInfo:")
            print(df.info())
        except Exception as e:
            print(f"Error reading {f}: {e}")
    else:
        print(f"File {f} not found.")
    print("\n" + "="*30 + "\n")
