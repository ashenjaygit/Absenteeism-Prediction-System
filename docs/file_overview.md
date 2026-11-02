# Project Structure & File Overview

This document explains every file in your project, helping you understand exactly "what runs what".

## 1. Where the "Brains" are created (Training Scripts)
These scripts are run *once* (or whenever you have new data) to learn from the data and save the "intelligence" into files.

*   **`scripts/process_data.py`**
    *   **Role**: The Chef 👨‍🍳
    *   **What it does**: Reads the raw Excel files (`Absenteeism 1.xlsx`), cleans them, calculates things like "Total Days Absent", and organizes them into clean CSV files inside the `processed/` folder.
    *   **Output**: `processed/employee_risk.csv` and `processed/daily_forecast.csv`.

*   **`scripts/train_models.py`** (Basic)
    *   **Role**: The Teacher (Classroom A) 👩‍🏫
    *   **What it does**: Reads the clean data and trains the **Random Forest** models.
    *   **Running this**: Creates the `classifier.pkl` (Risk Predictor) and `forecaster.pkl` (Simple Forecaster).

*   **`scripts/train_advanced_models.py`** (Advanced)
    *   **Role**: The Professor (Advanced Lab) 🔬
    *   **What it does**: Trains the complex models: **LSTM (Deep Learning)**, **XGBoost**, and **Gradient Boosting**. It compares them all and saves the best one.
    *   **Key Output**: `models/lstm_model.keras` (The Deep Learning Brain) and `models/model_metrics.json` (The report card of which model was best).

## 2. Where the "Brains" are stored (Model Files)
These are binary files. You generally cannot open them with a text editor. They contain the mathematical patterns the AI learned.

*   **`models/` Folder**:
    *   `classifier.pkl`: The "Risk Prediction" brain.
    *   `lstm_model.keras`: The "Daily Forecasting" brain (usually the most accurate).
    *   `xgb_model.pkl`: An alternative forecasting brain.
    *   `scaler_X.pkl` / `scaler_y.pkl`: Mathematical tools to translate raw numbers (like dates) into numbers the AI understands (0s and 1s).

## 3. The "Face" of the Project (The App)
This is what you interact with.

*   **`app.py`**
    *   **Role**: The Application Logic 🖥️
    *   **What it does**: 
        1.  It loads the "Brain" files from the `models/` folder.
        2.  It creates the website interface (buttons, sliders, graphs).
        3.  When you click "Predict", it sends your numbers to the loaded Brains and displays the answer.

*   **`run_app.bat`**
    *   **Role**: The Shortcut ⚡
    *   **What it does**: A simple Windows script that runs the command `streamlit run app.py` for you so you don't have to type it in the terminal every time.

*   **`requirements.txt`**
    *   **Role**: The Shopping List 🛒
    *   **What it does**: Tells Python which libraries it needs to download (like `pandas` for data, `streamlit` for the web app, `tensorflow` for the AI).
