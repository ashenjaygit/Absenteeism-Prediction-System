# Absenteeism Prediction System (Apparel Industry)

A machine learning-powered system designed to forecast absenteeism trends and classify employee risk in the apparel industry. This tool empowers HR and management with actionable insights for proactive workforce planning.

## 🚀 Features

*   **Employee Risk Classification**: Categorizes employees into **High**, **Moderate**, and **Low** risk based on historical absence patterns and demographics using a Random Forest Classifier.
*   **Smart Forecasting**: Predicts daily absence counts for the next 7-30 days using advanced times-series models (**LSTM**, **XGBoost**, **Gradient Boosting**, and **Hybrid** models).
*   **Interactive Dashboard**: A user-friendly web interface built with **Streamlit** to visualize risk distribution and forecasting trends.
*   **Power BI Support**: Includes a guide to visualize the processed data in Microsoft Power BI.

## 🛠️ Tech Stack

*   **Python 3.13**
*   **Machine Learning**: `scikit-learn`, `xgboost`, `tensorflow` (Keras)
*   **Data Processing**: `pandas`, `numpy`, `openpyxl`
*   **Visualization**: `matplotlib`, `seaborn`, `streamlit`

## 📂 Project Structure

```
├── processed/              # Processed CSV data files
├── models/                 # Saved ML models (.pkl, .keras)
├── scripts/
│   ├── process_data.py     # Data cleaning and feature engineering
│   ├── train_models.py     # Basic model training (RF)
│   ├── train_advanced_models.py # Advanced training (LSTM, XGBoost)
├── docs/
│   └── powerbi_setup.md    # Guide for Power BI Dashboard
├── app.py                  # Streamlit Dashboard application
├── requirements.txt        # Python dependencies
└── run_app.bat             # Shortcut to run the dashboard
```

## ⚙️ Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/ashenjaygit/Absenteeism-Prediction-System.git
    cd Absenteeism-Prediction-System
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install xgboost tensorflow
    ```

##  ▶️ Usage

### 1. Run the Dashboard
You can easily start the application by double-clicking `run_app.bat` or running:

```bash
streamlit run app.py
```

### 2. Retrain Models (Optional)
If you have new data in `Absenteeism 1.xlsx` or `daily_absence_summary.xlsx`:

```bash
# 1. Process new data
python scripts/process_data.py

# 2. Train models
python scripts/train_advanced_models.py
```

## 📊 Model Performance

The system evaluates multiple forecasting models and automatically selects the best one. Current performance metrics (MAE):

*   **LSTM (Deep Learning)**: ~2.60 (Best)
*   **Hybrid (XGB+LSTM)**: ~2.78
*   **Gradient Boosting**: ~2.82
*   **XGBoost**: ~3.13

## 📈 Power BI Integration

If you prefer Power BI for reporting, check the setup guide in:
`docs/powerbi_setup.md`
