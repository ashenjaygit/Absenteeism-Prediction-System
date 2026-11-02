# AI-Powered Employee Absenteeism Prediction & Intelligent HR Decision Support System

An enterprise-grade Decision Support Platform combining **Predictive Machine Learning**, **Explainable AI (SHAP)**, **Domain RAG (ChromaDB)**, and **Generative AI (Groq)** to optimize workforce planning and empower HR leadership in apparel manufacturing.

---

## 🎯 Architectural Layers

### 1. Predictive Intelligence Layer
*   **Employee Risk Classification**: Gradient Boosting Classifier (94% Accuracy) classifying employees into **Low**, **Moderate**, and **High** risk tiers based on 13 features (leave utilization, seniority, demographics, balances).
*   **Daily Absenteeism Forecasting**: Time-series forecasting using **LSTM (Deep Learning)**, **XGBoost**, and **Hybrid Models** over 7 to 30-day horizons to prevent apparel assembly line bottlenecks.

### 2. Explainable AI (XAI) Layer
*   **SHAP (Shapley Additive exPlanations)**: Calculates local feature attributions using `shap.KernelExplainer`, visualizing the exact directional impact (positive vs. negative) of every attribute driving an employee's risk score.

### 3. Generative AI & Decision Support Layer
*   **AI HR Explainer**: Powered by **Groq High-Speed Inference**, converting mathematical SHAP weights into empathetic, non-punitive HR recommendations (e.g., fatigue management, shift flexibility).
*   **Autonomous Executive Report Generator**: Synthesizes cross-team absenteeism patterns, leave distributions, and forecast trends into downloadable management briefs.

### 4. Domain Policy RAG Layer
*   **Vector Database**: ChromaDB vector store indexing Sri Lankan labor law (Shop & Office Employees Act / Factories Ordinance) and factory attendance SOPs.
*   **HR Intelligence Copilot**: Conversational interface answering complex operational and policy queries with grounded citations.

---

## 🛠️ Tech Stack
*   **Core Language**: Python 3.13
*   **Predictive ML & Deep Learning**: `scikit-learn`, `xgboost`, `tensorflow` (Keras)
*   **Explainable AI**: `shap`
*   **GenAI & LLM Inference**: `groq`, `python-dotenv`
*   **Vector Search & RAG**: `chromadb`
*   **Data & Visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `streamlit`

---

## 📂 Project Structure
```
├── processed/              # Cleaned datasets for ML (Risk & Forecast CSVs)
├── models/                 # Trained models (.pkl, .keras, scalers)
├── services/               # Core AI & Decision Support Services
│   ├── explainer_service.py # SHAP attribution & visual bar plot generator
│   ├── analytics_service.py # Workforce & 10 apparel line analytics
│   ├── rag_service.py      # ChromaDB policy indexing & semantic retrieval
│   ├── llm_service.py      # Groq client for explanations, chat & reports
│   └── agent_service.py    # Agentic tool execution & context coordinator
├── docs/                   # Documentation & HR Policies
│   ├── hr_policies/        # Apparel attendance SOPs & labor law guidelines
│   └── interview_prep_eArrow.md # Interview pitch & live demonstration guide
├── scripts/                # Data pipelines & model training scripts
├── app.py                  # Streamlit Multi-Tab HR Intelligence Dashboard
└── requirements.txt        # Project dependencies
```

## ⚙️ How to Run
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/ashenjaygit/Absenteeism-Prediction-System.git
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install xgboost tensorflow
    ```
3.  **Launch Dashboard**:
    ```bash
    streamlit run app.py
    ```
    *Windows users can simply double-click `run_app.bat`.*

## 📊 Performance (Optimized)
### 1. Risk Classification (Accuracy: ~94%)
*   **High Risk Precision**: **0.96** (We correctly identify 96% of high-risk employees).
*   **Improvement**: 
    1.  Upgraded to **Gradient Boosting Classifier**.
    2.  Added smart features: **Utilization Rate** and **Seniority**.
*   **Journey**: 47% (Basic) -> 55% (Tuned RF) -> **94%** (Gradient Boosting + Advanced Features).

### 2. Daily Forecasting (Aggressively Optimized)
*   **Best Model**: **LSTM (Deep Learning)**.
*   **Performance (Extreme Optimization)**:
    *   **LSTM**: **0.37 MAE** | **0.98 R²** (Winner 🏆)
    *   **Hybrid**: 1.16 MAE | 0.82 R²
    *   **XGBoost**: 2.37 MAE
*   **Verdict**: By feeding the model the **full weekly history** (Lags 1-7), we achieved high accuracy on the test set.

