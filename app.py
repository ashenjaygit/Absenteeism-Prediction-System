import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Load local services
from services.explainer_service import AbsenteeismExplainer
from services.analytics_service import HRAnalyticsService
from services.rag_service import HRPoplicyRAGService
from services.llm_service import LLMService
from services.agent_service import HRAgent

load_dotenv()

st.set_page_config(
    page_title="MAS HR Absenteeism Intelligence System",
    page_icon="👕",
    layout="wide"
)

# --- CACHED MODEL & SERVICE LOADERS ---
@st.cache_resource
def load_all_models():
    try:
        clf = joblib.load("models/classifier.pkl")
        le_gender = joblib.load("models/le_gender.pkl")
        le_marital = joblib.load("models/le_marital.pkl")
        le_shift = joblib.load("models/le_shift.pkl")
        
        # Forecasting Models
        try:
            scaler_X = joblib.load("models/scaler_X.pkl")
            scaler_y = joblib.load("models/scaler_y.pkl")
            gb_model = joblib.load("models/gb_model.pkl")
            xgb_model = joblib.load("models/xgb_model.pkl")
            lstm_model = load_model("models/lstm_model.keras")
            
            with open("models/model_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception as e:
            st.error(f"Error loading forecasting models: {e}")
            return clf, le_gender, le_marital, le_shift, None, None, None, None, None, None

        return clf, le_gender, le_marital, le_shift, scaler_X, scaler_y, gb_model, xgb_model, lstm_model, metrics
    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        return None, None, None, None, None, None, None, None, None, None

@st.cache_data
def load_data():
    try:
        df_emp = pd.read_csv("processed/employee_risk.csv")
        df_daily = pd.read_csv("processed/daily_forecast.csv")
        df_daily['AbsenceDate'] = pd.to_datetime(df_daily['AbsenceDate'])
        return df_emp, df_daily
    except:
        return pd.DataFrame(), pd.DataFrame()

@st.cache_resource
def get_explainer(_clf, _df_emp, _le_gender, _le_marital, _le_shift):
    if _clf is None or _df_emp.empty:
        return None
    features = [
        'Age', 'Gender_Code', 'Marital_Code', 'NumberOfDependents', 
        'Shift_Code', 'Tenure_Years', 
        'AnnualLeaveBalance', 'SickLeaveBalance', 'MaternityLeaveBalance', 'PaternityLeaveBalance',
        'TotalLeaveBalance', 'UtilizationRate', 'Seniority'
    ]
    df_bg = _df_emp.copy()
    df_bg['Gender_Code'] = _le_gender.transform(df_bg['Gender'])
    df_bg['Marital_Code'] = _le_marital.transform(df_bg['MaritalStatus'])
    df_bg['Shift_Code'] = _le_shift.transform(df_bg['Shift'])
    return AbsenteeismExplainer(_clf, df_bg, features)

@st.cache_resource
def get_ai_services():
    analytics = HRAnalyticsService()
    rag = HRPoplicyRAGService()
    try:
        llm = LLMService()
        agent = HRAgent(llm_service=llm, analytics_service=analytics, rag_service=rag)
        llm_ready = True
    except Exception as e:
        llm = None
        agent = None
        llm_ready = False
    return analytics, rag, llm, agent, llm_ready

# Load core components
clf, le_gender, le_marital, le_shift, scaler_X, scaler_y, gb_model, xgb_model, lstm_model, metrics = load_all_models()
df_emp, df_daily = load_data()
explainer = get_explainer(clf, df_emp, le_gender, le_marital, le_shift)
analytics_service, rag_service, llm_service, hr_agent, llm_ready = get_ai_services()

# --- SIDEBAR: SYSTEM TELEMETRY ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("System Telemetry")
    st.caption("AI-Powered HR Decision Support Layer")
    st.divider()

    st.subheader("Model Status")
    st.markdown("🟢 **Classifier:** Gradient Boosting (94% Acc)")
    st.markdown("🟢 **Forecaster:** LSTM + XGBoost Hybrid")
    st.markdown(f"{'🟢' if explainer else '🔴'} **Explainable AI:** SHAP KernelExplainer")
    st.markdown(f"{'🟢' if llm_ready else '🟡'} **GenAI Core:** Groq Llama 3 / Qwen")
    st.markdown(f"{'🟢' if rag_service else '🔴'} **Policy Store:** ChromaDB ({rag_service.collection.count() if rag_service and rag_service.collection else 0} SOPs)")
    
    st.divider()
    st.info("💡 **Interview Tip:** Emphasize how traditional ML handles prediction while SHAP + GenAI translates mathematical drivers into human-centric, ethical HR actions.")

# --- MAIN DASHBOARD HEADER ---
st.title("👕 AI-Powered HR Absenteeism Intelligence System")
st.markdown("*A Unified Platform Combining Predictive Machine Learning, SHAP Explainability, Policy RAG, and Autonomous Decision Support for the Apparel Industry.*")
st.divider()

if clf is None:
    st.error("ML Models not found! Please check the models/ directory.")
else:
    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Employee Risk & SHAP Explainer",
        "📅 Smart Workforce Forecasting",
        "🤖 HR Intelligence Copilot (RAG)",
        "📑 Executive Management Report"
    ])

    # =========================================================================
    # TAB 1: EMPLOYEE RISK PREDICTION & SHAP EXPLAINER
    # =========================================================================
    with tab1:
        st.header("Employee Absenteeism Risk & Explainable AI")
        st.write("Predict individual absence risk and generate mathematically grounded SHAP explanations with AI recommendations.")

        col_search, col_spacer = st.columns([2, 1])
        with col_search:
            emp_id_list = df_emp['EmployeeID'].unique() if not df_emp.empty else []
            selected_emp_id = st.selectbox("Search & Auto-fill by Employee ID", emp_id_list)

        # Default initialization
        default_age = 30
        default_gender = le_gender.classes_[0]
        default_marital = le_marital.classes_[0]
        default_dependents = 1
        default_shift = le_shift.classes_[0]
        default_tenure = 5
        default_annual = 10.0
        default_sick = 5.0
        default_mat = 0
        default_pat = 0
        default_past_absent = 0.0

        if not df_emp.empty and selected_emp_id:
            try:
                emp_data = df_emp[df_emp['EmployeeID'] == selected_emp_id].iloc[0]
                default_age = int(emp_data['Age'])
                default_gender = emp_data['Gender']
                default_marital = emp_data['MaritalStatus']
                default_dependents = int(emp_data['NumberOfDependents'])
                default_shift = emp_data['Shift']
                default_tenure = int(emp_data['Tenure_Years'])
                default_annual = float(emp_data['AnnualLeaveBalance'])
                default_sick = float(emp_data['SickLeaveBalance'])
                default_mat = int(emp_data['MaternityLeaveBalance'])
                default_pat = int(emp_data['PaternityLeaveBalance'])
                default_past_absent = float(emp_data['AbsenceDuration_Days'])
            except Exception as e:
                st.warning(f"Using default employee parameters: {e}")

        with st.expander("⚙️ Employee Profile & Feature Adjustments", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age", 18, 65, default_age)
                idx_gender = list(le_gender.classes_).index(default_gender) if default_gender in le_gender.classes_ else 0
                idx_marital = list(le_marital.classes_).index(default_marital) if default_marital in le_marital.classes_ else 0
                idx_shift = list(le_shift.classes_).index(default_shift) if default_shift in le_shift.classes_ else 0

                gender = st.selectbox("Gender", le_gender.classes_, index=idx_gender)
                marital = st.selectbox("Marital Status", le_marital.classes_, index=idx_marital)
                dependents = st.number_input("Number of Dependents", 0, 10, default_dependents)
                shift = st.selectbox("Shift", le_shift.classes_, index=idx_shift)
            
            with col2:
                tenure = st.number_input("Tenure (Years)", 0, 40, default_tenure)
                annual_bal = st.number_input("Annual Leave Balance", 0.0, 30.0, default_annual)
                sick_bal = st.number_input("Sick Leave Balance", 0.0, 30.0, default_sick)
                maternity_bal = st.number_input("Maternity Leave Balance", 0, 100, default_mat)
                paternity_bal = st.number_input("Paternity Leave Balance", 0, 100, default_pat)
                past_absent = st.number_input("Historical Absent Days", 0.0, 365.0, default_past_absent)

        # Build feature vector
        tot_leave = annual_bal + sick_bal + maternity_bal + paternity_bal
        util_rate = past_absent / (tenure * 365 + 1)
        seniority = age * tenure

        input_data = pd.DataFrame({
            'Age': [age],
            'Gender_Code': [le_gender.transform([gender])[0]],
            'Marital_Code': [le_marital.transform([marital])[0]],
            'NumberOfDependents': [dependents],
            'Shift_Code': [le_shift.transform([shift])[0]],
            'Tenure_Years': [tenure],
            'AnnualLeaveBalance': [annual_bal],
            'SickLeaveBalance': [sick_bal],
            'MaternityLeaveBalance': [maternity_bal],
            'PaternityLeaveBalance': [paternity_bal],
            'TotalLeaveBalance': [tot_leave],
            'UtilizationRate': [util_rate], 
            'Seniority': [seniority]
        })

        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            run_pred = st.button("🚀 Predict & Explain Risk", type="primary")

        if run_pred or "last_explanation" in st.session_state:
            # Predict
            prediction = clf.predict(input_data)[0]
            probs = clf.predict_proba(input_data)[0]
            confidence = max(probs)

            # Store in session state
            st.session_state["pred_risk"] = prediction
            st.session_state["pred_conf"] = confidence

            # Display prediction banner
            st.subheader(f"Risk Category: **{prediction}** (Confidence: {confidence:.1%})")
            if prediction == 'High':
                st.error("⚠️ HIGH RISK: Elevated probability of unscheduled absence. Early intervention recommended.")
            elif prediction == 'Moderate':
                st.warning("⚠️ MODERATE RISK: Baseline trends fluctuate. Proactive monitoring advised.")
            else:
                st.success("✅ LOW RISK: High attendance consistency and regular engagement.")

            st.divider()

            # --- EXPLAINABLE AI & GENAI INSIGHTS ---
            st.subheader("💡 Explainable AI: SHAP Attribution & AI HR Advisory")
            
            with st.spinner("Calculating SHAP values and synthesizing AI recommendations..."):
                if explainer:
                    shap_res = explainer.explain_instance(input_data)
                    st.session_state["shap_res"] = shap_res

                    col_plot, col_text = st.columns([1, 1])

                    with col_plot:
                        st.markdown("**Mathematical SHAP Feature Attribution**")
                        st.caption("Red bars increase predicted risk; green bars decrease predicted risk.")
                        fig_shap = explainer.generate_shap_plot(shap_res)
                        st.pyplot(fig_shap)

                        # Data table of drivers
                        driver_df = pd.DataFrame(shap_res["top_drivers"])[["display_name", "feature_value", "shap_value", "impact"]]
                        driver_df.columns = ["Factor", "Value", "SHAP Weight", "Direction"]
                        st.dataframe(driver_df, hide_index=True)

                    with col_text:
                        st.markdown("**Generative AI HR Decision Support**")
                        if llm_ready and llm_service:
                            try:
                                emp_profile_dict = {
                                    'Age': age, 'Gender': gender, 'MaritalStatus': marital,
                                    'Shift': shift, 'Tenure_Years': tenure,
                                    'AnnualLeaveBalance': annual_bal, 'SickLeaveBalance': sick_bal,
                                    'AbsenceDuration_Days': past_absent
                                }
                                ai_text = llm_service.explain_prediction(
                                    emp_id=selected_emp_id,
                                    emp_info=emp_profile_dict,
                                    risk_category=prediction,
                                    confidence=confidence,
                                    top_drivers=shap_res["top_drivers"]
                                )
                                st.markdown(ai_text)
                            except Exception as e:
                                st.error(f"GenAI generation notice: {e}")
                        else:
                            st.info("Groq API Key not detected. Add `GROQ_API_KEY` to `.env` to activate full natural language explanations.")
                            # Fallback display
                            st.markdown(f"""
                            **Automated Assessment:**
                            - Top Risk Driver: **{shap_res['top_drivers'][0]['display_name']}** (Impact: {shap_res['top_drivers'][0]['shap_value']:+.3f})
                            - Secondary Driver: **{shap_res['top_drivers'][1]['display_name']}** (Impact: {shap_res['top_drivers'][1]['shap_value']:+.3f})
                            
                            **Recommended Proactive Actions:**
                            1. Conduct a supportive check-in regarding recent schedule demands.
                            2. Verify leave balance reconciliation and health support availability.
                            3. Plan multi-skilled buffer support for this operator's cluster.
                            """)

    # =========================================================================
    # TAB 2: SMART WORKFORCE FORECASTING (PRESERVED & ENHANCED)
    # =========================================================================
    with tab2:
        st.header("Advanced Daily Absence Forecasting")
        st.write("Evaluate multi-model performance and forecast future workforce absences across the manufacturing facility.")

        if metrics:
            st.subheader("Model Performance Benchmark (MAE - Lower is Better)")
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Model', 'MAE']).sort_values('MAE')
            best_model_name = metrics_df.iloc[0]['Model']

            col_m1, col_m2 = st.columns([1, 2])
            with col_m1:
                st.table(metrics_df)
                st.success(f"🏆 Top Performing Model: **{best_model_name}**")
            with col_m2:
                fig_perf, ax_perf = plt.subplots(figsize=(7, 3))
                sns.barplot(data=metrics_df, x='MAE', y='Model', palette='magma', ax=ax_perf)
                ax_perf.set_title("Forecast Error Comparison (Mean Absolute Error)")
                st.pyplot(fig_perf)

        st.divider()
        st.subheader("Future Absence Forecast Simulation")

        if not df_daily.empty and scaler_X:
            days_to_predict = st.slider("Forecast Horizon (Days)", 7, 30, 14)
            last_date = df_daily['AbsenceDate'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)

            future_preds = []
            temp_history = df_daily['AbsentCount'].tolist()

            for date in future_dates:
                day_of_week = date.dayofweek
                is_weekend = 1 if day_of_week >= 5 else 0
                month = date.month
                day = date.day
                year = date.year

                lag_1 = temp_history[-1]
                lag_2 = temp_history[-2]
                lag_3 = temp_history[-3]
                lag_4 = temp_history[-4]
                lag_5 = temp_history[-5]
                lag_6 = temp_history[-6]
                lag_7 = temp_history[-7]
                lag_14 = temp_history[-14]
                lag_30 = temp_history[-30]
                roll_7 = np.mean(temp_history[-7:])
                roll_30 = np.mean(temp_history[-30:])

                month_sin = np.sin(2 * np.pi * month / 12)
                month_cos = np.cos(2 * np.pi * month / 12)
                day_sin = np.sin(2 * np.pi * day_of_week / 7)
                day_cos = np.cos(2 * np.pi * day_of_week / 7)

                roll_std_7 = np.std(temp_history[-7:])
                roll_std_30 = np.std(temp_history[-30:])
                diff_1 = temp_history[-1] - temp_history[-2]

                ft_vector = np.array([[year, month, day, day_of_week, is_weekend,
                                       lag_1, lag_2, lag_3, lag_4, lag_5, lag_6, lag_7,
                                       lag_14, lag_30, roll_7, roll_30,
                                       month_sin, month_cos, day_sin, day_cos,
                                       roll_std_7, roll_std_30, diff_1]])
                ft_scaled = scaler_X.transform(ft_vector)

                if best_model_name == 'Hybrid':
                    xgb_p = xgb_model.predict(ft_scaled)
                    lstm_p = lstm_model.predict(ft_scaled.reshape(1, 1, 23), verbose=0)
                    pred_scaled = (xgb_p + lstm_p) / 2
                elif best_model_name == 'XGBoost':
                    pred_scaled = xgb_model.predict(ft_scaled)
                elif best_model_name == 'GradientBoosting':
                    pred_scaled = gb_model.predict(ft_scaled)
                elif best_model_name == 'LSTM':
                    pred_scaled = lstm_model.predict(ft_scaled.reshape(1, 1, 23), verbose=0)
                else:
                    pred_scaled = xgb_model.predict(ft_scaled)

                pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                temp_history.append(pred)
                future_preds.append({'Date': date.strftime('%Y-%m-%d'), 'PredictedAbsences': max(0, int(round(pred)))})

            df_future = pd.DataFrame(future_preds)
            st.session_state["df_future"] = df_future

            col_chart, col_tbl = st.columns([2, 1])
            with col_chart:
                st.line_chart(df_future.set_index('Date')['PredictedAbsences'])
            with col_tbl:
                st.dataframe(df_future, height=260)

    # =========================================================================
    # TAB 3: HR INTELLIGENCE COPILOT (RAG + ANALYTICS)
    # =========================================================================
    with tab3:
        st.header("🤖 HR Intelligence Copilot & Policy Assistant")
        st.write("Ask natural language questions about workforce patterns, production team trends, or apparel HR policies.")

        # Preset query buttons for quick evaluation/interview demo
        st.markdown("**Quick Demonstration Queries:**")
        q_cols = st.columns(4)
        selected_prompt = None

        if q_cols[0].button("📊 Top Absenteeism Teams"):
            selected_prompt = "Which apparel production teams have the highest absenteeism, and what are their unauthorized leave rates?"
        if q_cols[1].button("📋 Unauthorized Leave SOP"):
            selected_prompt = "What is our standard operating procedure and disciplinary escalation for unauthorized leave (AWOL)?"
        if q_cols[2].button("🌙 Shift Comparison"):
            selected_prompt = "How does absenteeism compare between Day Shift and Night Shift in terms of duration and frequency?"
        if q_cols[3].button("🏥 Medical Certificate Policy"):
            selected_prompt = "What documentation and medical certificates are required for sick leave under our policy?"

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [
                {"role": "assistant", "content": "Hello! I am your HR Decision Support Copilot. Ask me anything about apparel production line attendance, leave types, or factory HR policies."}
            ]

        # Display history
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        user_input = st.chat_input("Ask a question about workforce analytics or HR policies...")
        prompt_to_run = user_input or selected_prompt

        if prompt_to_run:
            st.session_state["chat_history"].append({"role": "user", "content": prompt_to_run})
            with st.chat_message("user"):
                st.markdown(prompt_to_run)

            with st.chat_message("assistant"):
                with st.spinner("Consulting live analytics and HR policy knowledge base..."):
                    if hr_agent:
                        try:
                            reply = hr_agent.run_query(prompt_to_run, chat_history=st.session_state["chat_history"])
                        except Exception as e:
                            reply = f"Error processing query: {e}"
                    else:
                        reply = "HR Agent service is currently running in offline mode. Please verify `GROQ_API_KEY`."
                    st.markdown(reply)
                    st.session_state["chat_history"].append({"role": "assistant", "content": reply})

    # =========================================================================
    # TAB 4: EXECUTIVE MANAGEMENT REPORT
    # =========================================================================
    with tab4:
        st.header("📑 Executive Management Report Generator")
        st.write("Synthesize workforce predictions, team analytics, and policy recommendations into an executive summary.")

        # Top KPIs
        kpis = analytics_service.get_kpis() if analytics_service else {}
        if kpis:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Workforce", f"{kpis.get('total_employees', 0)} employees")
            k2.metric("High Risk Proportion", f"{kpis.get('high_risk_percentage', 0)}%", f"{kpis.get('high_risk_employees', 0)} workers")
            k3.metric("Avg Absent Days", f"{kpis.get('avg_absence_days', 0)} days")
            k4.metric("Unauthorized Incidents", f"{kpis.get('total_unauthorized_incidents', 0)} events")

        col_rep1, col_rep2 = st.columns([1, 1])

        with col_rep1:
            st.subheader("Team & Production Line Breakdown")
            team_df = analytics_service.get_team_analysis() if analytics_service else pd.DataFrame()
            if not team_df.empty:
                st.dataframe(team_df, hide_index=True)

        with col_rep2:
            st.subheader("Workforce Leave Distribution")
            leave_df = analytics_service.get_leave_type_distribution() if analytics_service else pd.DataFrame()
            if not leave_df.empty:
                fig_pie, ax_pie = plt.subplots(figsize=(5, 3))
                ax_pie.pie(leave_df['Count'], labels=leave_df['LeaveType'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
                ax_pie.axis('equal')
                st.pyplot(fig_pie)

        st.divider()

        if st.button("📝 Synthesize Executive AI Report", type="primary"):
            with st.spinner("Synthesizing multi-module analytics into executive management report..."):
                if llm_ready and llm_service:
                    try:
                        kpi_summary = json.dumps(kpis, indent=2)
                        team_summary_text = team_df.to_string(index=False) if not team_df.empty else "N/A"
                        leave_dist_text = leave_df.to_string(index=False) if not leave_df.empty else "N/A"
                        
                        forecast_text = "No forecast computed yet."
                        if "df_future" in st.session_state:
                            forecast_text = st.session_state["df_future"].head(7).to_string(index=False)

                        report_md = llm_service.generate_executive_report(
                            kpi_summary=kpi_summary,
                            team_summary_text=team_summary_text,
                            leave_dist_text=leave_dist_text,
                            forecast_summary_text=forecast_text
                        )
                        st.session_state["generated_report"] = report_md
                    except Exception as e:
                        st.error(f"Report synthesis notice: {e}")

        if "generated_report" in st.session_state:
            st.markdown(st.session_state["generated_report"])
            st.download_button(
                label="📥 Download Management Report (.md)",
                data=st.session_state["generated_report"],
                file_name="Absenteeism_Executive_Management_Report.md",
                mime="text/markdown"
            )
