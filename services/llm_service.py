import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]

class LLMService:
    def __init__(self, api_key=None, model="openai/gpt-oss-120b"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set. Please provide it in .env or pass it explicitly.")
        self.client = Groq(api_key=self.api_key)
        self.primary_model = model

    def _call_llm(self, messages, temperature=0.3, max_tokens=1000):
        """Calls Groq API with automatic model fallback."""
        models_to_try = [self.primary_model] + [m for m in DEFAULT_MODELS if m != self.primary_model]
        last_error = None

        for m in models_to_try:
            try:
                response = self.client.chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                continue
        
        raise RuntimeError(f"All Groq models failed. Last error: {last_error}")

    def explain_prediction(self, emp_id, emp_info, risk_category, confidence, top_drivers):
        """
        Translates mathematical SHAP values into an ethical, human-centric HR explanation.
        """
        drivers_text = "\n".join([
            f"- {d['display_name']}: Value = {d['feature_value']}, SHAP Impact = {d['shap_value']:+.3f} ({d['impact']})"
            for d in top_drivers
        ])

        system_prompt = (
            "You are an expert HR Decision Support Assistant specializing in the apparel manufacturing industry. "
            "You assist HR leaders by explaining predictive machine learning results transparently and ethically. "
            "IMPORTANT GUIDELINES:\n"
            "1. Predictions indicate operational risk/pattern likelihood, NOT employee wrongdoing or proven causality.\n"
            "2. Never make medical diagnoses or personal character judgments.\n"
            "3. Frame all recommendations around supportive, proactive HR actions (e.g., fatigue check-ins, line balancing, schedule flexibility).\n"
            "4. Keep the output structured, concise, and professional."
        )

        user_prompt = f"""
Employee ID: {emp_id}
Demographics & Balances:
- Age: {emp_info.get('Age', 'N/A')}
- Gender: {emp_info.get('Gender', 'N/A')} | Marital Status: {emp_info.get('MaritalStatus', 'N/A')}
- Shift: {emp_info.get('Shift', 'N/A')} | Tenure: {emp_info.get('Tenure_Years', 'N/A')} years
- Annual Leave Balance: {emp_info.get('AnnualLeaveBalance', 'N/A')} | Sick Leave Balance: {emp_info.get('SickLeaveBalance', 'N/A')}
- Past Absent Days: {emp_info.get('AbsenceDuration_Days', 'N/A')} days

Model Output:
- Predicted Absenteeism Risk: {risk_category}
- Prediction Confidence: {confidence:.1%}

Key Contributing Factors (from SHAP Analysis):
{drivers_text}

Provide an executive explanation with three concise sections:
1. **Risk Assessment Summary**: A 2-sentence summary of the predicted level and reliability.
2. **Key Driver Analysis**: Explain how the top SHAP features influenced this outcome in plain HR language.
3. **Recommended HR Proactive Actions**: 3 actionable, empathetic, non-punitive recommendations for HR or line supervisors.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self._call_llm(messages, temperature=0.2)

    def chat_with_copilot(self, user_question, analytics_context=None, policy_context=None, chat_history=None):
        """
        Answers natural language HR questions by combining live analytics context and retrieved policy snippets.
        """
        system_prompt = (
            "You are an AI-powered HR Decision Support Assistant for an apparel manufacturing enterprise. "
            "You have access to live workforce absenteeism data and factory HR policies (Sri Lankan labor law & SOPs).\n"
            "GUIDELINES:\n"
            "- Always base your answers on the provided Analytics Context and Policy Context.\n"
            "- If the question is about specific numbers (e.g. teams, rates), cite the numbers from the context.\n"
            "- If the question is about rules/procedures (e.g. sick leave, unauthorized leave), cite the relevant policy clauses.\n"
            "- Be concise, direct, professional, and clear."
        )

        context_blocks = []
        if analytics_context:
            context_blocks.append(f"### LIVE WORKFORCE ANALYTICS CONTEXT:\n{analytics_context}")
        if policy_context:
            context_blocks.append(f"### RETRIEVED HR POLICY CONTEXT:\n{policy_context}")

        full_context_str = "\n\n".join(context_blocks)

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add prior history if provided
        if chat_history:
            for item in chat_history[-6:]:
                messages.append({"role": item["role"], "content": item["content"]})

        current_content = f"{full_context_str}\n\n### HR MANAGER QUESTION:\n{user_question}"
        messages.append({"role": "user", "content": current_content})

        return self._call_llm(messages, temperature=0.3)

    def generate_executive_report(self, kpi_summary, team_summary_text, leave_dist_text, forecast_summary_text):
        """
        Generates an end-of-month / quarterly executive management report on workforce absenteeism.
        """
        system_prompt = (
            "You are a Senior Strategic HR & Workforce Operations Consultant. "
            "Generate a formal, publication-ready Executive Workforce Absenteeism & Decision Support Report."
        )

        user_prompt = f"""
Generate an Executive Absenteeism Intelligence Report for Apparel Factory Leadership based on the following verified data:

1. WORKFORCE KEY METRICS:
{kpi_summary}

2. PRODUCTION TEAM / LINE ABSENTEEISM BREAKDOWN:
{team_summary_text}

3. LEAVE DISTRIBUTION:
{leave_dist_text}

4. TIME-SERIES FORECASTING SUMMARY:
{forecast_summary_text}

Format the report using clean Markdown with:
- **Executive Summary** (Key takeaways for Operations & HR Heads)
- **Production Line / Team Vulnerabilities** (Highlighting lines with high unauthorized or total absences)
- **Workforce Risk & Shift Dynamics** (Observations on shift fatigue and risk tiers)
- **Operational & Policy Recommendations** (Specific actionable interventions for supervisors and HR)
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self._call_llm(messages, temperature=0.25, max_tokens=1500)
