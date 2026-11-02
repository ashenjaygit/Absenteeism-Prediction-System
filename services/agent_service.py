import json
from services.analytics_service import HRAnalyticsService
from services.rag_service import HRPoplicyRAGService
from services.llm_service import LLMService

class HRAgent:
    """
    Agentic Decision-Support Assistant.
    Can inspect data, consult policies, run SHAP explanations, and synthesize multi-step answers.
    """
    def __init__(self, llm_service=None, analytics_service=None, rag_service=None, explainer_service=None):
        self.llm = llm_service or LLMService()
        self.analytics = analytics_service or HRAnalyticsService()
        self.rag = rag_service or HRPoplicyRAGService()
        self.explainer = explainer_service

    def run_query(self, user_prompt, chat_history=None):
        """
        Executes an agentic reasoning loop:
        1. Analyzes prompt to identify required context (analytics vs policy vs employee).
        2. Retrieves relevant datasets or policy clauses.
        3. Generates grounded, multi-perspective answer.
        """
        user_lower = user_prompt.lower()
        analytics_snippets = []
        policy_snippets = []

        # 1. Check if analytics or team data is relevant
        if any(w in user_lower for w in ["team", "department", "line", "highest", "lowest", "compare", "unauthorized", "rate", "absent", "trend", "kpi"]):
            team_df = self.analytics.get_team_analysis()
            if not team_df.empty:
                top_teams = team_df.head(5).to_dict(orient="records")
                analytics_snippets.append(f"Team Performance (Top 5 by Absence Days):\n{json.dumps(top_teams, indent=2)}")
            
            kpis = self.analytics.get_kpis()
            analytics_snippets.append(f"Workforce Overview KPIs:\n{json.dumps(kpis, indent=2)}")

        # 2. Check if leave types are queried
        if any(w in user_lower for w in ["leave type", "sick", "annual", "maternity", "paternity", "casual"]):
            leave_dist = self.analytics.get_leave_type_distribution()
            if not leave_dist.empty:
                analytics_snippets.append(f"Leave Type Distribution:\n{leave_dist.to_string(index=False)}")

        # 3. Check if shift dynamics are queried
        if any(w in user_lower for w in ["shift", "day shift", "night shift"]):
            shift_df = self.analytics.get_shift_analysis()
            if not shift_df.empty:
                analytics_snippets.append(f"Shift Comparison Data:\n{shift_df.to_string(index=False)}")

        # 4. Check for HR Policy RAG relevance
        if any(w in user_lower for w in ["policy", "rule", "procedure", "sop", "law", "certificate", "medical", "disciplinary", "consecutive", "notice", "action"]):
            retrieved_policy = self.rag.query_policy(user_prompt, n_results=3)
            policy_snippets.append(retrieved_policy)

        # Build context
        analytics_ctx = "\n\n".join(analytics_snippets) if analytics_snippets else None
        policy_ctx = "\n\n".join(policy_snippets) if policy_snippets else None

        # Fallback if neither specific keyword was triggered, do a broad policy + KPI check
        if not analytics_ctx and not policy_ctx:
            kpis = self.analytics.get_kpis()
            analytics_ctx = f"Workforce KPIs: {json.dumps(kpis)}"
            policy_ctx = self.rag.query_policy(user_prompt, n_results=2)

        return self.llm.chat_with_copilot(
            user_question=user_prompt,
            analytics_context=analytics_ctx,
            policy_context=policy_ctx,
            chat_history=chat_history
        )
