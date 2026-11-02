import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Feature friendly names for HR presentation
FEATURE_NAME_MAP = {
    'Age': 'Employee Age',
    'Gender_Code': 'Gender',
    'Marital_Code': 'Marital Status',
    'NumberOfDependents': 'Dependents Count',
    'Shift_Code': 'Assigned Shift',
    'Tenure_Years': 'Tenure at Company (Years)',
    'AnnualLeaveBalance': 'Annual Leave Balance',
    'SickLeaveBalance': 'Sick Leave Balance',
    'MaternityLeaveBalance': 'Maternity Balance',
    'PaternityLeaveBalance': 'Paternity Balance',
    'TotalLeaveBalance': 'Total Leave Balance',
    'UtilizationRate': 'Historical Leave Utilization Rate',
    'Seniority': 'Seniority Index (Age x Tenure)'
}

class AbsenteeismExplainer:
    def __init__(self, model, background_data, feature_names):
        """
        Initializes SHAP KernelExplainer for the classification model.
        """
        self.model = model
        self.feature_names = feature_names
        # Take a sample of 30 rows as background for fast and accurate Kernel SHAP
        self.background = shap.sample(background_data[feature_names], min(30, len(background_data)), random_state=42)
        self.explainer = shap.KernelExplainer(self.model.predict_proba, self.background)
        self.classes = list(self.model.classes_)

    def explain_instance(self, input_row_df, nsamples=40):
        """
        Computes SHAP values for a single employee instance.
        Returns:
            - predicted_class: str
            - confidence: float
            - feature_contributions: list of dicts with feature details
            - shap_array: raw shap values for plotting
        """
        X_inst = input_row_df[self.feature_names]
        probs = self.model.predict_proba(X_inst)[0]
        pred_idx = int(np.argmax(probs))
        predicted_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        # Compute SHAP values: shape (1, num_features, num_classes) or list of arrays
        shap_vals = self.explainer.shap_values(X_inst, nsamples=nsamples)
        
        # Normalize shap_vals array format
        if isinstance(shap_vals, list):
            class_shap = shap_vals[pred_idx][0]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            class_shap = shap_vals[0, :, pred_idx]
        else:
            class_shap = np.array(shap_vals).flatten()

        contributions = []
        for feat, val, s_val in zip(self.feature_names, X_inst.iloc[0], class_shap):
            friendly_name = FEATURE_NAME_MAP.get(feat, feat)
            impact = "Increases Risk" if s_val > 0 else "Decreases Risk"
            contributions.append({
                "feature": feat,
                "display_name": friendly_name,
                "feature_value": float(val) if isinstance(val, (int, float, np.number)) else str(val),
                "shap_value": float(s_val),
                "impact": impact,
                "abs_importance": abs(float(s_val))
            })

        # Sort by absolute SHAP importance descending
        contributions.sort(key=lambda x: x["abs_importance"], reverse=True)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": {cls: float(p) for cls, p in zip(self.classes, probs)},
            "contributions": contributions,
            "top_drivers": contributions[:5]
        }

    def generate_shap_plot(self, explanation_result):
        """
        Creates a clean, executive horizontal bar plot of top SHAP contributors.
        """
        drivers = explanation_result["top_drivers"]
        names = [d["display_name"] for d in reversed(drivers)]
        values = [d["shap_value"] for d in reversed(drivers)]
        colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in values]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(names, values, color=colors, edgecolor='none', height=0.6)
        
        ax.axvline(0, color='#888888', linestyle='--', linewidth=0.8)
        ax.set_title(f"SHAP Feature Attribution: Drivers for '{explanation_result['predicted_class']}' Risk", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("SHAP Value (Contribution to Prediction Probability)", fontsize=9)
        
        # Annotate bar values
        for bar in bars:
            width = bar.get_width()
            align = 'left' if width >= 0 else 'right'
            offset = 0.005 if width >= 0 else -0.005
            ax.text(width + offset, bar.get_y() + bar.get_height()/2, f"{width:+.3f}",
                    va='center', ha=align, fontsize=8, color='#333333', fontweight='bold')
        
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        fig.tight_layout()
        return fig
