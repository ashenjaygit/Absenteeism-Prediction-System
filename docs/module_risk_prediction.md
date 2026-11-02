# Module 1: Employee Risk Prediction (Classification)

## 🎯 Goal
 To identify individual employees who are likely to be absent frequently, allowing HR to intervene early.

## 🧠 The "Brain" (Model)
*   **Algorithm**: Random Forest Classifier.
*   **Type**: Supervised Classification.
*   **Accuracy**: ~47% (on current dataset).

## 📥 Input Data (What the model looks at)
The model analyzes the technical profile of each employee:
1.  **Demographics**: Age, Gender, Marital Status, Number of Dependents.
2.  **Job Profile**: Shift, Tenure (Years in company).
3.  **Leave History**: Current balances of Annual, Sick, Maternity, and Paternity leave.

## ⚙️ How it works
1.  The model looks at thousands of past employee records.
2.  It learns patterns, for example: *"Employees with low tenure (new joiners) who have used up all their sick leave are highly likely to be habitual absentees."*
3.  It assigns a "Risk Score" to every employee.

## 📤 Output (The Result)
The model places every employee into one of three buckets:
*   🟢 **Low Risk**: Likely to have normal attendance.
*   🟡 **Moderate Risk**: Shows some warning signs.
*   🔴 **High Risk**: Statistically likely to have high absenteeism. Proactive meeting recommended.

## 📊 Statistical Performance
We evaluated the **Random Forest Classifier** using standard classification metrics:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **47%** | Correctly predicts the exact risk category 47% of the time. |

### Detailed Class Report
| Risk Level | Precision | Recall | F1-Score | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| **High Risk** | **0.59** | **0.52** | **0.56** | When we say someone is High Risk, we are right **59%** of the time. |
| Moderate | 0.41 | 0.39 | 0.40 | Hardest to predict (middle ground). |
| Low Risk | 0.42 | 0.48 | 0.45 | Captures roughly half of the safe employees. |

*   **Precision (0.59 for High Risk)**: Crucial for HR. It means we don't produce too many "False Alarms".
*   **Recall (0.52 for High Risk)**: Means we catch about half of the truly problem cases.

