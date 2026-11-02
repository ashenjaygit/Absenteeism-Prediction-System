# Theoretical Background: Supervised vs. Unsupervised Learning

## ❓ The Core Question
**Is this project Supervised or Unsupervised?**

**ANSWER**: This entire project is **100% SUPERVISED LEARNING**.

---

## 📚 Theory: What is the difference?

### 1. Supervised Learning (Like a Teacher & Student)
*   **Definition**: The model is trained on data where **we already know the correct answer** (the Label or Target).
*   **Process**: The model makes a guess, compares it to the "Answer Key", and corrects itself.
*   **Example**: "Here is a picture of a cat (Input). The label is 'Cat' (Output). Learn this."

### 2. Unsupervised Learning (Like Exploration)
*   **Definition**: The model is given data **without any labels** or correct answers. It must find patterns on its own.
*   **Process**: The model groups similar items together (Clustering).
*   **Example**: "Here are 1,000 photos. Group them into piles of similar things." (It might group cats together, but it doesn't know they are called "cats").

---

## 🛠️ How it applies to THIS Project

We used **Supervised Learning** for both parts of the project because we possess **Historical Data** with known outcomes.

### Scenario A: Employee Risk Prediction
*   **Type**: **Supervised Classification**
*   **Input**: Employee Age, Tenure, Shift, etc.
*   **The "Answer Key" (Label)**: `RiskCategory` (High/Moderate/Low).
*   **Why it is Supervised**:
    *   We didn't ask the computer to "group employees freely" (Clustering).
    *   Instead, we calculated the *actual* absent days from history, assigned a specific label (e.g., "High Risk"), and told the Random Forest: *"This employee had 20 absent days, so they are HIGH RISK. Learn to recognize employees like this."*

### Scenario B: Daily Forecasting
*   **Type**: **Supervised Regression** (Time-Series)
*   **Input**: Date, Day of Week, Previous Day's Count.
*   **The "Answer Key" (Label)**: `AbsentCount` (The actual number of people absent).
*   **Why it is Supervised**:
    *   We fed the LSTM model the history: *"On Jan 1st, 12 people were absent. On Jan 2nd, 15 people were absent."*
    *   The model tries to predict the number **15**, checks how close it got, and adjusts its math to minimize the error.

---

## 🎓 Summary Table

| Feature | Supervised Learning (Used Here) | Unsupervised Learning (Not Used) |
| :--- | :--- | :--- |
| **Data** | Labeled (Input + Correct Output) | Unlabeled (Input only) |
| **Goal** | Prediction / Classification | Clustering / Pattern Finding |
| **Feedback** | Direct ("You were wrong by 2 points") | None / Internal validity only |
| **Algorithms**| Random Forest, XGBoost, LSTM | K-Means Clustering, PCA |
