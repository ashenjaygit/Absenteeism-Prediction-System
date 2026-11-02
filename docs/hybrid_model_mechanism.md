# 🧬 The Hybrid Model: Mechanism & Architecture

This document explains the "Hybrid Ensemble" used in the Absenteeism Prediction System. It combines two fundamentally different Artificial Intelligence architectures—**XGBoost** and **LSTM**—to create a stronger, more robust predictor.

---

## 1. The Concept: "Ensemble Learning"
The core mechanism is **Ensemble Learning**. Instead of relying on a single "brain" to make decisions, we use two different experts and average their opinions.
*   **Analogy**: It’s like asking both a **Mathematician (XGBoost)** and a **Historian (LSTM)** to predict the future. The Mathematician looks at the stats, while the Historian looks at the timeline. Combining their answers usually eliminates extreme errors.

---

## 2. The Components (The Two Brains)

### 🧠 Model A: XGBoost (The Mathematician)
*   **Type**: Tree-Based Model (Gradient Boosting).
*   **Strength**: Excellent at interpreting distinct features like "Leave Balance", "Day of Week", and "Holiday Status". It sees the data as a structured table.
*   **Weakness**: Can sometimes struggle with complex, long-term trends in time-series data.

### 🧠 Model B: LSTM (The Historian)
*   **Type**: Deep Neural Network (Recurrent Neural Network).
*   **Strength**: Built specifically for **Time**. It has internal "memory cells" that remember what happened 7 days ago or 30 days ago. It treats the data as a flow, not a table.
*   **Weakness**: Requires a lot of data and can overreact to recent noise.

---

## 3. The Connection Mechanism

The models are **Not** connected during training (they train separately). They are connected during **Inference (Prediction)**.

### Step 1: Independent Training
*   We train **XGBoost** on the dataset to minimize error.
*   We train **LSTM** on the *same* dataset (reshaped for time-sequences) to minimize error.

### Step 2: Parallel Prediction
When you click "Predict" on the dashboard:
1.  Input data is sent to **XGBoost** → It outputs Prediction A (e.g., *12 people*).
2.  Input data is sent to **LSTM** → It outputs Prediction B (e.g., *14 people*).

### Step 3: Weighted Averaging (The "Hybrid" Logic)
The system takes both outputs and combines them using a mathematical formula:

$$ \text{Final Prediction} = \frac{\text{XGBoost Prediction} + \text{LSTM Prediction}}{2} $$

*   *Note: In our code, we use equal weighting (50/50), but this can be adjusted if one model is significantly better than the other.*

---

## 4. Why this works (The "Error Cancellation" Effect)
Different models make different *types* of mistakes.
*   If XGBoost predicts too high (Overestimates) and LSTM predicts too low (Underestimates), averaging them brings the final answer closer to the distinct truth.
*   This makes the Hybrid model **Low Variance**, meaning it is stable and less likely to crash or give distinctively wild predictions compared to a single model.
