"""
train_model.py
---------------
Generates the synthetic student dataset, trains the Linear Regression
model (same logic as the Colab script from Step 1), and saves the
trained model to disk so a separate web app (Step 3) can load it and
make predictions -- without needing to retrain every time.

Run with:
    python train_model.py
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Generate the synthetic dataset (identical logic to Step 1)
# ---------------------------------------------------------
np.random.seed(42)
n_samples = 300

study_hours = np.random.normal(4, 1.5, n_samples).clip(0, 10)
attendance = np.random.normal(80, 12, n_samples).clip(40, 100)
previous_score = np.random.normal(65, 15, n_samples).clip(0, 100)
sleep_hours = np.random.normal(6.5, 1.2, n_samples).clip(3, 10)

noise = np.random.normal(0, 5, n_samples)
final_score = (
    10
    + 5.0 * study_hours
    + 0.25 * attendance
    + 0.35 * previous_score
    + 1.5 * sleep_hours
    + noise
).clip(0, 100)

df = pd.DataFrame({
    "study_hours": study_hours.round(2),
    "attendance": attendance.round(2),
    "previous_score": previous_score.round(2),
    "sleep_hours": sleep_hours.round(2),
    "final_score": final_score.round(2)
})

df.to_csv("data/student_data.csv", index=False)
print(f"Dataset saved to data/student_data.csv ({len(df)} rows)")

# ---------------------------------------------------------
# 2. Train the Linear Regression model
# ---------------------------------------------------------
X = df[["study_hours", "attendance", "previous_score", "sleep_hours"]]
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Model R^2 on test set: {r2:.3f}")

# ---------------------------------------------------------
# 3. Save the trained model
# ---------------------------------------------------------
# This is the KEY change from the Colab version: instead of taking
# console input right here, we save the model so app.py (Step 3) can
# load it independently and get input from a web form instead.
joblib.dump(model, "model/linear_regression_model.pkl")
print("Model saved to model/linear_regression_model.pkl")

# ---------------------------------------------------------
# 4. Save the Actual vs Predicted plot to a file
# ---------------------------------------------------------
# Changed from plt.show() (Colab) to plt.savefig() (script), since a
# plain script has no notebook cell to display the plot inline.
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="steelblue")
plt.plot([0, 100], [0, 100], color="red", linestyle="--", label="Perfect Prediction")
plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Scores")
plt.legend()
plt.savefig("model/actual_vs_predicted.png", dpi=120)
print("Plot saved to model/actual_vs_predicted.png")
