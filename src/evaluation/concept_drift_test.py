import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

print("Loading Dataset...")

df = pd.read_csv(
    "results/prediction_dataset_v3.csv"
)

features = [
    "activity_ratio",
    "consecutive_activity",
    "time_since_last_seen",
    "last_5_0",
    "last_5_1",
    "last_5_2",
    "last_5_3",
    "last_5_4",
    "avg_signal_strength",
    "num_receivers",
]

X = df[features]
y = df["target"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining LightGBM...")

model = LGBMClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# -----------------------------
# BASELINE
# -----------------------------

baseline_pred = model.predict(X_test)

baseline_f1 = f1_score(
    y_test,
    baseline_pred
)

# -----------------------------
# CONCEPT DRIFT SIMULATION
# -----------------------------

drift_X = X_test.copy()

# Convert to float before modifying
drift_X["consecutive_activity"] = (
    drift_X["consecutive_activity"]
    .astype(float)
)

drift_X["avg_signal_strength"] = (
    drift_X["avg_signal_strength"]
    .astype(float)
)

# Emitters become less persistent
drift_X["consecutive_activity"] *= 0.50

# Signals become weaker
drift_X["avg_signal_strength"] *= 0.70

# Receivers lose coverage
drift_X["num_receivers"] = (
    drift_X["num_receivers"] - 1
).clip(lower=1)

# -----------------------------
# DRIFT EVALUATION
# -----------------------------

drift_pred = model.predict(
    drift_X
)

accuracy = accuracy_score(
    y_test,
    drift_pred
)

precision = precision_score(
    y_test,
    drift_pred
)

recall = recall_score(
    y_test,
    drift_pred
)

drift_f1 = f1_score(
    y_test,
    drift_pred
)

print("\nCONCEPT DRIFT TEST")
print("-" * 60)

print(
    f"Baseline F1 : {baseline_f1:.4f}"
)

print(
    f"Drift F1    : {drift_f1:.4f}"
)

print(
    f"Accuracy    : {accuracy:.4f}"
)

print(
    f"Precision   : {precision:.4f}"
)

print(
    f"Recall      : {recall:.4f}"
)

print(
    f"F1 Score    : {drift_f1:.4f}"
)

drop_percent = (
    (baseline_f1 - drift_f1)
    / baseline_f1
) * 100

print(
    f"\nPerformance Drop : {drop_percent:.2f}%"
)