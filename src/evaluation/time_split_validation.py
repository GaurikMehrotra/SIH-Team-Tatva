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

split_index = int(
    len(df) * 0.8
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

model = LGBMClassifier(
    random_state=42
)

print("\nTraining Model...")

model.fit(
    X_train,
    y_train
)

print("Running Predictions...")

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("\nTIME SPLIT RESULTS")
print("-" * 60)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)