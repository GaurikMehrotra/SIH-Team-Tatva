import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


print("Loading Model...")

model = joblib.load(
    "models/CatBoost_activity.pkl"
)

print("Loading Dataset...")

df = pd.read_csv(
    "results/prediction_dataset_v2.csv"
)

feature_cols = [

    "activity_ratio",
    "consecutive_activity",
    "time_since_last_seen",

    "last_5_0",
    "last_5_1",
    "last_5_2",
    "last_5_3",
    "last_5_4",
]

X = df[feature_cols]

y = df["target"]

print(
    f"Dataset Shape: {df.shape}"
)

print(
    "\nRunning Predictions..."
)

predictions = model.predict(X)

accuracy = accuracy_score(
    y,
    predictions
)

precision = precision_score(
    y,
    predictions
)

recall = recall_score(
    y,
    predictions
)

f1 = f1_score(
    y,
    predictions
)

cm = confusion_matrix(
    y,
    predictions
)

tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]

pd_metric = tp / (tp + fn)

pfa_metric = fp / (fp + tn)

print("\nRESULTS")
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

print()

print(
    f"Pd  : {pd_metric:.4f}"
)

print(
    f"Pfa : {pfa_metric:.4f}"
)

print()

print("CONFUSION MATRIX")
print("-" * 60)

print(cm)

results = pd.DataFrame({

    "Metric": [

        "Accuracy",
        "Precision",
        "Recall",
        "F1",

        "Pd",
        "Pfa",
    ],

    "Value": [

        accuracy,
        precision,
        recall,
        f1,

        pd_metric,
        pfa_metric,
    ]
})

results.to_csv(
    "results/replay_evaluation.csv",
    index=False
)

print(
    "\nSaved -> results/replay_evaluation.csv"
)