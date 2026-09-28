import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split

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

X = df[features].copy()

y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTraining LightGBM...")

model = LGBMClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)

baseline_pred = model.predict(X_test)

baseline_f1 = f1_score(
    y_test,
    baseline_pred
)

adversarial_X = X_test.copy()

np.random.seed(42)

rows = int(
    len(adversarial_X) * 0.20
)

indices = np.random.choice(
    adversarial_X.index,
    rows,
    replace=False,
)

adversarial_X.loc[
    indices,
    "avg_signal_strength"
] *= 0.20

adversarial_X.loc[
    indices,
    "num_receivers"
] = 1

adversarial_pred = model.predict(
    adversarial_X
)

accuracy = accuracy_score(
    y_test,
    adversarial_pred
)

precision = precision_score(
    y_test,
    adversarial_pred
)

recall = recall_score(
    y_test,
    adversarial_pred
)

f1 = f1_score(
    y_test,
    adversarial_pred
)

print("\nADVERSARIAL EW TEST")
print("-" * 60)

print(
    f"Baseline F1 : {baseline_f1:.4f}"
)

print(
    f"Adversarial F1 : {f1:.4f}"
)

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

drop = (
    (baseline_f1 - f1)
    /
    baseline_f1
) * 100

print(
    f"\nPerformance Drop : {drop:.2f}%"
)