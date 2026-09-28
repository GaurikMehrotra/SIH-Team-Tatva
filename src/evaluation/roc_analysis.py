import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LGBMClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)

probs = model.predict_proba(
    X_test
)[:, 1]

auc = roc_auc_score(
    y_test,
    probs
)

fpr, tpr, thresholds = roc_curve(
    y_test,
    probs
)

print("\nROC ANALYSIS")
print("-" * 60)

print(
    f"AUC Score : {auc:.4f}"
)

print("\nSample ROC Points")
print("-" * 60)

for i in range(
    0,
    len(fpr),
    max(1, len(fpr)//10)
):

    print(
        f"Threshold={thresholds[i]:.4f} "
        f"TPR={tpr[i]:.4f} "
        f"FPR={fpr[i]:.4f}"
    )