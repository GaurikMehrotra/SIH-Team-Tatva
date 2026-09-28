import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


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

model = LGBMClassifier(
    random_state=42
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="f1"
)

print("\nCROSS VALIDATION")
print("-" * 60)

for i, score in enumerate(scores):

    print(
        f"Fold {i+1}: {score:.4f}"
    )

print(
    f"\nMean F1 : {scores.mean():.4f}"
)

print(
    f"Std Dev : {scores.std():.4f}"
)