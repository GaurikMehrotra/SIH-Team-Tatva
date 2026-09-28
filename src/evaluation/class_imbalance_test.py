import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from lightgbm import LGBMClassifier

print("Loading Dataset...")

df = pd.read_csv("results/prediction_dataset_v3.csv")

X = df.drop(columns=["emitter_id", "target"])
y = df["target"]

# --------------------------------------------------
# Create Severe Class Imbalance
# Keep all negatives, keep only 10% positives
# --------------------------------------------------

positive = df[df["target"] == 1]
negative = df[df["target"] == 0]

positive_small = positive.sample(
    frac=0.10,
    random_state=42
)

imbalanced_df = pd.concat(
    [negative, positive_small],
    ignore_index=True
)

print("\nCLASS DISTRIBUTION")
print("-" * 60)

print(
    imbalanced_df["target"]
    .value_counts()
    .sort_index()
)

X = imbalanced_df.drop(
    columns=["emitter_id", "target"]
)

y = imbalanced_df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
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

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

print("\nCLASS IMBALANCE TEST")
print("-" * 60)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")