import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


print("Loading Dataset...")

df = pd.read_csv(
    "results/prediction_dataset_v3.csv"
)

X = df.drop(
    columns=[
        "emitter_id",
        "target"
    ]
)

y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining LightGBM...")

model = LGBMClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)

noise_levels = [
    0.00,
    0.05,
    0.10,
    0.20,
]

results = []

print("\nNOISE ROBUSTNESS TEST")
print("-" * 60)

for noise in noise_levels:

    X_noisy = X_test.copy()

    numeric_columns = X_noisy.columns

    for col in numeric_columns:

        std = X_noisy[col].std()

        noise_values = np.random.normal(
            0,
            std * noise,
            size=len(X_noisy)
        )

        X_noisy[col] = (
            X_noisy[col]
            + noise_values
        )

    y_pred = model.predict(
        X_noisy
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

    print(
        f"\nNoise Level {int(noise*100)}%"
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

    results.append(
        {
            "noise_level": noise,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "results/noise_robustness.csv",
    index=False
)

print(
    "\nSaved -> results/noise_robustness.csv"
)