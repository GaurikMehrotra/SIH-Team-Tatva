import pandas as pd

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

experiments = {

    "FullModel": [

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
    ],

    "NoSignalStrength": [

        "activity_ratio",
        "consecutive_activity",
        "time_since_last_seen",
        "last_5_0",
        "last_5_1",
        "last_5_2",
        "last_5_3",
        "last_5_4",
        "num_receivers",
    ],

    "NoCooperativeFeatures": [

        "activity_ratio",
        "consecutive_activity",
        "time_since_last_seen",
        "last_5_0",
        "last_5_1",
        "last_5_2",
        "last_5_3",
        "last_5_4",
    ],
}

results = []

for experiment_name, features in experiments.items():

    print("\n" + "=" * 60)
    print(f"Training {experiment_name}")
    print("=" * 60)

    X = df[features]

    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
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
            "Experiment": experiment_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    )

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)

print("\n")
print("=" * 60)
print("SENSOR ABLATION RESULTS")
print("=" * 60)

print(results_df)

results_df.to_csv(
    "results/sensor_ablation.csv",
    index=False
)

print(
    "\nSaved -> results/sensor_ablation.csv"
)