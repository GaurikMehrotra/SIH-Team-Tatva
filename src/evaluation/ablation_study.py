import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from catboost import CatBoostClassifier


print("Loading Dataset...")

df = pd.read_csv(
    "results/prediction_dataset_v2.csv"
)


experiments = {

    "ActivityRatio": [

        "activity_ratio",

    ],

    "ActivityRatio_TimeSinceSeen": [

        "activity_ratio",
        "time_since_last_seen",

    ],

    "CoreFeatures": [

        "activity_ratio",
        "time_since_last_seen",
        "consecutive_activity",

    ],

    "FullFeatures": [

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

y = df["target"]


for name, feature_list in experiments.items():

    print("\n" + "=" * 60)
    print(f"Training {name}")
    print("=" * 60)

    X = df[feature_list]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        verbose=0,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict(
        X_test
    )

    acc = accuracy_score(
        y_test,
        preds
    )

    prec = precision_score(
        y_test,
        preds
    )

    rec = recall_score(
        y_test,
        preds
    )

    f1 = f1_score(
        y_test,
        preds
    )

    print(
        f"Accuracy  : {acc:.4f}"
    )

    print(
        f"Precision : {prec:.4f}"
    )

    print(
        f"Recall    : {rec:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    results.append(
        [
            name,
            acc,
            prec,
            rec,
            f1,
        ]
    )


results_df = pd.DataFrame(

    results,

    columns=[
        "Experiment",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]
)

results_df = results_df.sort_values(
    by="F1",
    ascending=False,
)

print("\n")
print("=" * 60)
print("ABLATION STUDY RESULTS")
print("=" * 60)

print(results_df)

results_df.to_csv(
    "results/ablation_study.csv",
    index=False,
)

print(
    "\nSaved -> results/ablation_study.csv"
)