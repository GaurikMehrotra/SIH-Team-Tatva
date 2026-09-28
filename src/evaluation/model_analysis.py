import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)

df = pd.read_csv(
    "results/prediction_dataset_v2.csv"
)

X = df.drop(
    columns=[
        "emitter_id",
        "target",
    ]
)

y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = joblib.load(
    "models/CatBoost_activity.pkl"
)

preds = model.predict(X_test)

print("\nCONFUSION MATRIX")
print("-" * 50)

print(
    confusion_matrix(
        y_test,
        preds
    )
)

print("\nCLASSIFICATION REPORT")
print("-" * 50)

print(
    classification_report(
        y_test,
        preds
    )
)