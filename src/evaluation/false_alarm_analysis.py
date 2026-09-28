from sklearn.metrics import confusion_matrix
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split


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

tn, fp, fn, tp = confusion_matrix(
    y_test,
    preds
).ravel()

false_alarm_rate = fp / (fp + tn)

detection_rate = tp / (tp + fn)

print("\nEW METRICS")
print("-" * 50)

print(
    f"Detection Rate (Pd) : {detection_rate:.4f}"
)

print(
    f"False Alarm Rate(Pfa): {false_alarm_rate:.4f}"
)