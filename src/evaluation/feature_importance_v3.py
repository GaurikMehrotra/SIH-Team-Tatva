import pandas as pd
import joblib


model = joblib.load(
    "models/LightGBM_v3.pkl"
)

df = pd.read_csv(
    "results/prediction_dataset_v3.csv"
)

X = df.drop(
    columns=[
        "emitter_id",
        "target"
    ]
)

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
        model.feature_importances_
})

importance = importance.sort_values(

    by="Importance",

    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("-" * 60)

print(importance)

importance.to_csv(

    "results/feature_importance_v3.csv",

    index=False
)

print(
    "\nSaved -> results/feature_importance_v3.csv"
)