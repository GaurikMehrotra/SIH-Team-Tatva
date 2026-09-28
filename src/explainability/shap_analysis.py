import joblib
import shap
import pandas as pd
import numpy as np


print("Loading model...")

model = joblib.load(
    "models/CatBoost_activity.pkl"
)

print("Loading dataset...")

df = pd.read_csv(
    "results/prediction_dataset_v2.csv"
)

X = df.drop(
    columns=[
        "emitter_id",
        "target",
    ]
)

print("Dataset Shape:", X.shape)

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(
    model
)

sample = X.sample(
    min(1000, len(X)),
    random_state=42
)

shap_values = explainer.shap_values(
    sample
)

mean_abs_shap = np.abs(
    shap_values
).mean(axis=0)

importance_df = pd.DataFrame({

    "Feature": sample.columns,

    "MeanAbsSHAP": mean_abs_shap

})

importance_df = importance_df.sort_values(
    by="MeanAbsSHAP",
    ascending=False
)

print("\nSHAP FEATURE IMPORTANCE")
print("-" * 60)

print(importance_df)

importance_df.to_csv(
    "results/shap_importance.csv",
    index=False
)

print(
    "\nSaved -> results/shap_importance.csv"
)