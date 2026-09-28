import joblib
import pandas as pd


model = joblib.load(
    "models/CatBoost_activity.pkl"
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
]

importance = model.feature_importances_

df = pd.DataFrame({

    "Feature": features,

    "Importance": importance

})

df = df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("-" * 60)

print(df)

df.to_csv(
    "results/feature_importance.csv",
    index=False
)

print(
    "\nSaved -> results/feature_importance.csv"
)