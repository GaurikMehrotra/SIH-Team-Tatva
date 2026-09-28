import pandas as pd

from simulator.environment import RFEnvironment
from simulator.history_tracker import HistoryTracker
from features.activity_features import extract_features


env = RFEnvironment(10)

tracker = HistoryTracker(20)

rows = []

N_STEPS = 500


for step in range(N_STEPS):

    active_now = env.step()

    tracker.update(active_now)

    if step < 6:
        continue

    feature_rows = []

    for emitter in env.emitters:

        features = extract_features(
            tracker,
            emitter.emitter_id
        )

        row = {

            "emitter_id":
                emitter.emitter_id,

            "activity_ratio":
                features["activity_ratio"],

            "consecutive_activity":
                features["consecutive_activity"],

            "time_since_last_seen":
                features["time_since_last_seen"],

            "last_5_0":
                features["last_5_activity"][0],

            "last_5_1":
                features["last_5_activity"][1],

            "last_5_2":
                features["last_5_activity"][2],

            "last_5_3":
                features["last_5_activity"][3],

            "last_5_4":
                features["last_5_activity"][4],
        }

        feature_rows.append(row)

    active_next = env.step()

    next_ids = {
        e.emitter_id
        for e in active_next
    }

    for row in feature_rows:

        row["target"] = (
            1
            if row["emitter_id"] in next_ids
            else 0
        )

        rows.append(row)

    tracker.update(active_next)


df = pd.DataFrame(rows)

print(df.head())

print("\nShape:")
print(df.shape)

df.to_csv(
    "results/prediction_dataset_v2.csv",
    index=False
)

print(
    "\nSaved -> results/prediction_dataset_v2.csv"
)