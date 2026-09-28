import pandas as pd

from simulator.environment import RFEnvironment
from simulator.history_tracker import HistoryTracker
from features.activity_features import extract_features


env = RFEnvironment(10)

tracker = HistoryTracker(20)

rows = []

N_STEPS = 500


for step in range(N_STEPS):

    active_emitters = env.step()

    tracker.update(active_emitters)

    active_ids = {
        e.emitter_id
        for e in active_emitters
    }

    for emitter in env.emitters:

        history = tracker.get_history(
            emitter.emitter_id
        )

        if len(history) < 6:
            continue

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

            "target":
                1 if emitter.emitter_id in active_ids else 0
        }

        rows.append(row)


df = pd.DataFrame(rows)

print(df.head())

print("\nShape:")
print(df.shape)

df.to_csv(
    "results/prediction_dataset.csv",
    index=False
)

print(
    "\nSaved -> results/prediction_dataset.csv"
)