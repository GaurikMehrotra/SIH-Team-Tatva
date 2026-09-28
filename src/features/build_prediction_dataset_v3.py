import pandas as pd

from simulator.cooperative_environment import (
    CooperativeEnvironment
)

from simulator.fusion_engine import (
    FusionEngine
)

from simulator.history_tracker import (
    HistoryTracker
)

from features.activity_features import (
    extract_features
)


env = CooperativeEnvironment(

    n_emitters=10,

    n_receivers=3
)

fusion_engine = FusionEngine()

tracker = HistoryTracker(20)

rows = []

N_STEPS = 500


for step in range(N_STEPS):

    observations = env.step()

    fused = fusion_engine.fuse(
        observations
    )

    active_ids = set(
        fused.keys()
    )

    tracker.update(

        [

            type(
                "Emitter",
                (),
                {
                    "emitter_id": eid
                }
            )

            for eid in active_ids
        ]
    )

    if step < 6:
        continue

    feature_rows = []

    for emitter in env.emitters:

        emitter_id = emitter.emitter_id

        features = extract_features(
            tracker,
            emitter_id
        )

        fusion_data = fused.get(

            emitter_id,

            {
                "avg_strength": 0.0,
                "num_receivers": 0
            }
        )

        row = {

            "emitter_id":
                emitter_id,

            "activity_ratio":
                features["activity_ratio"],

            "consecutive_activity":
                features["consecutive_activity"],

            "time_since_last_seen":
                features[
                    "time_since_last_seen"
                ],

            "last_5_0":
                features[
                    "last_5_activity"
                ][0],

            "last_5_1":
                features[
                    "last_5_activity"
                ][1],

            "last_5_2":
                features[
                    "last_5_activity"
                ][2],

            "last_5_3":
                features[
                    "last_5_activity"
                ][3],

            "last_5_4":
                features[
                    "last_5_activity"
                ][4],

            "avg_signal_strength":
                fusion_data[
                    "avg_strength"
                ],

            "num_receivers":
                fusion_data[
                    "num_receivers"
                ]
        }

        feature_rows.append(
            row
        )

    next_obs = env.step()

    next_fused = fusion_engine.fuse(
        next_obs
    )

    next_ids = set(
        next_fused.keys()
    )

    for row in feature_rows:

        row["target"] = (

            1

            if row["emitter_id"]
            in next_ids

            else 0
        )

        rows.append(row)

    tracker.update(

        [

            type(
                "Emitter",
                (),
                {
                    "emitter_id": eid
                }
            )

            for eid in next_ids
        ]
    )


df = pd.DataFrame(rows)

print(df.head())

print("\nShape:")
print(df.shape)

df.to_csv(

    "results/prediction_dataset_v3.csv",

    index=False
)

print(
    "\nSaved -> results/prediction_dataset_v3.csv"
)