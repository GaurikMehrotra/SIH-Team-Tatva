import joblib
import pandas as pd

from simulator.environment import RFEnvironment
from simulator.history_tracker import HistoryTracker

from features.activity_features import extract_features

from scheduler.ml_scheduler import MLScheduler


model = joblib.load(
    "models/LightGBM_activity_v3.pkl"
)

env = RFEnvironment(10)

tracker = HistoryTracker(20)


print("\nREAL TIME AI SCANNING")
print("-" * 60)


for step in range(50):

    active_emitters = env.step()

    tracker.update(active_emitters)

    if step < 6:
        continue

    scheduler = MLScheduler()

    for emitter in env.emitters:

        features = extract_features(
            tracker,
            emitter.emitter_id
        )

        row = pd.DataFrame([{
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

            "avg_signal_strength":
                emitter.signal_strength,

            "num_receivers":
                1
        }])

        probability = model.predict_proba(
            row
        )[0][1]

        threat = emitter.signal_strength

        scheduler.add_emitter(
            emitter.emitter_id,
            probability,
            threat
        )

    best = scheduler.next_scan()

    if best is None:
        continue

    priority = -best[0]
    emitter_id = best[1]
    probability = best[2]
    threat = best[3]

    print(
        f"\nStep {step}"
    )

    print(
        f"Target={emitter_id}"
    )

    print(
        f"Prediction={probability:.3f}"
    )

    print(
        f"Threat={threat:.3f}"
    )

    print(
        f"Priority={priority:.3f}"
    )