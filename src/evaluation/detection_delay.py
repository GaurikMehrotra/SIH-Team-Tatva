from simulator.environment import RFEnvironment

from simulator.history_tracker import HistoryTracker
from features.activity_features import extract_features

from models.activity_predictor import ActivityPredictor
from scheduler.ml_scheduler import MLScheduler
from scheduler.threat_scoring import (
    Emitter,
    threat_score,
)


N_STEPS = 1000


env = RFEnvironment(10)

tracker = HistoryTracker(20)

predictor = ActivityPredictor()

delays = []

waiting_emitters = {}


for step in range(N_STEPS):

    active_now = env.step()

    tracker.update(active_now)

    active_ids = {
        e.emitter_id
        for e in active_now
    }

    # Record first appearance
    for emitter_id in active_ids:

        if emitter_id not in waiting_emitters:

            waiting_emitters[emitter_id] = step

    scheduler = MLScheduler()

    for emitter in env.emitters:

        features = extract_features(
            tracker,
            emitter.emitter_id
        )

        model_features = {

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

        probability = predictor.predict_probability(
            model_features
        )

        ew_emitter = Emitter(
            emitter_id=emitter.emitter_id,
            frequency=emitter.frequency,
            confidence=emitter.activity_probability,
            signal_strength=emitter.signal_strength,
            recency=1.0,
        )

        threat = threat_score(
            ew_emitter
        )

        scheduler.add_emitter(
            emitter.emitter_id,
            probability,
            threat,
        )

    target = scheduler.next_scan()

    chosen_id = target[1]

    if chosen_id in waiting_emitters:

        delay = (
            step
            - waiting_emitters[chosen_id]
        )

        delays.append(delay)

        del waiting_emitters[
            chosen_id
        ]


print("\nDETECTION DELAY")
print("-" * 50)

if delays:

    avg_delay = sum(delays) / len(delays)

    print(
        f"Average Delay : "
        f"{avg_delay:.2f}"
    )

    print(
        f"Minimum Delay : "
        f"{min(delays)}"
    )

    print(
        f"Maximum Delay : "
        f"{max(delays)}"
    )

    delays_sorted = sorted(delays)

    median_delay = delays_sorted[
        len(delays_sorted)//2
    ]

    print(
        f"Median Delay  : "
        f"{median_delay}"
    )

else:

    print("No detections recorded.")