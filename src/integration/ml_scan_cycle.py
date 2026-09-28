from simulator.environment import RFEnvironment

from simulator.history_tracker import HistoryTracker

from features.activity_features import extract_features

from models.activity_predictor import ActivityPredictor

from scheduler.ml_scheduler import MLScheduler

from scheduler.threat_scoring import (
    Emitter,
    threat_score
)


env = RFEnvironment(10)

tracker = HistoryTracker(20)

predictor = ActivityPredictor()

hits = 0
misses = 0

N_STEPS = 100


for step in range(N_STEPS):

    active_now = env.step()

    tracker.update(active_now)

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
            recency=1.0
        )

        score = threat_score(
            ew_emitter
        )

        scheduler.add_emitter(
            emitter.emitter_id,
            probability,
            score
        )

    target = scheduler.next_scan()

    chosen_emitter = target[1]

    active_next = env.step()

    next_ids = {

        e.emitter_id

        for e in active_next
    }

    if chosen_emitter in next_ids:

        hits += 1

    else:

        misses += 1

    tracker.update(active_next)


print("\nML SCHEDULER RESULTS")
print("-" * 50)

print("Hits   :", hits)
print("Misses :", misses)

total = hits + misses

if total > 0:

    print(
        "Interception Rate:",
        round(hits / total, 4)
    )