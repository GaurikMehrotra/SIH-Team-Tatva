from simulator.history_tracker import HistoryTracker


def extract_features(tracker, emitter_id):

    history = tracker.get_history(emitter_id)

    last_5 = history[-5:]

    while len(last_5) < 5:
        last_5.insert(0, 0)

    features = {

        "activity_ratio":
            tracker.activity_ratio(emitter_id),

        "consecutive_activity":
            tracker.consecutive_activity(emitter_id),

        "time_since_last_seen":
            tracker.time_since_last_seen(emitter_id),

        "last_5_activity":
            last_5

    }

    return features


if __name__ == "__main__":

    class Dummy:

        def __init__(self, emitter_id):

            self.emitter_id = emitter_id


    tracker = HistoryTracker(20)

    tracker.update([Dummy("E1")])
    tracker.update([Dummy("E1")])
    tracker.update([])
    tracker.update([Dummy("E1")])
    tracker.update([])
    tracker.update([Dummy("E1")])

    features = extract_features(
        tracker,
        "E1"
    )

    print(features)