from collections import defaultdict, deque


class HistoryTracker:

    def __init__(self, window_size=20):

        self.window_size = window_size

        self.history = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def update(self, active_emitters):

        active_ids = {
            emitter.emitter_id
            for emitter in active_emitters
        }

        known_emitters = set(
            self.history.keys()
        ).union(active_ids)

        for emitter_id in known_emitters:

            if emitter_id in active_ids:

                self.history[emitter_id].append(1)

            else:

                self.history[emitter_id].append(0)

    def get_history(self, emitter_id):

        return list(
            self.history[emitter_id]
        )

    def activity_ratio(self, emitter_id):

        h = self.get_history(emitter_id)

        if len(h) == 0:
            return 0

        return sum(h) / len(h)

    def consecutive_activity(self, emitter_id):

        h = self.get_history(emitter_id)

        count = 0

        for x in reversed(h):

            if x == 1:
                count += 1
            else:
                break

        return count

    def time_since_last_seen(self, emitter_id):

        h = self.get_history(emitter_id)

        for i, x in enumerate(reversed(h)):

            if x == 1:
                return i

        return len(h)


if __name__ == "__main__":

    class Dummy:

        def __init__(self, emitter_id):

            self.emitter_id = emitter_id


    tracker = HistoryTracker(10)

    tracker.update(
        [
            Dummy("E1"),
            Dummy("E2")
        ]
    )

    tracker.update(
        [
            Dummy("E1")
        ]
    )

    tracker.update(
        [
            Dummy("E2")
        ]
    )

    print(
        tracker.get_history("E1")
    )

    print(
        tracker.activity_ratio("E1")
    )

    print(
        tracker.consecutive_activity("E1")
    )

    print(
        tracker.time_since_last_seen("E1")
    )