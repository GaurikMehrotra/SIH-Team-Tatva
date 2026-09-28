import heapq


class ExplorationScheduler:

    def __init__(
        self,
        exploration_weight=0.05
    ):

        self.queue = []

        self.exploration_weight = (
            exploration_weight
        )

    def add_emitter(
        self,
        emitter_id,
        probability,
        threat,
        time_since_scan,
    ):

        exploitation_score = (
            probability * threat
        )

        exploration_bonus = (
            self.exploration_weight
            * time_since_scan
        )

        priority = (
            exploitation_score
            + exploration_bonus
        )

        heapq.heappush(
            self.queue,
            (
                -priority,
                emitter_id,
                probability,
                threat,
                time_since_scan,
            )
        )

    def next_scan(self):

        if not self.queue:
            return None

        return heapq.heappop(
            self.queue
        )