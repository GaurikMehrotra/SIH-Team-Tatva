import heapq


class MLScheduler:

    def __init__(self):
        self.queue = []

    def add_emitter(
        self,
        emitter,
        prediction_probability,
        threat_score
    ):

        priority = (
            prediction_probability
            * threat_score
        )

        heapq.heappush(
            self.queue,
            (
                -priority,
                emitter,
                prediction_probability,
                threat_score,
            )
        )

    def next_scan(self):

        if not self.queue:
            return None

        return heapq.heappop(self.queue)