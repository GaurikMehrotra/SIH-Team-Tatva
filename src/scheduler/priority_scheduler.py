import heapq
from scheduler.threat_scoring import threat_score


class SmartScheduler:

    def __init__(self):
        self.queue = []

    def add_emitter(self, emitter):

        score = threat_score(emitter)

        heapq.heappush(
            self.queue,
            (-score, emitter)
        )

    def next_scan(self):

        if not self.queue:
            return None

        return heapq.heappop(self.queue)[1]
