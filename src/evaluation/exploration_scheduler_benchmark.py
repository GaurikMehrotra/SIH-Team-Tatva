from collections import defaultdict
import random

from scheduler.exploration_scheduler import (
    ExplorationScheduler
)

N_STEPS = 5000

hits = 0
misses = 0

last_seen = {}

delays = []

per_emitter_delays = defaultdict(list)

emitters = [

    "E0",
    "E1",
    "E2",
    "E3",
    "E4",
]

for step in range(N_STEPS):

    active_emitters = []

    for emitter in emitters:

        if random.random() < 0.4:

            active_emitters.append(emitter)

    if not active_emitters:
        continue

    scheduler = ExplorationScheduler(
        exploration_weight=0.05
    )

    for emitter in active_emitters:

        probability = random.uniform(
            0.3,
            0.95
        )

        threat = random.uniform(
            0.3,
            1.0
        )

        age = step - last_seen.get(
            emitter,
            0
        )

        scheduler.add_emitter(
            emitter_id=emitter,
            probability=probability,
            threat=threat,
            time_since_scan=age
        )

    item = scheduler.next_scan()

    if item is None:
        continue

    target = item[1]

    if target in active_emitters:

        hits += 1

        delay = (
            step
            - last_seen.get(
                target,
                step
            )
        )

        delays.append(delay)

        per_emitter_delays[
            target
        ].append(delay)

        last_seen[target] = step

    else:

        misses += 1


print("\nEXPLORATION SCHEDULER")
print("-" * 60)

total = hits + misses

print(
    f"Hits   : {hits}"
)

print(
    f"Misses : {misses}"
)

print(
    f"Interception Rate : "
    f"{hits / total:.4f}"
)

print()

print(
    f"Average Delay : "
    f"{sum(delays)/len(delays):.2f}"
)

print(
    f"Maximum Delay : "
    f"{max(delays)}"
)

print(
    f"Median Samples : "
    f"{len(delays)}"
)

print()
print("PER EMITTER")
print("-" * 60)

for emitter, values in sorted(
    per_emitter_delays.items()
):

    print(
        f"{emitter:<10}"
        f"Avg={sum(values)/len(values):.2f}"
        f"   Max={max(values)}"
        f"   Samples={len(values)}"
    )