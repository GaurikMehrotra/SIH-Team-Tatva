from scheduler.exploration_scheduler import (
    ExplorationScheduler
)


emitters = [

    ("E20", 0.92, 0.85, 1),

    ("E42", 0.40, 0.95, 15),

    ("E15", 0.75, 0.70, 3),

    ("E08", 0.25, 0.50, 25),

    ("E99", 0.88, 0.60, 2),
]


weights = [

    0.01,
    0.05,
    0.10,
]


for weight in weights:

    print("\n")
    print("=" * 60)

    print(
        f"Exploration Weight = {weight}"
    )

    print("=" * 60)

    scheduler = ExplorationScheduler(
        exploration_weight=weight
    )

    for emitter in emitters:

        scheduler.add_emitter(
            emitter_id=emitter[0],
            probability=emitter[1],
            threat=emitter[2],
            time_since_scan=emitter[3],
        )

    rank = 1

    while True:

        item = scheduler.next_scan()

        if item is None:
            break

        priority = -item[0]

        print(
            f"{rank}. "
            f"{item[1]} "
            f"P={item[2]:.2f} "
            f"T={item[3]:.2f} "
            f"Age={item[4]} "
            f"Priority={priority:.4f}"
        )

        rank += 1