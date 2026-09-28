from scheduler.ml_scheduler import MLScheduler


scheduler = MLScheduler()


emitters = [

    ("E20", 0.92, 0.85),

    ("E42", 0.40, 0.95),

    ("E15", 0.75, 0.70),

    ("E08", 0.25, 0.50),

    ("E99", 0.88, 0.60),
]


for emitter_id, probability, threat in emitters:

    scheduler.add_emitter(
        emitter_id,
        probability,
        threat
    )


print("\nML Scheduler Ranking")
print("-" * 50)


rank = 1

while True:

    item = scheduler.next_scan()

    if item is None:
        break

    priority = -item[0]
    emitter = item[1]
    probability = item[2]
    threat = item[3]

    print(
        f"{rank}. "
        f"{emitter} "
        f"P={probability:.2f} "
        f"Threat={threat:.2f} "
        f"Priority={priority:.4f}"
    )

    rank += 1