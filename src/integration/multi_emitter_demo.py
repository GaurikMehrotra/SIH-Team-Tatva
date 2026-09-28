from scheduler.threat_scoring import Emitter, threat_score
from scheduler.priority_scheduler import SmartScheduler


scheduler = SmartScheduler()

emitters = [

    Emitter(
        "Emitter_20",
        9400,
        0.90,
        0.80,
        0.90
    ),

    Emitter(
        "Emitter_42",
        5600,
        0.70,
        0.95,
        0.95
    ),

    Emitter(
        "Emitter_15",
        425,
        0.60,
        0.60,
        0.70
    ),

    Emitter(
        "Emitter_08",
        1200,
        0.40,
        0.50,
        0.50
    ),

    Emitter(
        "Emitter_99",
        2800,
        0.95,
        0.40,
        0.60
    ),

]

print("\nThreat Scores\n")
print("-" * 60)

for emitter in emitters:

    score = threat_score(emitter)

    print(
        f"{emitter.emitter_id:<12} "
        f"Freq={emitter.frequency:<6} MHz "
        f"Score={score:.4f}"
    )

    scheduler.add_emitter(emitter)

print("\nScan Order\n")
print("-" * 60)

rank = 1

while True:

    target = scheduler.next_scan()

    if target is None:
        break

    print(
        f"{rank}. "
        f"{target.emitter_id} "
        f"@ {target.frequency} MHz"
    )

    rank += 1