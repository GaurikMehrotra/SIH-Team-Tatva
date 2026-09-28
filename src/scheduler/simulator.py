from threat_scoring import Emitter
from priority_scheduler import SmartScheduler


scheduler = SmartScheduler()

scheduler.add_emitter(
    Emitter(
        "Radar_A",
        confidence=0.95,
        signal_strength=0.90,
        recency=0.85
    )
)

scheduler.add_emitter(
    Emitter(
        "Radar_B",
        confidence=0.60,
        signal_strength=0.40,
        recency=0.50
    )
)

scheduler.add_emitter(
    Emitter(
        "Radar_C",
        confidence=0.80,
        signal_strength=0.70,
        recency=0.95
    )
)

print("\nScan Order:\n")

while True:

    target = scheduler.next_scan()

    if target is None:
        break

    print(target.emitter_id)