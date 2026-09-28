from simulator.environment import RFEnvironment
from scheduler.priority_scheduler import SmartScheduler
from scheduler.threat_scoring import Emitter


env = RFEnvironment(10)

hits = 0
misses = 0

N_STEPS = 100


for step in range(N_STEPS):

    scheduler = SmartScheduler()

    active_emitters = env.step()

    if not active_emitters:
        continue

    for emitter in active_emitters:

        scheduler.add_emitter(

            Emitter(
                emitter_id=emitter.emitter_id,
                frequency=emitter.frequency,
                confidence=emitter.activity_probability,
                signal_strength=emitter.signal_strength,
                recency=1.0
            )
        )

    target = scheduler.next_scan()

    active_ids = {
        e.emitter_id
        for e in active_emitters
    }

    if target.emitter_id in active_ids:

        hits += 1

    else:

        misses += 1


print("\nRESULTS")
print("-" * 40)

print("Hits   :", hits)
print("Misses :", misses)

total = hits + misses

if total > 0:

    interception_rate = hits / total

    print(
        f"Interception Rate: "
        f"{interception_rate:.4f}"
    )