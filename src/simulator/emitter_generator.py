from dataclasses import dataclass
import random


@dataclass
class SimulatedEmitter:

    emitter_id: str

    frequency: float

    activity_probability: float

    signal_strength: float

    emitter_type: str


def generate_emitters(n=10):

    emitters = []

    emitter_types = [
        "periodic",
        "burst",
        "hopper",
        "random"
    ]

    for i in range(n):

        emitter = SimulatedEmitter(

            emitter_id=f"E{i}",

            frequency=random.randint(
                100,
                18000
            ),

            activity_probability=random.uniform(
                0.1,
                0.95
            ),

            signal_strength=random.uniform(
                0.2,
                1.0
            ),

            emitter_type=random.choice(
                emitter_types
            )
        )

        emitters.append(emitter)

    return emitters


if __name__ == "__main__":

    emitters = generate_emitters(10)

    print("\nGenerated Emitters\n")
    print("-" * 60)

    for e in emitters:

        print(
            f"{e.emitter_id:<5} "
            f"Freq={e.frequency:<6} "
            f"Prob={e.activity_probability:.2f} "
            f"Power={e.signal_strength:.2f} "
            f"Type={e.emitter_type}"
        )