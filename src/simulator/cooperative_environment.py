import random

from simulator.emitter_generator import (
    generate_emitters
)


class CooperativeEnvironment:

    def __init__(

        self,

        n_emitters=10,

        n_receivers=3

    ):

        self.emitters = generate_emitters(
            n_emitters
        )

        self.n_receivers = n_receivers

        self.time_step = 0

    def is_active(
        self,
        emitter
    ):

        if emitter.emitter_type == "periodic":

            return self.time_step % 3 == 0

        elif emitter.emitter_type == "burst":

            return (
                self.time_step % 10
            ) in [0, 1, 2]

        else:

            return (
                random.random()
                < emitter.activity_probability
            )

    def step(self):

        self.time_step += 1

        observations = {}

        active_emitters = []

        for emitter in self.emitters:

            if self.is_active(emitter):

                active_emitters.append(
                    emitter
                )

        for receiver_id in range(
            self.n_receivers
        ):

            receiver_view = []

            for emitter in active_emitters:

                observed_strength = max(

                    0.0,

                    emitter.signal_strength
                    + random.uniform(
                        -0.2,
                        0.2
                    )
                )

                receiver_view.append({

                    "emitter_id":
                        emitter.emitter_id,

                    "frequency":
                        emitter.frequency,

                    "signal_strength":
                        observed_strength,

                    "type":
                        emitter.emitter_type,
                })

            observations[
                f"Receiver_{receiver_id}"
            ] = receiver_view

        return observations


if __name__ == "__main__":

    env = CooperativeEnvironment(

        n_emitters=10,

        n_receivers=3
    )

    for step in range(5):

        observations = env.step()

        print(
            f"\nTIME STEP {step+1}"
        )

        print("-" * 60)

        for receiver, data in observations.items():

            print(f"\n{receiver}")

            for item in data:

                print(

                    f"{item['emitter_id']:<5}"

                    f"{item['frequency']:>8} MHz "

                    f"Strength="

                    f"{item['signal_strength']:.2f}"
                )