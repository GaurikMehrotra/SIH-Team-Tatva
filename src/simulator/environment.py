from simulator.emitter_generator import generate_emitters

import random


class RFEnvironment:

    def __init__(self, n_emitters=10):

        self.emitters = generate_emitters(n_emitters)

        self.time_step = 0

    def is_active(self, emitter):

        # PERIODIC EMITTER

        if emitter.emitter_type == "periodic":

            period = 3

            return self.time_step % period == 0

        # BURST EMITTER

        elif emitter.emitter_type == "burst":

            cycle = self.time_step % 10

            return cycle in [0, 1, 2]

        # RANDOM EMITTER

        elif emitter.emitter_type == "random":

            return (
                random.random()
                < emitter.activity_probability
            )

        # HOPPER

        elif emitter.emitter_type == "hopper":

            return (
                random.random()
                < emitter.activity_probability
            )

        return False

    def update_frequency(self, emitter):

        if emitter.emitter_type == "hopper":

            emitter.frequency += random.randint(
                -300,
                300
            )

            emitter.frequency = max(
                100,
                emitter.frequency
            )

    def get_active_emitters(self):

        active_emitters = []

        for emitter in self.emitters:

            self.update_frequency(emitter)

            if self.is_active(emitter):

                active_emitters.append(emitter)

        return active_emitters

    def step(self):

        self.time_step += 1

        return self.get_active_emitters()


if __name__ == "__main__":

    env = RFEnvironment(10)

    for t in range(20):

        active = env.step()

        print(f"\nTime Step {t+1}")
        print("-" * 40)

        for emitter in active:

            print(
                f"{emitter.emitter_id:<5} "
                f"{emitter.emitter_type:<10} "
                f"{emitter.frequency} MHz"
            )