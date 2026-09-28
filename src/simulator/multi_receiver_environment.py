from simulator.environment import RFEnvironment


class MultiReceiverEnvironment:

    def __init__(
        self,
        n_receivers=3,
        n_emitters=20
    ):

        self.receivers = []

        for _ in range(n_receivers):

            env = RFEnvironment(
                n_emitters
            )

            self.receivers.append(env)

        self.time_step = 0

    def step(self):

        self.time_step += 1

        observations = {}

        for idx, receiver in enumerate(
            self.receivers
        ):

            active_emitters = receiver.step()

            observations[
                f"Receiver_{idx}"
            ] = active_emitters

        return observations


if __name__ == "__main__":

    env = MultiReceiverEnvironment(
        n_receivers=3,
        n_emitters=10
    )

    for step in range(5):

        observations = env.step()

        print()
        print(
            f"TIME STEP {step+1}"
        )

        print("-" * 60)

        for receiver_id, emitters in observations.items():

            print()

            print(receiver_id)

            for emitter in emitters:

                print(
                    f"{emitter.emitter_id:<5}"
                    f"{emitter.frequency:>8} MHz"
                )