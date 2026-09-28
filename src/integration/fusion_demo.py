from simulator.cooperative_environment import (
    CooperativeEnvironment
)

from simulator.fusion_engine import (
    FusionEngine
)


env = CooperativeEnvironment(

    n_emitters=10,

    n_receivers=3
)

fusion = FusionEngine()


observations = env.step()

fused = fusion.fuse(
    observations
)

print("\nFUSED EMITTER TABLE")
print("-" * 60)

for emitter_id, data in fused.items():

    print(

        f"{emitter_id:<6}"

        f"{data['frequency']:>8} MHz "

        f"Receivers="

        f"{data['num_receivers']} "

        f"Strength="

        f"{data['avg_strength']:.2f}"
    )