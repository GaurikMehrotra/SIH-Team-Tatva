from collections import defaultdict


class FusionEngine:

    def fuse(self, observations):

        fused = {}

        emitter_data = defaultdict(list)

        for receiver in observations.values():

            for obs in receiver:

                emitter_data[
                    obs["emitter_id"]
                ].append(obs)

        for emitter_id, reports in emitter_data.items():

            avg_strength = sum(
                r["signal_strength"]
                for r in reports
            ) / len(reports)

            frequency = reports[0]["frequency"]

            emitter_type = reports[0]["type"]

            fused[emitter_id] = {

                "frequency": frequency,

                "type": emitter_type,

                "num_receivers": len(reports),

                "avg_strength": avg_strength,
            }

        return fused