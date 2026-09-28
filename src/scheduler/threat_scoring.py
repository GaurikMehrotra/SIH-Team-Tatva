from dataclasses import dataclass


@dataclass
class Emitter:
    emitter_id: str
    frequency: float
    confidence: float
    signal_strength: float
    recency: float

def threat_score(emitter):

    score = (
        0.4 * emitter.confidence +
        0.3 * emitter.signal_strength +
        0.3 * emitter.recency
    )

    return score