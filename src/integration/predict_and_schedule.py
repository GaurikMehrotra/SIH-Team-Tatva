import joblib
import pandas as pd

from scheduler.threat_scoring import Emitter
from scheduler.priority_scheduler import SmartScheduler


print("Loading model...")

model = joblib.load("models/XGBoost.pkl")

sample = pd.DataFrame([
    {
        "UTCTime": 5000000,
        "RF": 9400,
        "PulseWidth": 0.05,
        "AOA": 40,
        "PA": -90,
        "PA_missing": 0,
    }
])

prediction = model.predict(sample)[0]

confidence = float(
    model.predict_proba(sample).max()
)

print("Prediction:", prediction)
print("Confidence:", confidence)

scheduler = SmartScheduler()

emitter = Emitter(
    emitter_id=f"Emitter_{prediction}",
    confidence=confidence,
    signal_strength=0.80,
    recency=0.90,
)

scheduler.add_emitter(emitter)

next_target = scheduler.next_scan()

print()
print("NEXT TARGET:")
print(next_target.emitter_id)