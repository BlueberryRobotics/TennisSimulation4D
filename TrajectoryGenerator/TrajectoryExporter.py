# EXPORTS TRAJECTORIES TO JSON

import json
import pickle
from pathlib import Path
import numpy as np

INPUT_PKL = Path("Trajectory4DLibrary005.pkl")
OUTPUT_JSON = Path("Trajectory4DLibrary005.sample.json")
MAX_ROWS = 1000  # adjust

def to_list(x):
    if isinstance(x, np.ndarray):
        return x.astype(float).tolist()
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return x

with INPUT_PKL.open("rb") as f:
    data = pickle.load(f)

entries = list(data.values()) if isinstance(data, dict) else list(data)
entries = entries[:MAX_ROWS]

out = []
for e in entries:
    out.append({
        "interceptPoint": to_list(e["interceptPoint"]),
        "bouncePoint": to_list(e["bouncePoint"]),
        "distance": float(e["distance"]),
        "apex_height": float(e["apex_height"]),
        "spin_top_rpm": int(e["spin_top_rpm"]),
        "spin_side_rpm": int(e["spin_side_rpm"]),
        "canonX": to_list(e["canonX"]),
        "canonY": to_list(e["canonY"]),
        "canonZ": to_list(e["canonZ"]),
        "time": to_list(e["time"]),
        "bounceIndex": int(e["bounceIndex"]),
        "initialVelocity": float(e.get("initialVelocity", 0.0)),
        "airTravelDistance": float(e.get("airTravelDistance", 0.0)),
        "landingY": float(e.get("landingY", to_list(e["canonY"])[-1])),
        "solve_mode": e.get("solve_mode"),
        "speed_mph": float(e.get("speed_mph", 0.0)),
    })

with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(out, f)

print(f"Wrote {len(out)} sample rows to {OUTPUT_JSON}")