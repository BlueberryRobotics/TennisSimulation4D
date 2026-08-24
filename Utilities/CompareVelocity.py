import pickle
import numpy as np


# ----------------------------
# CONFIG
# ----------------------------
FILE_PATH = "TrajectoryLibrary.pkl"
THRESHOLD = 18.0  # m/s (~40 mph)


# ----------------------------
# LOAD DATA
# ----------------------------
with open(FILE_PATH, "rb") as f:
    trajs = pickle.load(f)

print(f"Loaded {len(trajs):,} trajectories")

if len(trajs) == 0:
    raise ValueError("Trajectory list is empty")


# ----------------------------
# DETERMINE HOW TO ACCESS VELOCITY
# ----------------------------
sample = trajs[0]

def get_velocity(t):
    # Try object attribute
    if hasattr(t, "initialVelocity"):
        return t.initialVelocity

    # Try dict-like
    if isinstance(t, dict) and "initialVelocity" in t:
        return t["initialVelocity"]

    # Try tuple (you may need to adjust this index)
    if isinstance(t, tuple):
        # Try common index guesses
        # You may need to change this if it fails
        for i in range(len(t)):
            if isinstance(t[i], (float, int)):
                # crude heuristic: velocity is usually a float in a realistic range
                if 0 < t[i] < 100:  
                    return t[i]

    raise ValueError(f"Could not determine velocity for trajectory: {t}")


# ----------------------------
# EXTRACT VELOCITIES
# ----------------------------
velocities = []

for t in trajs:
    try:
        v = get_velocity(t)
        velocities.append(v)
    except Exception as e:
        print("Skipping one trajectory due to error:", e)

velocities = np.array(velocities)

print(f"Valid velocity count: {len(velocities):,}")


# ----------------------------
# BASIC STATS
# ----------------------------
print("\n--- BASIC STATS (m/s) ---")
print(f"Min: {velocities.min():.2f}")
print(f"Max: {velocities.max():.2f}")
print(f"Mean: {velocities.mean():.2f}")
print(f"Median: {np.median(velocities):.2f}")

print("\n--- BASIC STATS (mph) ---")
mph = velocities * 2.237
print(f"Min: {mph.min():.2f}")
print(f"Max: {mph.max():.2f}")
print(f"Mean: {mph.mean():.2f}")
print(f"Median: {np.median(mph):.2f}")


# ----------------------------
# THRESHOLD ANALYSIS
# ----------------------------
slow_mask = velocities <= THRESHOLD
fast_mask = velocities > THRESHOLD

slow_count = np.sum(slow_mask)
fast_count = np.sum(fast_mask)

total = len(velocities)

print("\n--- SPEED BREAKDOWN ---")
print(f"Threshold: {THRESHOLD} m/s (~{THRESHOLD * 2.237:.1f} mph)")
print(f"Slow (<= threshold): {slow_count:,} ({slow_count / total * 100:.2f}%)")
print(f"Fast (> threshold):  {fast_count:,} ({fast_count / total * 100:.2f}%)")


# ----------------------------
# HIGH-SPEED CATEGORIES
# ----------------------------
print("\n--- HIGH SPEED CATEGORIES ---")

def pct(condition):
    c = np.sum(condition)
    return c, (c / total * 100)

for cutoff in [20, 25, 30, 35, 40, 45, 50]:
    count, percent = pct(velocities >= cutoff)
    print(f">= {cutoff:2} m/s ({cutoff * 2.237:.0f} mph): {count:,} ({percent:.2f}%)")


# ----------------------------
# DISTRIBUTION (TEXT HISTOGRAM)
# ----------------------------
print("\n--- DISTRIBUTION (m/s bins) ---")

bins = [0, 10, 15, 18, 20, 25, 30, 40, 60]
counts, edges = np.histogram(velocities, bins=bins)

for i in range(len(counts)):
    low = edges[i]
    high = edges[i + 1]
    count = counts[i]
    percent = count / total * 100
    print(f"{low:5.1f}–{high:5.1f} m/s : {count:8,d} ({percent:6.2f}%)")