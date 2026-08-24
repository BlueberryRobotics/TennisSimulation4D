import pickle
import numpy as np

INPUT_FILE  = "trajectoryLibrary4d.pkl"

print(f"Loading canonical library from {INPUT_FILE}...")
with open(INPUT_FILE, "rb") as f:
    canonicalLibrary = pickle.load(f)
# Pick the first entry in the canonical library
entry = canonicalLibrary[0]

print("\n=== ENTRY KEYS ===")
print(entry.keys())

def show(name):
    if name in entry:
        arr = entry[name]
        print(f"\n{name}:")
        print("  type:", type(arr))
        try:
            print("  shape:", arr.shape)
        except:
            print("  (no .shape attribute)")
        print("  value:", arr)
    else:
        print(f"\n{name}:  (not present)")

# Show the intercept windows
show("interceptPreDesc")
show("interceptPostAsc")
show("interceptPostDesc")
show("postBounceApex")

# Show trajectory and time arrays
show("trajectory2D")
show("time")

# Show bounce index and distance to bounce
show("bounceIndex")
show("distanceToBounce")