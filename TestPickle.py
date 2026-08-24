# import pickle

# with open("CanonicalTraj2DData003.pkl", "rb") as f:
#     lib = pickle.load(f)

# print("Keys in first entry:")
# print(lib[0].keys())

import pickle
import numpy as np

with open("trajectoryLibrary004.pkl", "rb") as f:
    lib = pickle.load(f)

entry = lib[150000]
print("LIB LEN:", len(entry["canonX"]), "bIdx:", entry["bounceIndex"])

print(type(lib))
print(len(lib))
print(type(entry))
print(entry.keys())
print(entry.values())


#print("trajectory2D shape:", np.array(entry["trajectory2D"]).shape)
#print("trajectoryZ exists:", "trajectoryZ" in entry)