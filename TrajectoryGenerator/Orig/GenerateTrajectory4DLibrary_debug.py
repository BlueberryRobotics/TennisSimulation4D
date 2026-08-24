import pickle
import json
import random
import math
import numpy as np
from Trajectory4DGenerator import Trajectory4DGenerator

YARD = 0.9144
CELL_SIZE = 1.5 * YARD
HALF_CELL = CELL_SIZE / 2.0

# Court constants baked into library; consistent with RunSimulation4D
SERVER_BASELINE_Y = 6.4008
NET_Y = 18.288
OPP_BASELINE_Y = 30.1752
SINGLES_LEFT_X = 5.0292
SINGLES_RIGHT_X = 13.2592

# ---------------------------
# the SAME grids you used
# ---------------------------

interceptYValues = np.arange(
    SERVER_BASELINE_Y - (2 * CELL_SIZE),    # ~3.6576 m
    NET_Y - HALF_CELL,                      # up to 0.75 yd before net
    CELL_SIZE                               
)

interceptZValues = [
    0.30, 0.60, 0.90,
    1.20, 1.50,
    1.80, 2.10, 2.40,
    2.70, 3.00, 3.30
]
bounceYValues = np.arange(
    NET_Y + HALF_CELL,
    OPP_BASELINE_Y - HALF_CELL,
    CELL_SIZE
)
apexValues = [
    1.0, 1.3, 1.6, 2.0,
    2.6, 2.7, 2.8,
    3.0, 3.6, 4.5,
    6.0, 8.0, 10.0
]
topSpins = [
    -3000, -1500, 0, 1500, 3000
]
sideSpins = [
    -3000, -1500, 0, 1500, 3000
]

# ---------------------------
# Build param space
# ---------------------------

def build_param_space():
    plist = []
    for iy in interceptYValues:
        for iz in interceptZValues:
            for by in bounceYValues:
                for ah in apexValues:
                    for st in topSpins:
                        for ss in sideSpins:
                            plist.append((iy, iz, by, ah, st, ss))
    return plist


# ---------------------------
# Diagnostic wrapper around generateCanonicalEntry
# ---------------------------

def run_diagnostic_sample(params, sample_index):
    iy, iz, by, ah, st, ss = params

    gen = Trajectory4DGenerator()

    interceptX = 0.0
    interceptPoint = (interceptX, float(iy), float(iz))
    bouncePoint    = (interceptX, float(by))

    print("\n--- Diagnostic Sample", sample_index, "---")
    print("Input Params:")
    print("  interceptZ   =", iz)
    print("  interceptY   =", iy)
    print("  bounceY      =", by)
    print("  apexHeight   =", ah)
    print("  spinTopRPM   =", st)
    print("  spinSideRPM  =", ss)

    entry = gen.generateCanonicalEntry(
        interceptPoint=interceptPoint,
        bouncePoint=bouncePoint,
        apexHeight=ah,
        spinTopRpm=st,
        spinSideRpm=ss
    )

    if entry is None:
        print("  RESULT: None (trajectory generation failed)")
        return None

    # actual physics apex
    bounceIndex = entry["bounceIndex"]
    actualApex  = float(np.max(entry["fencesZ"][:bounceIndex+1]))

    print("Output Entry:")
    print("  actualApex        =", actualApex)
    print("  storedApex        =", entry["apex_height"])
    print("  storedSpinTopRPM  =", entry["spin_top_rpm"])
    print("  storedSpinSideRPM =", entry["spin_side_rpm"])

    # return a JSON-safe diagnostic snapshot
    return {
        "inputInterceptZ": iz,
        "inputInterceptY": iy,
        "inputBounceY": by,
        "inputApexHeight": ah,
        "inputSpinTop": st,
        "inputSpinSide": ss,
        "actualApex": actualApex,
        "storedApex": entry["apex_height"],
        "storedSpinTop": entry["spin_top_rpm"],
        "storedSpinSide": entry["spin_side_rpm"],
        "bounceIndex": entry["bounceIndex"],
        "firstZ": float(entry["fencesZ"][0]),
        "maxZ": float(np.max(entry["fencesZ"])),
        "len": len(entry["fencesZ"])
    }


# ---------------------------
# Run diagnostics
# ---------------------------

def main():
    plist = build_param_space()
    print("Total param combinations:", len(plist))

    # choose 20 random samples
    sample_indices = random.sample(range(len(plist)), 20)

    output = []
    for i, idx in enumerate(sample_indices):
        params = plist[idx]
        diag = run_diagnostic_sample(params, i)
        output.append(diag)

    with open("diagnostic_output.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nDiagnostic report saved")

if __name__ == "__main__":
    main()