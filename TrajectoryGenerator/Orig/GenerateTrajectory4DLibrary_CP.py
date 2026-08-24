# GenerateTrajectory4DLibrary.py
#
# Build a unified 4D trajectory library using:
#  - Trajectory4DPhysics (via Trajectory4DGenerator)
#  - Uniform 1.5 yard cell grid across entire opposite court
#  - Unified intercept-height grid (0.40–3.40 m)
#  - Expanded apex grid including 2.6 & 2.8
#  - True simulated bounce distance for indexing
#
# Output: trajectoryLibrary4d.pkl
#
# -----------------------------------------------------------------------------

import pickle
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

from Trajectory4DGenerator import Trajectory4DGenerator


# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
YARD = 0.9144
CELL_SIZE = 1.5 * YARD
HALF_CELL = CELL_SIZE / 2.0

# Court constants baked into library; consistent with RunSimulation4D
SERVER_BASELINE_Y = 6.4008
NET_Y = 18.288
OPP_BASELINE_Y = 30.1752
SINGLES_LEFT_X = 5.0292
SINGLES_RIGHT_X = 13.2592


# -----------------------------------------------------------------------------
# PARAMETER GRID (canonical)
# These are centerline-based canonical values.
# Full transforms are applied at runtime (Trajectory4DCanonical + TransformLayer4D)
# -----------------------------------------------------------------------------

# Unified intercept Y grid:
# Start 2 cells (3 yards ≈ 2.7432 m ≈ 9 ft) behind the server baseline
interceptYValues = np.arange(
    SERVER_BASELINE_Y - (2 * CELL_SIZE),    # ~3.6576 m
    NET_Y - HALF_CELL,                      # up to 0.75 yd before net
    CELL_SIZE                               
)

# Unified intercept Z grid (formerly launchZ)
interceptZValues = np.array([
    0.30, 0.60, 0.90,
    1.20, 1.50,
    1.80, 2.10, 2.40,
    2.70, 3.00, 3.30
])

# Uniform 1.5-yard bounce grid (opponent half only)
bounceYValues = np.arange(
    NET_Y + HALF_CELL,
    OPP_BASELINE_Y - HALF_CELL,
    CELL_SIZE
)

# Apex heights, including 2.6 and 2.8
apexValues = np.array([
    1.0, 1.3, 1.6, 2.0,
    2.6, 2.7, 2.8,
    3.0, 3.6, 4.5,
    6.0, 8.0, 10.0
])

# Expanded spin sets
topSpins  = np.array([-3000, -1500, 0, 1500, 3000])
sideSpins = np.array([-2000, -1000, 0, 1000, 2000])


# -----------------------------------------------------------------------------
# Build all canonical parameter combinations
# -----------------------------------------------------------------------------
def build_param_space():
    paramList = []
    for interceptY in interceptYValues:
        for interceptZ in interceptZValues:
            for bounceY in bounceYValues:
                for apex in apexValues:
                    for spinTop in topSpins:
                        for spinSide in sideSpins:
                            paramList.append(
                                (interceptY, interceptZ, bounceY, apex, spinTop, spinSide)
                            )
    return paramList


# -----------------------------------------------------------------------------
# Worker function for multiprocessing
# -----------------------------------------------------------------------------
def generateOne(params):
    """
    params = (interceptY, interceptZ, bounceX, bounceY, apex, spinTop, spinSide)
    """

    interceptY, interceptZ, bounceX, bounceY, apex, spinTop, spinSide = params

    # Canonical trajectories use centerline X mapping via bounceX
    interceptPoint = (float(bounceX), float(interceptY), float(interceptZ))
    bouncePoint    = (float(bounceX), float(bounceY))

    generator = Trajectory4DGenerator()
    entry = generator.generateCanonicalEntry(
        interceptPoint=interceptPoint,
        bouncePoint=bouncePoint,
        apexHeight=apex,
        spinTopRpm=spinTop,
        spinSideRpm=spinSide
    )

    if entry is None:
        return None

    # -------------------------------------------------------------------------
    # Compute TRUE bounce distance from simulated fencesX/fencesY
    # -------------------------------------------------------------------------
    bounceIndex = entry["bounceIndex"]
    bxSim = float(entry["fencesX"][bounceIndex])
    bySim = float(entry["fencesY"][bounceIndex])
    x0, y0, _ = entry["interceptPoint"]

    distanceActual = float(np.hypot(bxSim - x0, bySim - y0))
    entry["distance"] = distanceActual

    return entry


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    print("Building unified canonical parameter space...")
    paramSpace = build_param_space()
    total = len(paramSpace)
    print(f"Total combinations: {total}")

    results = []

    print("Starting multiprocessing...")
    with ProcessPoolExecutor(max_workers=None) as pool:
        futures = [pool.submit(generateOne, p) for p in paramSpace]

        for fut in tqdm(as_completed(futures), total=total, desc="Generating"):
            entry = fut.result()
            if entry is not None:
                results.append(entry)

    print(f"\nCompleted. Valid trajectories: {len(results)}")

    # ---------------------------------------------------------------------
    # Bounce-distance gap check (optional but helpful)
    # ---------------------------------------------------------------------
    distances = np.array([e["distance"] for e in results])
    distances.sort()
    gaps = np.diff(distances)
    maxGap = np.max(gaps) if len(gaps) > 1 else 0.0

    print(f"Largest bounce-distance gap: {maxGap:.3f} m")

    # ---------------------------------------------------------------------
    # Save as requested
    # ---------------------------------------------------------------------
    outfile = "trajectoryLibrary4d.pkl"
    with open(outfile, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved unified trajectory library → {outfile}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()