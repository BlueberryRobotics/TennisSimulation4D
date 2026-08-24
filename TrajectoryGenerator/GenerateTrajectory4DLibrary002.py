import pickle
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
# from ..CourtSettings import Court
from Trajectory4DGenerator002 import Trajectory4DGenerator
import importlib, Trajectory4DGenerator
importlib.reload(Trajectory4DGenerator)

# court = Court()

# -------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------
# YARD = 0.9144
# CELL_SIZE = 1.5 * YARD
# HALF_CELL = CELL_SIZE / 2.0

# SERVER_BASELINE_Y = court.serverBaselineY
# NET_Y             = court.netY
# OPP_BASELINE_Y    = court.opponentBaselineY
# SINGLES_LEFT_X    = court.singlesLeftX
# SINGLES_RIGHT_X   = court.singlesRightX

# -------------------------------------------------------------------
# PARAMETER GRIDS (canonical)
# -------------------------------------------------------------------

# These values control *relative* forward distance only.
interceptYValues = np.array([0])

# FOR REAL ***************************************

# note that interceptZValues must have equivalent apexValues 
# for 1m and higher to enable higher velocity shots
interceptZValues = np.array([
    0.30, 0.60, 1.00, 1.25, 1.50, 1.80,
    2.10, 2.40, 2.70, 3.00, 3.30
])

bounceYValues = np.arange(2, 36, 0.5)

apexValues = np.array([
    1.00, 1.25, 1.50, 1.80, 2.10, 2.40, 2.70,
    3.00, 3.30, 3.60, 4.50, 6.00, 8.00, 10.00
])

topSpins = np.array([-3000, -1500, 0, 1500, 2500, 3500])
sideSpins = np.array([-2000, -1000, 0, 1000, 2000])

# FOR LIMITED TESTING *********************************************

# bounceYValues = np.arange(2,4,1)

# interceptZValues = np.array([
#     2.70
# ])

# apexValues = np.array([
#  2.7, 2.8
# ])

# topSpins = np.array([1500])
# sideSpins = np.array([-1500])

# -------------------------------------------------------------------
# BUILD PARAMETER SPACE
# -------------------------------------------------------------------
def build_param_space():
    params = []
    for y0 in interceptYValues:
        for z0 in interceptZValues:
            for yb in bounceYValues:
                for H in apexValues:
                    for st in topSpins:
                        for ss in sideSpins:
                            params.append((y0, z0, yb, H, st, ss))
    return params


# -------------------------------------------------------------------
# WORKER
# -------------------------------------------------------------------
def generateOne(params):
    """
    params is: (interceptY, interceptZ, bounceY, apexH, spinTop, spinSide)
    BUT canonical intercept must be (0,0,z)
    and canonical bounce must be relative forward distance.
    """
    from Trajectory4DGenerator002 import Trajectory4DGenerator

    y0, z0, yb, apexH, st, ss = params

    # canonical intercept ALWAYS at (0,0,z)
    canonicalInterceptPoint = (0.0, 0.0, float(z0))

    # canonical bounce ALWAYS at (0, bounceY - interceptY)
    canonicalForwardY = float(yb - y0)
    canonicalBouncePoint = (0.0, canonicalForwardY)

    # print("Forward Y: " + str(canonicalForwardY) + "Bounce Point: " + str(canonicalBouncePoint))

    gen = Trajectory4DGenerator()
    entry = gen.generateCanonicalEntry(
        interceptPoint=canonicalInterceptPoint,
        bouncePoint=canonicalBouncePoint,
        apexHeight=apexH,
        spinTopRpm=st,
        spinSideRpm=ss,
        apexValues=apexValues  
    )
    # "interceptPoint": (x0, y0, z0),
    # "bouncePoint": (xb, yb),
    # "distance": canonicalDistance,
    # "apex_height": self._snapApexToBins(actualApex, apexHeights),

    # print("Trajectory Y: " + str(entry["interceptPoint"]))

    return entry  # may be None



# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    print("Building canonical parameter space...")
    params = build_param_space()
    total = len(params)
    print(f"Total canonical combinations: {total}")

    results = []

    print("Starting multiprocessing...")
    with ProcessPoolExecutor(max_workers=None) as pool:
        futures = [pool.submit(generateOne, p) for p in params]

        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as e:
                print("Worker failed:", e)
                continue

            if r is None:
                continue

            results.append(r)

    for i, entry in enumerate(results):
        entry["id"] = i

    print(f"\nCompleted. Valid trajectories: {len(results)}")

    outfile = "trajectoryLibrary005.pkl"
    with open(outfile, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved canonical library to {outfile}")


if __name__ == "__main__":
    main()