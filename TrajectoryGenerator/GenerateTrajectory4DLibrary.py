import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

from Trajectory4DGenerator import Trajectory4DGenerator

# ------------------------------------------
# PARAMETER GRIDS
# ------------------------------------------

interceptYValues = np.array([0.0])

# the bounce distances, 2 to 36 meters in half meter increments
bounceYValues = np.arange(2, 36, 0.5)

#--------------------------------------------------------------------------
# You cannot change any of the InterceptZValues, apexValues and topSpins
# unless you also change the same values in the Simulation and rerun
#--------------------------------------------------------------------------

# note that interceptZValues must have equivalent apexValues 
# for 1m and higher to enable higher velocity shots
interceptZValues = np.array([
    0.30, 0.60, 1.00, 1.25, 1.50, 1.80,
    2.10, 2.40, 2.70, 3.00, 3.30
])

apexValues = np.array([
    1.00, 1.25, 1.50, 1.80, 2.10, 2.40, 2.70,
    3.00, 3.30, 3.60, 4.50, 6.00, 8.00, 10.00
])

topSpins = np.array([-3000, -1500, 0, 1500, 2500, 3500])
sideSpins = np.array([-2000, -1000, 0, 1000, 2000])

velocityArrayMph = [70, 85, 100, 115, 130]

# ------------------------------------------
# BUILD PARAM SPACE
# ------------------------------------------

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


# ------------------------------------------
# WORKER
# ------------------------------------------

def generateOne(params):
    y0, z0, yb, H, st, ss = params

    gen = Trajectory4DGenerator(
        velocityArrayMph=velocityArrayMph
    )

    interceptPoint = (0.0, 0.0, z0)
    bouncePoint = (0.0, yb)

    return gen.generateCanonicalEntry(
        interceptPoint,
        bouncePoint,
        H,
        st,
        ss,
        apexValues
    )


# ------------------------------------------
# MAIN
# ------------------------------------------

def main():
    params = build_param_space()
    library = []

    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(generateOne, p) for p in params]
        for f in tqdm(as_completed(futures), total=len(futures)):
            entries = f.result()
            if entries:
                library.extend(entries)

    with open("Trajectory4DLibrary007.pkl", "wb") as f:
        pickle.dump(library, f)

    print(f"Generated {len(library)} trajectories.")


if __name__ == "__main__":
    main()

# change output file above 
# python GenerateTrajectory4DLibrary.py