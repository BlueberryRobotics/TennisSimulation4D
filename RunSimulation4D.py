# RunSimulation4D.py
"""
Grid-targeted simulation runner (4D).

Supports:
 - Dynamic physics-based trajectory generation (Trajectory4DDynamicGenerator)
 - Canonical trajectory selection (Trajectory4DCanonical + canonical4D.pkl)

RunSimulation4D is the top-level orchestration layer where
a user selects which trajectory generator to use.

Reference file selection behavior:
    1) Run from existing reference file(s):
         - By default, this script uses the latest GenXReference.parquet in IterativeSimulation.
         - Example (visualizations enabled by default):
                 python RunSimulation4D.py --numberOfPoints 20 --seed 123
         - If IterativeSimulation/Gen6Reference.parquet is the latest file, it will be used.

    2) Build a new reference from a consolidated parquet first:
         - Provide --consolidatedInputPath to build a reference before simulation starts.
         - Optional: provide --referenceOutputPath to control the generated reference file path.
         - Example:
                 python RunSimulation4D.py --consolidatedInputPath IterativeSimulation/ConsolidatedGen6.parquet --referenceOutputPath IterativeSimulation/Gen6Reference.parquet --numberOfPoints 20 --seed 123

The selected reference path is always printed at startup as:
    [INIT] Using reference selector file: <path>

Geometry behavior:
    - This runner now always uses short row geometry (rows 13/14 are 3 ft).
"""
import time
import sys
import csv
import os, shutil
import pickle
import re
import argparse
import random
import numpy as np
from CourtPlayerSettings import Court

# ---------------------------------------------------------
# Ensure local packages importable
# ---------------------------------------------------------
sys.path.append("Trajectory4D")

from Trajectory4D.Trajectory4DDynamicGenerator import Trajectory4DDynamicGenerator
from Trajectory4D.Trajectory4DCanonical import Trajectory4DCanonical
from Trajectory4D.TransformLayer import TransformLayer
from Trajectory4D.Visualizer import SaveTrajectoryPlot
from Trajectory4D.SelectivePressure import SelectivePressure

from Trajectory4D.ServeShotRunner import ServeShotRunner as ServeShotRunner4D
from Trajectory4D.RallyShotRunner import RallyShotRunner as RallyShotRunner4D

from Trajectory4D.PointRunner import PointRunner
from Trajectory4D.PlayerMovement import PlayerMovement
from Trajectory4D.ShotValueTracker import ShotValueTracker


VISUALIZE = True  # Set False for large runs


def PrintCourtGeometryCheck(court: Court):
    print("[GEOM] Geometry check start")
    print(
        "[GEOM] "
        f"mode={getattr(court, 'geometryMode', 'uniform')} "
        f"gridRows={court.gridRows} "
        f"lengthFence_m={court.lengthFence:.4f}"
    )

    keyBoundaries = [4, 8, 13, 18, 22]
    for boundaryIndex in keyBoundaries:
        boundaryY = court.RowBoundaryY(boundaryIndex) if hasattr(court, "RowBoundaryY") else (boundaryIndex * court.granularity)
        print(f"[GEOM] boundary {boundaryIndex}/{boundaryIndex + 1}: y={boundaryY:.4f}")

    print(
        "[GEOM] lines "
        f"serverBaselineY={court.serverBaselineY:.4f} "
        f"serviceLineY={court.serviceLineY:.4f} "
        f"netY={court.netY:.4f} "
        f"opponentServiceLineY={court.opponentServiceLineY:.4f} "
        f"receiverBaselineY={court.receiverBaselineY:.4f}"
    )

    sampleRows = [12, 13, 14, 15, 22, 23]
    for row in sampleRows:
        centerY = court.GetRowCenterY(row) if hasattr(court, "GetRowCenterY") else ((row - 0.5) * court.granularity)
        rowHeight = court.GetRowHeight(row) if hasattr(court, "GetRowHeight") else court.granularity
        print(f"[GEOM] row={row:02d} centerY={centerY:.4f} rowHeight={rowHeight:.4f}")

    print("[GEOM] Geometry check end")


def FindLatestReferenceFile(iterativeDirectory: str) -> str:
    latestGeneration = -1
    latestReferencePath = ""

    if not os.path.isdir(iterativeDirectory):
        return latestReferencePath

    for name in os.listdir(iterativeDirectory):
        match = re.match(r"^Gen(\d+)Reference\.parquet$", name)
        if not match:
            continue
        generation = int(match.group(1))
        if generation > latestGeneration:
            latestGeneration = generation
            latestReferencePath = os.path.join(iterativeDirectory, name)

    return latestReferencePath


def ParseCommandLineArguments():
    usageExamples = (
        "Examples:\n"
        "  Run using latest existing GenXReference.parquet (default behavior):\n"
        "    python RunSimulation4D.py --numberOfPoints 20 --seed 123\n"
        "\n"
        "  Build a reference from consolidated parquet and run:\n"
        "    python RunSimulation4D.py --consolidatedInputPath IterativeSimulation/ConsolidatedGen6.parquet --referenceOutputPath IterativeSimulation/Gen6Reference.parquet --numberOfPoints 20 --seed 123\n"
        "\n"
        "  Disable visualization output:\n"
        "    python RunSimulation4D.py --no-visualize --numberOfPoints 100"
    )

    parser = argparse.ArgumentParser(
        description=(
            "RunSimulation4D selector setup. By default uses latest GenXReference.parquet. "
            "Optionally build a reference from a consolidated parquet first."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=usageExamples,
    )
    parser.add_argument(
        "--consolidatedInputPath",
        type=str,
        default=None,
        help=(
            "Optional consolidated parquet path used to build a reference before running."
        ),
    )
    parser.add_argument(
        "--referenceOutputPath",
        type=str,
        default=None,
        help=(
            "Optional explicit output path for built reference parquet. "
            "If omitted, defaults to IterativeSimulation/Gen<parsed>Reference.parquet or "
            "IterativeSimulation/ManualReference.parquet when generation cannot be parsed."
        ),
    )
    parser.add_argument(
        "--printCourtGeometryCheck",
        action="store_true",
        default=os.environ.get("SIM_PRINT_COURT_GEOMETRY_CHECK", "0") == "1",
        help="Print startup geometry boundary/row-center diagnostics.",
    )
    parser.add_argument(
        "--numberOfPoints",
        type=int,
        default=10,
        help="Number of points to simulate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible A/B validation.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=os.environ.get("SIM_VISUALIZE", "1") == "1",
        help="Enable per-shot visualization output.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Disable per-shot visualization output.",
    )
    return parser.parse_args()


def ParseConsolidatedGenerationFromPath(consolidatedPath: str):
    fileName = os.path.basename(consolidatedPath)
    match = re.match(r"^ConsolidatedGen(\d+)(?:_.*)?\.parquet$", fileName)
    if not match:
        return None
    return int(match.group(1))


def ResolveReferenceSelectorPath(commandLineArgs) -> str:
    consolidatedInputPath = commandLineArgs.consolidatedInputPath
    requestedReferenceOutputPath = commandLineArgs.referenceOutputPath

    if consolidatedInputPath:
        if not os.path.exists(consolidatedInputPath):
            raise FileNotFoundError(
                f"Consolidated input file not found: {consolidatedInputPath}"
            )

        if requestedReferenceOutputPath:
            resolvedReferenceOutputPath = requestedReferenceOutputPath
        else:
            parsedGeneration = ParseConsolidatedGenerationFromPath(consolidatedInputPath)
            if parsedGeneration is None:
                resolvedReferenceOutputPath = os.path.join(
                    "IterativeSimulation",
                    "ManualReference.parquet",
                )
            else:
                resolvedReferenceOutputPath = os.path.join(
                    "IterativeSimulation",
                    f"Gen{parsedGeneration}Reference.parquet",
                )

        print(
            "[INIT] Building reference from consolidated input: "
            f"{consolidatedInputPath} -> {resolvedReferenceOutputPath}"
        )
        SelectivePressure.BuildReferenceFile(
            sourceParquetPath=consolidatedInputPath,
            outputParquetPath=resolvedReferenceOutputPath,
        )

        return resolvedReferenceOutputPath

    latestReferencePath = FindLatestReferenceFile("IterativeSimulation")
    if not latestReferencePath:
        raise FileNotFoundError(
            "No GenXReference.parquet found in IterativeSimulation. "
            "Provide --consolidatedInputPath to build one, or run RunSImulation4DMP.py first."
        )
    return latestReferencePath

commandLineArgs = ParseCommandLineArguments()
court = Court(geometryMode="short_rows_13_14")
referenceSelectorPath = ResolveReferenceSelectorPath(commandLineArgs)

print(f"[INIT] Using reference selector file: {referenceSelectorPath}")
print("[INIT] Court geometry mode: short_rows_13_14")
print(f"[INIT] Visualize: {commandLineArgs.visualize}")
print(f"[INIT] Number of points: {commandLineArgs.numberOfPoints}")

if commandLineArgs.seed is not None:
    seedValue = int(commandLineArgs.seed)
    np.random.seed(seedValue)
    random.seed(seedValue)
    print(f"[INIT] Random seed: {commandLineArgs.seed}")

if commandLineArgs.printCourtGeometryCheck:
    PrintCourtGeometryCheck(court)
trajecticsSelector = SelectivePressure(
    referenceSelectorPath,
    0.5,
    3,
    court=court,
    isPreFilteredReference=True,
    debug=False,
    debugLevel1=True,
)

# ---------------------------------------------------------
# 4) Choose trajectory generator mode
# ---------------------------------------------------------
USE_CANONICAL = True
CANONICAL_PATH = "Trajectory4DLibrary007.pkl"

trajectoryLibrary = None
canonicalSpec = None

#--------------------------------------------------------------------------
# You cannot change any of the InterceptZValues, apexValues and topSpins
# unless you also change the same values in the Trajectory Generator and rerun
#--------------------------------------------------------------------------

interceptZValues = [
    0.30, 0.60, 1.00, 1.25, 1.50, 1.80,
    2.10, 2.40, 2.70, 3.00, 3.30
]

spinTopValues = np.array([-3000, -1500, 0, 1500, 2500, 3500])
spinSideValues = np.array([-2000, -1000, 0, 1000, 2000])

apexValues = [
    1.00, 1.25, 1.50, 1.80, 2.10, 2.40, 2.70,
    3.00, 3.30, 3.60, 4.50, 6.00, 8.00, 10.00
]

if USE_CANONICAL:
    print("[INIT] Using canonical trajectory generator.")
    with open(CANONICAL_PATH, "rb") as f:
        trajectoryLibrary = pickle.load(f)

    # canonicalSpec = extractCanonicalSpec(trajectoryLibrary, debug=True)
    transformLayer = TransformLayer(debug=False)

    apexHeight = float(np.random.choice(apexValues)) if apexValues else 0.0

    gen4D = Trajectory4DCanonical(
        trajectoryLibrary=trajectoryLibrary,
        transformLayer=transformLayer,
        interceptZValues=interceptZValues,
        apexHeight=apexHeight,
        apexValues=apexValues,
        spinTopValues=spinTopValues,
        spinSideValues=spinSideValues,
        bounceDistanceTolerance=0.75,
        debug=False,
        court=court
    )

else:
    print("[INIT] Using dynamic 4D trajectory generator.")
    gen4D = Trajectory4DDynamicGenerator(court)


# ---------------------------------------------------------
# 5) Movement model + Shot runners
# ---------------------------------------------------------
movementModel = PlayerMovement(court=court)

serveRunner = ServeShotRunner4D(
    court=court,
    generator4D=gen4D,
    spinTopValues=spinTopValues,
    spinSideValues=spinSideValues,
    apexValues=apexValues,
    interceptZValues=interceptZValues,
    trajecticSelector=trajecticsSelector,
    debug=False,
    debugLevel1=True,
)

rallyRunner = RallyShotRunner4D(
    court=court,
    generator4D=gen4D,
    spinTopValues=spinTopValues,
    spinSideValues=spinSideValues,
    apexValues=apexValues,
    interceptZValues=interceptZValues,
    trajecticSelector=trajecticsSelector,
    debug=False,
    debugLevel1=True,
)

pointRunner = PointRunner(
    court=court,
    serveRunner=serveRunner,
    rallyRunner=rallyRunner,
    movementModel=movementModel,
    maxShots=35,
    enableExecutionErrors=False,
)

tracker = ShotValueTracker(court, interceptZValues, apexValues)


# ---------------------------------------------------------
# 6) Run simulation
# ---------------------------------------------------------
allRows = []
numberOfPoints = max(1, int(commandLineArgs.numberOfPoints))

for i in range(numberOfPoints):
    serveSide = "DEUCE" if (i % 2 == 0) else "AD"
    pointResult = pointRunner.PlayPoint(serveSide)
    # print("PointResult Interception Point: " + str(pointResult))
    pointShotCount = len(pointResult["shots"])
    rows = tracker.ProcessPoint(pointResult, pointShotCount)
    allRows.extend(rows)

    print(
        f"Point {i+1}: winner = {pointResult['winningPlayer']},  "
        f"shots = {len(rows)},  "
        f"reason = {pointResult.get('reason', 'UNKNOWN')}"
    )

    # Visualization
    if commandLineArgs.visualize:
        outDir = "visualizations4d"
        os.makedirs(outDir, exist_ok=True)

        for si, shot in enumerate(pointResult["shots"]):
            # print("Point Result: " + str(pointResult))
            nextIntercept = None
            if si + 1 < len(pointResult["shots"]):
                nxt = pointResult["shots"][si + 1]
                if nxt.get("interceptPoint") is not None:
                    intercept_point = nxt["interceptPoint"]
                    if isinstance(intercept_point, (list, tuple)) and len(intercept_point) >= 3:
                        ix = float(intercept_point[0])
                        iy = float(intercept_point[1])
                        iz = float(intercept_point[2])
                        nextIntercept = (ix, iy, iz, 0.0)

            if shot.get("transformed") is not None:
                SaveTrajectoryPlot(
                    shot=shot,
                    transformed=shot["transformed"],
                    court=court,
                    filename=f"{outDir}/point_{i+1}_shot_{si+1}.png",
                    title=f"Point {i+1} Shot {si+1} ({shot['type']})",
                    # defensivePosX=shot["defensivePosX"],
                    # defensivePosY=shot["defensivePosY"],
                    nextIntercept=nextIntercept,
                    reachZMin=getattr(movementModel, "reachZMin", 0.10),
                    reachZMax=getattr(movementModel, "reachZMax", 3.0),
                    showInterceptCircle=True,
                    movementModel=movementModel
                )

# ---------------------------------------------------------
# 7) Export CSV
# ---------------------------------------------------------
if allRows:
    outCsv = "SimulationShotData4D001.csv"
    with open(outCsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=allRows[0].keys())
        writer.writeheader()
        writer.writerows(allRows)
    print(f"Export complete: {outCsv}")
else:
    print("No rows to export.")

# Remove-Item "visualizations4d\*" -Recurse -Force 
# python RunSimulation4D.py --numberOfPoints 10 --seed 123 

# note you can use this file to just build a new reference file on a laptop
# you need to set some memory limits to get it to work
# otherwise a laptop will crash
# the RunSimulation4DMP.py typically runs on a beefier machine, so not needed

# $env:SELECTIVE_PRESSURE_DUCKDB_TEMP_DIR = "D:\duckdb_spill"
# $env:SELECTIVE_PRESSURE_DUCKDB_MAX_TEMP_DIR_SIZE = "400GiB"
# $env:SELECTIVE_PRESSURE_DUCKDB_MEMORY_LIMIT = "6GiB"
# $env:SELECTIVE_PRESSURE_DUCKDB_THREADS = "1"
# $env:SELECTIVE_PRESSURE_DUCKDB_PRESERVE_INSERTION_ORDER = "false"

# python RunSimulation4D.py --consolidatedInputPath IterativeSimulation/ConsolidatedGen23.parquet --referenceOutputPath IterativeSimulation/Gen23Reference.parquet --numberOfPoints 1 --no-visualize

# you can also generate the GamePlay file here

# $env:SELECTIVE_PRESSURE_DUCKDB_TEMP_DIR = "D:\duckdb_spill"
# $env:SELECTIVE_PRESSURE_DUCKDB_MAX_TEMP_DIR_SIZE = "400GiB"
# $env:SELECTIVE_PRESSURE_DUCKDB_MEMORY_LIMIT = "6GiB"
# $env:SELECTIVE_PRESSURE_DUCKDB_THREADS = "1"
# $env:SELECTIVE_PRESSURE_DUCKDB_PRESERVE_INSERTION_ORDER = "false"

# python IterativeSimulation/GamePlayBuilder.py IterativeSimulation/ConsolidatedGen23.parquet --outputParquet IterativeSimulation/Gen23GamePlay.parquet --topBounceCellsPerContext 10 --shotsPerBounceCell 5 --includeAllServeOptions