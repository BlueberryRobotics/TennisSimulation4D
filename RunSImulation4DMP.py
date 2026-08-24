# RunSimulation4DMP.py
"""
High-performance multiprocessing simulation runner for 4D tennis trajectories.

Features:
    • Uses Trajectory4DCanonical ONLY (no dynamic generator)
    • Streams shot rows to a writer process (constant memory)
    • Outputs nightly Parquet file
    • Fully multiprocessing (workers + writer)
    • Includes a multiprocessing-safe progress bar
    • Snaps intercept/bounce/opponent/defensive positions to grid
    • Adds win flag
    • Uses PascalCase + camelCase naming conventions
"""

import os
import sys
import pickle
import datetime
import re
import random
import numpy as np
from multiprocessing import Process, Queue, cpu_count
import traceback
import pyarrow as pa
import pyarrow.parquet as pq
import subprocess


# ---------------------------------------------------------
# Import trajectory modules
# ---------------------------------------------------------
sys.path.append("Trajectory4D")

from CourtPlayerSettings import Court
from Trajectory4D.Trajectory4DCanonical import Trajectory4DCanonical
from Trajectory4D.TransformLayer import TransformLayer
from Trajectory4D.ServeShotRunner import ServeShotRunner
from Trajectory4D.RallyShotRunner import RallyShotRunner
from Trajectory4D.PointRunner import PointRunner
from Trajectory4D.PlayerMovement import PlayerMovement
from Trajectory4D.ShotValueTracker import ShotValueTracker
from Trajectory4D.SelectivePressure import SelectivePressure


def _GetIntEnv(name: str, defaultValue: int) -> int:
    rawValue = os.environ.get(name)
    if rawValue is None:
        return defaultValue
    try:
        return int(rawValue)
    except ValueError:
        return defaultValue


def DetectHardwareProfile() -> str:
    forcedProfile = os.environ.get("SIM_HW_PROFILE")
    if forcedProfile:
        normalizedProfile = forcedProfile.strip().lower()
        if normalizedProfile in ("laptop", "desktop", "msi"):
            return normalizedProfile

    logicalCpuCount = cpu_count() or 4
    if logicalCpuCount >= 16:
        return "msi"
    if logicalCpuCount >= 8:
        return "desktop"
    return "laptop"


def AutoTuneSimulationResources(numberOfPoints: int) -> dict:
    profile = DetectHardwareProfile()
    logicalCpuCount = cpu_count() or 4

    if profile == "msi":
        defaultWorkers = min(max(4, logicalCpuCount // 2), 8)
        defaultQueueMaxSize = 1024
        defaultWorkerRowBatchSize = 384
        defaultProgressBatchSize = 400
        defaultWriterBufferRows = 150_000
        defaultWriterRowGroupSize = 100_000
    elif profile == "desktop":
        defaultWorkers = min(max(3, logicalCpuCount // 2), 6)
        defaultQueueMaxSize = 768
        defaultWorkerRowBatchSize = 320
        defaultProgressBatchSize = 300
        defaultWriterBufferRows = 120_000
        defaultWriterRowGroupSize = 100_000
    else:
        defaultWorkers = min(max(1, logicalCpuCount // 2), 3)
        defaultQueueMaxSize = 256
        defaultWorkerRowBatchSize = 128
        defaultProgressBatchSize = 200
        defaultWriterBufferRows = 60_000
        defaultWriterRowGroupSize = 50_000

    if numberOfPoints < 100_000:
        defaultWorkers = max(1, min(defaultWorkers, 2))

    numWorkers = max(1, _GetIntEnv("SIM_NUM_WORKERS", defaultWorkers))
    queueMaxSize = max(64, _GetIntEnv("SIM_QUEUE_MAXSIZE", defaultQueueMaxSize))
    workerRowBatchSize = max(32, _GetIntEnv("SIM_WORKER_ROW_BATCH", defaultWorkerRowBatchSize))
    progressBatchSize = max(1, _GetIntEnv("SIM_PROGRESS_BATCH", defaultProgressBatchSize))
    writerBufferRows = max(10_000, _GetIntEnv("SIM_WRITER_BUFFER_ROWS", defaultWriterBufferRows))
    writerRowGroupSize = max(10_000, _GetIntEnv("SIM_WRITER_ROW_GROUP", defaultWriterRowGroupSize))

    return {
        "profile": profile,
        "logicalCpuCount": logicalCpuCount,
        "numWorkers": numWorkers,
        "queueMaxSize": queueMaxSize,
        "workerRowBatchSize": workerRowBatchSize,
        "progressBatchSize": progressBatchSize,
        "writerBufferRows": writerBufferRows,
        "writerRowGroupSize": writerRowGroupSize,
    }


# print("MODULE IMPORT:", __name__)
# =========================================================
# Build one shot row for Parquet output
# =========================================================

def BuildShotRow(shotData: dict):
                #  pointShotCount: int,
                #  cellSizeMeters: float,
                #  allowedZHeights,
                #  allowedApexHeights):
    """
    Convert the shot dictionary created by the tracker into
    a clean output row for Parquet writing.

    # NOTE: You must map shotData fields to your exact names.
    # """
   
    return {
        "interceptCol": int(shotData["interceptCol"]),
        "interceptRow": int(shotData["interceptRow"]),
        "interceptZ": float(shotData["interceptZ"]),
        "opponentCol": int(shotData["opponentCol"]),
        "opponentRow": int(shotData["opponentRow"]),
        "defensiveCol": int(shotData["defensiveCol"]),
        "defensiveRow": int(shotData["defensiveRow"]),
        "bounceCol": int(shotData["bounceCol"]),
        "bounceRow": int(shotData["bounceRow"]),
        "apexHeight": float(shotData["apexHeight"]),
        "spinTopRpm": int(shotData["spinTopRpm"]),
        "spinSideRpm": int(shotData["spinSideRpm"]),
        "initialVelocity": float(shotData["initialVelocity"]),
        "airTravelDistance": float(shotData["airTravelDistance"]),
        "netClearance": float(shotData["netClearance"]),
        "downhillSpeed": float(shotData.get("downhillSpeed", 0.0) or 0.0),
        "winner": bool(shotData.get("winner", False)),
        "wins": int(shotData["wins"]),
        "pointShotCount": int(shotData["pointShotCount"]),
        "winShotCount": int(shotData.get("winShotCount", 0))
        }


# =========================================================
# Writer process with integrated progress bar
# =========================================================

def WriterProcess(messageQueue,
                  outputPath,
                  parquetSchema,
                  expectedPoints,
                  writerRowGroupSize,
                  writerMaxBufferRows):
    ROW_GROUP_SIZE = max(10_000, int(writerRowGroupSize))
    MAX_BUFFER_ROWS = max(10_000, int(writerMaxBufferRows))
    TEMP_SUFFIX = ".tmp"

    tempPath = outputPath + TEMP_SUFFIX
    buffer = []

    totalRowsWritten = 0
    completedPoints = 0
    lastReport = 0
    reportInterval = max(1, expectedPoints // 100)

    writer = None

    try:
        writer = pq.ParquetWriter(
            tempPath,
            parquetSchema,
            compression="snappy",
        )

        while True:
            message = messageQueue.get()

            if message is None:
                break

            messageType, payload = message

            if messageType == "ROW":
                buffer.append(payload)

                if len(buffer) >= MAX_BUFFER_ROWS:
                    table = pa.Table.from_pylist(buffer, schema=parquetSchema)
                    writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
                    totalRowsWritten += len(buffer)
                    buffer.clear()

            elif messageType == "ROW_BATCH":
                if payload:
                    buffer.extend(payload)

                if len(buffer) >= MAX_BUFFER_ROWS:
                    table = pa.Table.from_pylist(buffer, schema=parquetSchema)
                    writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
                    totalRowsWritten += len(buffer)
                    buffer.clear()

            elif messageType == "PROGRESS":
                completedPoints += int(payload)
                if completedPoints - lastReport >= reportInterval:
                    percent = 100.0 * completedPoints / expectedPoints
                    print(
                        f"Progress: {percent:6.2f}% "
                        f"({completedPoints}/{expectedPoints} points)",
                        flush=True
                    )
                    lastReport = completedPoints

        # Final flush
        if buffer:
            table = pa.Table.from_pylist(buffer, schema=parquetSchema)
            writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
            totalRowsWritten += len(buffer)
            buffer.clear()

        writer.close()
        writer = None
        os.replace(tempPath, outputPath)

        print(
            f"Progress: {((100.0 * completedPoints / expectedPoints) if expectedPoints else 100.0):6.2f}% "
            f"({completedPoints}/{expectedPoints} points)",
            flush=True
        )

    except Exception:
        print("[Writer] FATAL ERROR — output file not finalized")
        traceback.print_exc()

        try:
            if writer:
                writer.close()
            if os.path.exists(tempPath):
                os.remove(tempPath)
        except Exception:
            pass

        raise

# =========================================================
# Worker process
# =========================================================

def WorkerProcess(pointStart,
                  pointEnd,
                  pointStep,
                  messageQueue,
                  trajectoryLibraryPath,
                  allowedZHeights,
                  apexValueList,
                  spinTopValues,
                  spinSideValues,
                  sharedCourt,
                  cellSizeMeters,
                  trajecticsSelectorFile,
                  workerRowBatchSize,
                  progressBatchSize,
                  workerSeed=None):
    """
    Reconstructs the full simulation pipeline inside each worker.
    Sends progress and shot rows back to WriterProcess.
    """

    if pointStart >= pointEnd:
        print("[WORKER EXITING: empty batch]")
        return

    if workerSeed is not None:
        np.random.seed(int(workerSeed))
        random.seed(int(workerSeed))
    
    with open(trajectoryLibraryPath, "rb") as fp:
        trajectoryLibrary = pickle.load(fp)

    court = sharedCourt

    # Initialize TrajecticsSelector only if a seed file is provided
    # If None, use fully random selection
    if trajecticsSelectorFile is not None:
        trajecticsSelector = SelectivePressure(
            trajecticsSelectorFile,
            0.5, 
            2,
            court=court,
            isPreFilteredReference=True
        )
    else:
        trajecticsSelector = None

    # Canonical generator
    transformLayer = TransformLayer(debug=False)
    generator4D = Trajectory4DCanonical(
        trajectoryLibrary=trajectoryLibrary,
        transformLayer=transformLayer,
        interceptZValues=allowedZHeights,
        apexHeight=int(np.random.choice(apexValueList)),
        apexValues=apexValueList,
        spinTopValues=spinTopValues,
        spinSideValues=spinSideValues,
        bounceDistanceTolerance=0.75,
        debug=False,
        court=court
    )

    movementModel = PlayerMovement(court)

    serveRunner = ServeShotRunner(
        court=court,
        generator4D=generator4D,
        spinTopValues=spinTopValues,
        spinSideValues=spinSideValues,
        apexValues=apexValueList,
        interceptZValues=allowedZHeights,
        trajecticSelector=trajecticsSelector
    )

    rallyRunner = RallyShotRunner(
        court=court,
        generator4D=generator4D,
        spinTopValues=spinTopValues,
        spinSideValues=spinSideValues,
        apexValues=apexValueList,
        interceptZValues=allowedZHeights,
        trajecticSelector=trajecticsSelector
    )

    pointRunner = PointRunner(
        court=court,
        serveRunner=serveRunner,
        rallyRunner=rallyRunner,
        movementModel=movementModel,
        maxShots=35,
        enableExecutionErrors=False,
    )

    # CURRENTLY ONLY USE PROCESS POINT
    tracker = ShotValueTracker(court, allowedZHeights, apexValueList)
    rowBatch = []
    pointsSinceProgress = 0
    workerRowBatchSize = max(32, int(workerRowBatchSize))
    progressBatchSize = max(1, int(progressBatchSize))

    skippedPointWinnerUnknown = 0
    skippedMissingInitialVelocity = 0
    skippedAllInvalidShots = 0
    skippedBuildFailed = 0
    serveFailureReasonCounts = {}

    # Loop over assigned points
    for pointIndex in range(pointStart, pointEnd, pointStep):
        serveSide = "DEUCE" if (pointIndex % 2 == 0) else "AD"

        try:
            pointResult = pointRunner.PlayPoint(serveSide)
        except Exception:
            # Rare malformed trajectic/shot states should not crash a long run.
            pointsSinceProgress += 1
            if pointsSinceProgress >= progressBatchSize:
                messageQueue.put(("PROGRESS", pointsSinceProgress))
                pointsSinceProgress = 0
            continue

        pointWinner = pointResult["winningPlayer"]
        pointShotCount = len(pointResult["shots"])

        if pointShotCount > 0:
            firstShot = pointResult["shots"][0]
            if firstShot.get("shotType") == "SERVE" and firstShot.get("outcome") != "IN":
                failureReason = str(firstShot.get("reason") or "UNKNOWN")
                serveFailureReasonCounts[failureReason] = serveFailureReasonCounts.get(failureReason, 0) + 1

        if pointWinner == "UNKNOWN":
            skippedPointWinnerUnknown += 1
            pointsSinceProgress += 1
            if pointsSinceProgress >= progressBatchSize:
                messageQueue.put(("PROGRESS", pointsSinceProgress))
                pointsSinceProgress = 0
            continue

        if any("initialVelocity" not in shot for shot in pointResult["shots"]):
            skippedMissingInitialVelocity += 1
            pointsSinceProgress += 1
            if pointsSinceProgress >= progressBatchSize:
                messageQueue.put(("PROGRESS", pointsSinceProgress))
                pointsSinceProgress = 0
            continue

        try:
            shotList = tracker.ProcessPoint(pointResult, pointShotCount)
        except Exception:
            pointsSinceProgress += 1
            if pointsSinceProgress >= progressBatchSize:
                messageQueue.put(("PROGRESS", pointsSinceProgress))
                pointsSinceProgress = 0
            continue
        #print("ShotList " + str(shotList))

        # ---------------------------------------------------------
        # Keep valid rows even when a point has one malformed shot.
        # Only skip the entire point when all shot rows are invalid.
        # ---------------------------------------------------------
        validShotList = [
            shot for shot in shotList
            if shot.get("apexHeight") != 0.0
        ]

        if not validShotList:
            skippedAllInvalidShots += 1
            pointsSinceProgress += 1
            if pointsSinceProgress >= progressBatchSize:
                messageQueue.put(("PROGRESS", pointsSinceProgress))
                pointsSinceProgress = 0
            continue

        pointRows = []
        buildFailed = False
        for shotDict in validShotList:
            try:
                row = BuildShotRow(shotDict)
            except (TypeError, ValueError, KeyError):
                buildFailed = True
                break
            pointRows.append(row)

        if buildFailed:
            skippedBuildFailed += 1
            pointsSinceProgress += 1
            if pointsSinceProgress >= progressBatchSize:
                messageQueue.put(("PROGRESS", pointsSinceProgress))
                pointsSinceProgress = 0
            continue

        for row in pointRows:
            rowBatch.append(row)
            if len(rowBatch) >= workerRowBatchSize:
                messageQueue.put(("ROW_BATCH", rowBatch))
                rowBatch = []

        pointsSinceProgress += 1
        if pointsSinceProgress >= progressBatchSize:
            messageQueue.put(("PROGRESS", pointsSinceProgress))
            pointsSinceProgress = 0

    if rowBatch:
        messageQueue.put(("ROW_BATCH", rowBatch))
    if pointsSinceProgress:
        messageQueue.put(("PROGRESS", pointsSinceProgress))

    if serveFailureReasonCounts:
        sortedReasons = sorted(
            serveFailureReasonCounts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        reasonSummary = ", ".join(
            f"{reason}={count}"
            for reason, count in sortedReasons
        )
        print(
            "[WORKER SUMMARY] "
            f"pointRange={pointStart}:{pointEnd}:{pointStep} "
            f"serveFailures: {reasonSummary}",
            flush=True,
        )

# =========================================================
# Main entry point: runs multiprocessing simulation
# =========================================================

def RunSimulationMP(numberOfPoints: int,
                    trajectorySelectorFile: str,
                    winPercentageThreshold: float,
                    countThreshold: int,
                    outputDirectory="IterativeSimulation/ShotData",
                    cellSizeMeters=1.3716,
                    printCourtGeometryCheck: bool = False,
                    baseSeed: int = None):
    
    """Main driver for multiprocessing canonical simulation."""
    os.makedirs(outputDirectory, exist_ok=True)

    # Nightly timestamped Parquet file
    timestampText = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outputPath = os.path.join(outputDirectory,
                              f"shots_{timestampText}.parquet")

    # Parquet schema
    parquetSchema = pa.schema([
        ("interceptCol", pa.int32()),
        ("interceptRow", pa.int32()),
        ("interceptZ",   pa.float32()),

        ("opponentCol", pa.int32()),
        ("opponentRow", pa.int32()),

        ("defensiveCol", pa.int32()),
        ("defensiveRow", pa.int32()),

        ("bounceCol", pa.int32()),
        ("bounceRow", pa.int32()),

        ("apexHeight", pa.float32()),

        ("spinTopRpm",    pa.int32()),
        ("spinSideRpm",   pa.int32()),

        ("initialVelocity", pa.float32()),
        ("airTravelDistance", pa.float32()),
        ("netClearance", pa.float32()),
        ("downhillSpeed", pa.float32()),

        ("winner", pa.bool_()),
        ("wins", pa.int32()),
        ("pointShotCount", pa.int32()),
        ("winShotCount", pa.int32())
    ])

    #-------------------------------------------------
    # Canonical value lists - Schema level invariants
    # These cannot be changed without rebuilding the 
    # Trajectory Library and running a new simulation
    #-------------------------------------------------

    # 11 values
    interceptZValues = [
        0.30, 0.60, 1.00, 1.25, 1.50, 1.80,
        2.10, 2.40, 2.70, 3.00, 3.30
    ]

    spinTopValues = np.array([-3000, -1500, 0, 1500, 2500, 3500])
    spinSideValues = np.array([-2000, -1000, 0, 1000, 2000])

    # 14 values
    apexValues = [
        1.00, 1.25, 1.50, 1.80, 2.10, 2.40, 2.70,
        3.00, 3.30, 3.60, 4.50, 6.00, 8.00, 10.00
    ]

    trajectoryLibraryPath = "Trajectory4DLibrary007.pkl"

    # Shared Court instance
    court = Court(geometryMode="short_rows_13_14")
    averageShotsPerPoint = 6.5
    expectedTotalRows = numberOfPoints * averageShotsPerPoint

    # Auto-tune process and buffering settings by hardware profile.
    tunedResources = AutoTuneSimulationResources(numberOfPoints)
    print(
        "[TUNE] "
        f"profile={tunedResources['profile']} "
        f"logicalCpuCount={tunedResources['logicalCpuCount']} "
        f"numWorkers={tunedResources['numWorkers']} "
        f"queueMaxSize={tunedResources['queueMaxSize']} "
        f"workerRowBatchSize={tunedResources['workerRowBatchSize']} "
        f"progressBatchSize={tunedResources['progressBatchSize']} "
        f"writerBufferRows={tunedResources['writerBufferRows']} "
        f"writerRowGroupSize={tunedResources['writerRowGroupSize']}"
    )
    print("[TUNE] courtGeometryMode=short_rows_13_14")
    if baseSeed is not None:
        print(f"[TUNE] baseSeed={baseSeed}")

    if printCourtGeometryCheck:
        print("[GEOM] Geometry check start")
        print(
            "[GEOM] "
            f"mode={getattr(court, 'geometryMode', 'uniform')} "
            f"gridRows={court.gridRows} "
            f"lengthFence_m={court.lengthFence:.4f}"
        )
        for boundaryIndex in [4, 8, 13, 18, 22]:
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
        print("[GEOM] Geometry check end")

    # Launch writer
    messageQueue = Queue(maxsize=tunedResources["queueMaxSize"])
    writerProcess = Process(
        target=WriterProcess,
        args=(
            messageQueue,
            outputPath,
            parquetSchema,
            numberOfPoints,
            tunedResources["writerRowGroupSize"],
            tunedResources["writerBufferRows"],
        )
    )
    writerProcess.start()

    # Worker distribution
    numWorkers = tunedResources["numWorkers"]

    workerRanges = [
        (workerIndex, numberOfPoints, numWorkers)
        for workerIndex in range(numWorkers)
    ]
    workerList = []

    for workerIndex, (pointStart, pointEnd, pointStep) in enumerate(workerRanges):
        workerSeed = (int(baseSeed) + workerIndex) if baseSeed is not None else None
        worker = Process(
            target=WorkerProcess,
            args=(
                pointStart,
                pointEnd,
                pointStep,
                messageQueue,
                trajectoryLibraryPath,
                interceptZValues,
                apexValues,
                spinTopValues,
                spinSideValues,
                court,
                cellSizeMeters,
                trajectorySelectorFile,
                tunedResources["workerRowBatchSize"],
                tunedResources["progressBatchSize"],
                workerSeed,
            )
        )
        worker.start()
        workerList.append(worker)

    # 1. Wait for all workers to finish
    for worker in workerList:
        worker.join()

    failedWorkers = [
        (index + 1, worker.exitcode)
        for index, worker in enumerate(workerList)
        if worker.exitcode not in (0, None)
    ]

    # Tell writer to stop
    messageQueue.put(None)
    writerProcess.join()

    if failedWorkers:
        raise RuntimeError(
            "One or more worker processes failed: "
            + ", ".join(
                f"Process-{workerId} exitcode={exitCode}"
                for workerId, exitCode in failedWorkers
            )
        )

    print(
        f"[DONE] Parquet file written: {outputPath} at time: {datetime.datetime.now()} "
        f"expectedRows~{int(expectedTotalRows):,} workers={numWorkers} "
        f"profile={tunedResources['profile']}"
    )
    return outputPath


def FindLatestGenerationReference(iterativeDirectory: str):
    latestGeneration = -1
    latestReferenceFile = None

    if not os.path.isdir(iterativeDirectory):
        return latestGeneration, latestReferenceFile

    for name in os.listdir(iterativeDirectory):
        generationMatch = re.match(r"^Gen(\d+)Reference\.parquet$", name)
        if not generationMatch:
            continue

        generationNumber = int(generationMatch.group(1))
        if generationNumber > latestGeneration:
            latestGeneration = generationNumber
            latestReferenceFile = os.path.join(iterativeDirectory, name)

    return latestGeneration, latestReferenceFile


def FindLatestGenerationConsolidated(iterativeDirectory: str):
    latestGeneration = -1
    latestConsolidatedFile = None

    if not os.path.isdir(iterativeDirectory):
        return latestGeneration, latestConsolidatedFile

    for name in os.listdir(iterativeDirectory):
        generationMatch = re.match(r"^ConsolidatedGen(\d+)(?:_.*)?\.parquet$", name)
        if not generationMatch:
            continue

        generationNumber = int(generationMatch.group(1))
        if generationNumber > latestGeneration:
            latestGeneration = generationNumber
            latestConsolidatedFile = os.path.join(iterativeDirectory, name)

    return latestGeneration, latestConsolidatedFile


def ParseConsolidatedGenerationFromPath(consolidatedPath: str):
    fileName = os.path.basename(consolidatedPath)
    match = re.match(r"^ConsolidatedGen(\d+)(?:_.*)?\.parquet$", fileName)
    if not match:
        return None
    return int(match.group(1))


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    NUMBER_OF_POINTS = 10_000_000 # Number of points per generation
    GENERATIONS_TO_RUN = 10  # Number of generations to execute before stopping

    ITERATIVE_DIRECTORY = "IterativeSimulation"
    RESUME_FROM_LAST_REFERENCE = True
    CONSOLIDATE_THREADS = 1
    CONSOLIDATE_MEMORY_LIMIT = os.environ.get("SIM_CONSOLIDATE_MEMORY_LIMIT", "24GB")
    CONSOLIDATE_DISABLE_INSERTION_ORDER = True
    INCLUDE_PREVIOUS_FILE_IN_CONSOLIDATION = True
    PRINT_COURT_GEOMETRY_CHECK = os.environ.get("SIM_PRINT_COURT_GEOMETRY_CHECK", "0") == "1"
    SIM_SEED = os.environ.get("SIM_SEED")

    baseSeed = None
    if SIM_SEED is not None:
        try:
            baseSeed = int(SIM_SEED)
        except ValueError:
            raise ValueError("SIM_SEED must be an integer when provided")

    # Initial seed consolidated file for fresh runs.
    # Set to None for fully random selection.
    initialSeedConsolidatedFile = "IterativeSimulation/ConsolidatedGen2_006.parquet"

    currentSelectorFile = None
    currentReferenceFile = None
    startGeneration = 1

    # Optional automatic resume with safeguard:
    # - Prefer latest GenXReference
    # - If missing but ConsolidatedGenX exists, rebuild GenXReference and resume
    if RESUME_FROM_LAST_REFERENCE:
        latestGeneration, latestReferenceFile = FindLatestGenerationReference(ITERATIVE_DIRECTORY)
        latestConsolidatedGeneration, latestConsolidatedFile = FindLatestGenerationConsolidated(ITERATIVE_DIRECTORY)

        # Primary resume path: last completed generation is defined by consolidated output.
        if latestConsolidatedGeneration >= 0:
            resumeGeneration = latestConsolidatedGeneration
            priorConsolidatedFile = latestConsolidatedFile
            priorReferenceFile = os.path.join(
                ITERATIVE_DIRECTORY,
                f"Gen{resumeGeneration}Reference.parquet"
            )

            if not os.path.exists(priorReferenceFile):
                print(
                    "[RECOVER] Found consolidated without matching reference; "
                    f"building missing file: {priorReferenceFile}"
                )
                SelectivePressure.BuildReferenceFile(
                    sourceParquetPath=priorConsolidatedFile,
                    outputParquetPath=priorReferenceFile
                )

            currentReferenceFile = priorReferenceFile
            currentSelectorFile = priorConsolidatedFile
            startGeneration = resumeGeneration + 1

            print(f"[RESUME] Using {currentReferenceFile}")
            print(f"[RESUME] Previous consolidated: {currentSelectorFile}")
            print(f"[RESUME] Continuing at generation {startGeneration}")

        # Reference-only resume path: resume from latest reference when consolidated output is unavailable.
        elif latestGeneration >= 0 and latestReferenceFile is not None:
            currentReferenceFile = latestReferenceFile
            currentSelectorFile = latestConsolidatedFile
            startGeneration = latestGeneration + 1
            print(f"[RESUME] Using latest reference: {currentReferenceFile}")
            if currentSelectorFile is None:
                print("[RESUME] No consolidated file found to merge as --previousFile")

    # Fresh run path if resume did not find prior generation output.
    if currentReferenceFile is None:
        currentSelectorFile = initialSeedConsolidatedFile
        if currentSelectorFile is not None:
            if not os.path.exists(currentSelectorFile):
                print(f"[ERROR] Seed file not found: {currentSelectorFile}")
                sys.exit(1)

            seedGeneration = ParseConsolidatedGenerationFromPath(currentSelectorFile)
            if seedGeneration is None:
                seedGeneration = 0

            seedReferenceFile = os.path.join(
                ITERATIVE_DIRECTORY,
                f"Gen{seedGeneration}Reference.parquet"
            )
            print(f"[INFO] Building initial reference file: {seedReferenceFile}")
            SelectivePressure.BuildReferenceFile(
                sourceParquetPath=currentSelectorFile,
                outputParquetPath=seedReferenceFile
            )
            currentReferenceFile = seedReferenceFile
            startGeneration = seedGeneration + 1
        else:
            print("[INFO] No seed file - using fully random selection")

    winPercentageThreshold = 0.2
    countThreshold = 2

    if GENERATIONS_TO_RUN <= 0:
        print("[INFO] GENERATIONS_TO_RUN <= 0. Nothing to run.")
        sys.exit(0)

    endGeneration = startGeneration + GENERATIONS_TO_RUN - 1

    if startGeneration > endGeneration:
        print(
            f"[INFO] Nothing to run. startGeneration={startGeneration} "
            f"endGeneration={endGeneration}."
        )
        sys.exit(0)

    print(
        f"[INFO] Running generations {startGeneration} through {endGeneration} "
        f"({GENERATIONS_TO_RUN} total)."
    )

    for generation in range(startGeneration, endGeneration + 1):

        print(f"\n=== GENERATION {generation} ===")

        # ----------------------------
        # 1. RUN SIMULATION (using previous generation reference)
        # ----------------------------
        selectorInputForWorkers = currentReferenceFile
        newDataFile = RunSimulationMP(
            NUMBER_OF_POINTS,
            selectorInputForWorkers,
            winPercentageThreshold,
            countThreshold,
            outputDirectory=ITERATIVE_DIRECTORY,
            printCourtGeometryCheck=PRINT_COURT_GEOMETRY_CHECK,
            baseSeed=baseSeed,
        )

        # ----------------------------
        # 2. CONSOLIDATE
        # ----------------------------
        consolidatedFile = os.path.join(
            ITERATIVE_DIRECTORY,
            f"ConsolidatedGen{generation}.parquet"
        )

        # command to run the consolidation script with appropriate arguments
        cmd = [
            "python",
            "IterativeSimulation/ConsolidateResults009.py",
            newDataFile,
            consolidatedFile,
            "--minPrevCount", "1",
            "--threads", str(CONSOLIDATE_THREADS),
            "--memoryLimit", CONSOLIDATE_MEMORY_LIMIT
        ]

        if CONSOLIDATE_DISABLE_INSERTION_ORDER:
            cmd.append("--disableInsertionOrder")

        # Include previous consolidated results when enabled.
        if INCLUDE_PREVIOUS_FILE_IN_CONSOLIDATION and currentSelectorFile is not None:
            cmd.extend([
                "--previousFile", currentSelectorFile,
                "--minWinPct", "0.2"
            ])
        elif currentSelectorFile is not None:
            print("[INFO] Consolidation test mode: skipping previousFile merge")

        print("[INFO] Running consolidation...")
        subprocess.run(cmd, check=True)

        # ----------------------------
        # 3. BUILD THIS GENERATION'S REFERENCE (restart point)
        # ----------------------------
        generationReferenceFile = os.path.join(
            ITERATIVE_DIRECTORY,
            f"Gen{generation}Reference.parquet"
        )
        print(f"[INFO] Building generation reference file: {generationReferenceFile}")
        SelectivePressure.BuildReferenceFile(
            sourceParquetPath=consolidatedFile,
            outputParquetPath=generationReferenceFile
        )

        # ----------------------------
        # 4. UPDATE INPUTS FOR NEXT GENERATION
        # ----------------------------
        currentSelectorFile = consolidatedFile
        currentReferenceFile = generationReferenceFile

        print(f"[GEN {generation}] Consolidated: {currentSelectorFile}")
        print(f"[GEN {generation}] Reference: {currentReferenceFile}")

# SET NUMBER OF POINTS AND NUMBER OF GENERATIONS LINE 707
# SET debugLevel1: bool = False in SelectivePressure.py, RallyShotRunner.py and ServeShotRunner.py


