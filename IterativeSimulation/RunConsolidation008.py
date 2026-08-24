import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple


_TIMESTAMP_PATTERN = re.compile(r"(\d{8})_(\d{6})")


def _NormalizePath(pathValue: str) -> str:
    return os.path.normpath(os.path.abspath(pathValue))


def _FormatDuration(seconds: float) -> str:
    totalSeconds = int(max(0, seconds))
    hours = totalSeconds // 3600
    minutes = (totalSeconds % 3600) // 60
    secs = totalSeconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _ExtractTimestampSortKey(filePath: str) -> Tuple[int, datetime, str]:
    fileName = os.path.basename(filePath)
    match = _TIMESTAMP_PATTERN.search(fileName)
    if not match:
        # Put non-timestamped files after timestamped ones; fallback to name.
        return (1, datetime.max, fileName)

    try:
        parsed = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        return (0, parsed, fileName)
    except ValueError:
        return (1, datetime.max, fileName)


def _FindInputFiles(inputFolder: str, filePattern: str, recursive: bool) -> List[str]:
    pattern = os.path.join(inputFolder, "**", filePattern) if recursive else os.path.join(inputFolder, filePattern)
    matches = glob.glob(pattern, recursive=recursive)
    files = [
        _NormalizePath(path)
        for path in matches
        if os.path.isfile(path)
    ]
    files.sort(key=_ExtractTimestampSortKey)
    return files


def _BuildConsolidationCommand(
    consolidateScript: str,
    inputPattern: str,
    outputFile: str,
    previousFile: Optional[str],
    minPrevCount: int,
    priorCountCap: int,
    minWinPct: float,
    threads: int,
    memoryLimit: str,
    disableInsertionOrder: bool,
) -> List[str]:
    command = [
        sys.executable,
        consolidateScript,
        inputPattern,
        outputFile,
        "--minPrevCount",
        str(int(minPrevCount)),
        "--priorCountCap",
        str(int(priorCountCap)),
        "--minWinPct",
        str(float(minWinPct)),
        "--threads",
        str(int(threads)),
        "--memoryLimit",
        str(memoryLimit),
    ]

    if previousFile:
        command.extend(["--previousFile", previousFile])

    if disableInsertionOrder:
        command.append("--disableInsertionOrder")

    return command


def _RunOne(
    consolidateScript: str,
    inputPattern: str,
    outputFile: str,
    previousFile: Optional[str],
    minPrevCount: int,
    priorCountCap: int,
    minWinPct: float,
    threads: int,
    memoryLimit: str,
    disableInsertionOrder: bool,
    dryRun: bool,
) -> int:
    outputDir = os.path.dirname(outputFile)
    if outputDir:
        os.makedirs(outputDir, exist_ok=True)

    command = _BuildConsolidationCommand(
        consolidateScript=consolidateScript,
        inputPattern=inputPattern,
        outputFile=outputFile,
        previousFile=previousFile,
        minPrevCount=minPrevCount,
        priorCountCap=priorCountCap,
        minWinPct=minWinPct,
        threads=threads,
        memoryLimit=memoryLimit,
        disableInsertionOrder=disableInsertionOrder,
    )

    print("Command:")
    print(" ".join(command))

    if dryRun:
        return 0

    start = time.time()
    result = subprocess.run(command)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"FAILED in {_FormatDuration(elapsed)} with code {result.returncode}")
    else:
        print(f"Completed in {_FormatDuration(elapsed)}")
    return int(result.returncode)


def _RunMergeMode(args) -> int:
    files = _FindInputFiles(args.inputFolder, args.filePattern, args.recursive)
    if not files:
        raise FileNotFoundError(
            f"No input files found in folder '{args.inputFolder}' with pattern '{args.filePattern}'"
        )

    globPattern = os.path.join(args.inputFolder, "**", args.filePattern) if args.recursive else os.path.join(args.inputFolder, args.filePattern)
    inputPattern = _NormalizePath(globPattern)
    outputFile = _NormalizePath(args.outputFile)
    previousFile = _NormalizePath(args.previousFile) if args.previousFile else None

    if previousFile and not os.path.exists(previousFile):
        raise FileNotFoundError(f"Previous file not found: {previousFile}")

    print("=== Consolidation 008: Merge Mode ===")
    print(f"Input folder: {args.inputFolder}")
    print(f"Pattern: {args.filePattern}")
    print(f"Matched files: {len(files)}")
    print(f"Output: {outputFile}")
    if previousFile:
        print(f"Previous: {previousFile}")

    return _RunOne(
        consolidateScript=args.consolidateScript,
        inputPattern=inputPattern,
        outputFile=outputFile,
        previousFile=previousFile,
        minPrevCount=args.minPrevCount,
        priorCountCap=args.priorCountCap,
        minWinPct=args.minWinPct,
        threads=args.threads,
        memoryLimit=args.memoryLimit,
        disableInsertionOrder=args.disableInsertionOrder,
        dryRun=args.dryRun,
    )


def _RunSequentialMode(args) -> int:
    files = _FindInputFiles(args.inputFolder, args.filePattern, args.recursive)
    if not files:
        raise FileNotFoundError(
            f"No input files found in folder '{args.inputFolder}' with pattern '{args.filePattern}'"
        )

    print("=== Consolidation 008: Sequential Mode ===")
    print(f"Input folder: {args.inputFolder}")
    print(f"Pattern: {args.filePattern}")
    print(f"Matched files: {len(files)}")
    print(f"Start generation: {args.startGen}")
    print(f"Output template: {args.outputTemplate}")
    if args.previousFile:
        print(f"Initial previous file: {args.previousFile}")

    for index, path in enumerate(files):
        gen = args.startGen + index
        print(f"  Gen {gen:02d} <- {os.path.basename(path)}")

    previousOutput: Optional[str] = _NormalizePath(args.previousFile) if args.previousFile else None
    if previousOutput and not os.path.exists(previousOutput):
        raise FileNotFoundError(f"Initial previous file not found: {previousOutput}")
    totalStart = time.time()

    for index, inputFile in enumerate(files):
        generation = args.startGen + index
        outputFile = _NormalizePath(args.outputTemplate.format(gen=generation))

        if args.resume and os.path.exists(outputFile):
            print(f"Skipping Gen {generation:02d}; output exists: {outputFile}")
            previousOutput = outputFile
            continue

        if previousOutput and not os.path.exists(previousOutput):
            raise FileNotFoundError(
                f"Previous generation file is missing for Gen {generation:02d}: {previousOutput}"
            )

        print("\n" + "=" * 80)
        print(f"Starting Gen {generation:02d}")

        exitCode = _RunOne(
            consolidateScript=args.consolidateScript,
            inputPattern=inputFile,
            outputFile=outputFile,
            previousFile=previousOutput,
            minPrevCount=args.minPrevCount,
            priorCountCap=args.priorCountCap,
            minWinPct=args.minWinPct,
            threads=args.threads,
            memoryLimit=args.memoryLimit,
            disableInsertionOrder=args.disableInsertionOrder,
            dryRun=args.dryRun,
        )
        if exitCode != 0:
            return exitCode

        previousOutput = outputFile

    print(f"All sequential generations completed in {_FormatDuration(time.time() - totalStart)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run ConsolidateResults008.py in either merge mode (all files in a folder -> one generation) "
            "or sequential mode (one file per generation, oldest to newest)."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["merge", "sequential"],
        required=True,
        help="merge: all files to one output; sequential: one file per generation",
    )
    parser.add_argument(
        "--inputFolder",
        required=True,
        help="Folder containing shot parquet files",
    )
    parser.add_argument(
        "--filePattern",
        default="shots_*.parquet",
        help="Glob pattern for input files inside inputFolder",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search inputFolder recursively",
    )

    parser.add_argument(
        "--consolidateScript",
        default="IterativeSimulation/ConsolidateResults008.py",
        help="Path to ConsolidateResults008.py",
    )

    parser.add_argument(
        "--outputFile",
        default="",
        help="Required for merge mode: consolidated output parquet path",
    )
    parser.add_argument(
        "--outputTemplate",
        default="IterativeSimulation/ConsolidatedGen{gen}_008.parquet",
        help="Used in sequential mode; must include {gen}",
    )
    parser.add_argument(
        "--previousFile",
        default="",
        help=(
            "Optional previous consolidated parquet. If provided in sequential mode, "
            "it is used only for the first generation run; subsequent runs chain from "
            "newly produced consolidated outputs."
        ),
    )
    parser.add_argument(
        "--startGen",
        type=int,
        default=0,
        help="Starting generation number in sequential mode",
    )

    parser.add_argument("--minPrevCount", type=int, default=2)
    parser.add_argument("--priorCountCap", type=int, default=50)
    parser.add_argument("--minWinPct", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memoryLimit", default="8GB")
    parser.add_argument("--disableInsertionOrder", action="store_true")

    parser.add_argument("--resume", action="store_true", help="Sequential mode: skip outputs that already exist")
    parser.add_argument("--dryRun", action="store_true", help="Print resolved commands without executing")

    args = parser.parse_args()

    args.inputFolder = _NormalizePath(args.inputFolder)
    args.consolidateScript = _NormalizePath(args.consolidateScript)

    if not os.path.exists(args.consolidateScript):
        raise FileNotFoundError(f"Consolidation script not found: {args.consolidateScript}")

    if not os.path.isdir(args.inputFolder):
        raise NotADirectoryError(f"Input folder not found: {args.inputFolder}")

    if args.mode == "merge":
        if not args.outputFile:
            raise ValueError("--outputFile is required when --mode merge")
        return _RunMergeMode(args)

    if "{gen}" not in args.outputTemplate:
        raise ValueError("--outputTemplate must include '{gen}' in sequential mode")

    return _RunSequentialMode(args)


if __name__ == "__main__":
    raise SystemExit(main())
