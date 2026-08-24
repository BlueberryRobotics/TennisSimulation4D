import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple


def _ParseOverrides(raw: str) -> Dict[int, str]:
    overrides: Dict[int, str] = {}
    if not raw:
        return overrides

    # Format: "0=path_or_glob;1=other_path_or_glob"
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"Invalid override '{chunk}'. Expected format like 0=pattern;1=pattern"
            )
        left, right = chunk.split("=", 1)
        generation = int(left.strip())
        pattern = right.strip()
        if not pattern:
            raise ValueError(f"Generation {generation} override has empty pattern")
        overrides[generation] = pattern

    return overrides


def _ParseInputPlanCsv(csv_path: str) -> Dict[int, str]:
    if not csv_path:
        return {}

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input plan CSV not found: {csv_path}")

    mappings: Dict[int, str] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"gen", "inputPattern"}
        if not required_columns.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "Input plan CSV must contain headers: gen,inputPattern"
            )

        for row in reader:
            generation = int(str(row["gen"]).strip())
            input_pattern = str(row["inputPattern"]).strip()
            if not input_pattern:
                raise ValueError(f"Empty inputPattern for generation {generation}")
            mappings[generation] = input_pattern

    return mappings


def _ResolveInputPattern(
    generation: int,
    csv_mappings: Dict[int, str],
    overrides: Dict[int, str],
    multi_file_end_gen: int,
    multi_file_pattern: str,
    single_file_template: str,
) -> str:
    if generation in csv_mappings:
        return csv_mappings[generation].format(gen=generation)

    if generation in overrides:
        return overrides[generation].format(gen=generation)

    if generation <= multi_file_end_gen:
        if not multi_file_pattern:
            raise ValueError(
                f"Generation {generation} requires --multiFilePattern or an override"
            )
        return multi_file_pattern.format(gen=generation)

    if not single_file_template:
        raise ValueError(
            f"Generation {generation} requires --singleFileTemplate or an override"
        )
    return single_file_template.format(gen=generation)


def _GlobMatches(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern))


def _BuildCommand(
    consolidate_script: str,
    input_pattern: str,
    output_file: str,
    previous_file: str,
    min_prev_count: int,
    prior_count_cap: int,
    min_win_pct: float,
    threads: int,
    memory_limit: str,
    disable_insertion_order: bool,
) -> List[str]:
    command = [
        sys.executable,
        consolidate_script,
        input_pattern,
        output_file,
        "--minPrevCount",
        str(min_prev_count),
        "--priorCountCap",
        str(prior_count_cap),
        "--minWinPct",
        str(min_win_pct),
        "--threads",
        str(threads),
        "--memoryLimit",
        memory_limit,
    ]

    if previous_file:
        command.extend(["--previousFile", previous_file])

    if disable_insertion_order:
        command.append("--disableInsertionOrder")

    return command


def _FormatDuration(seconds: float) -> str:
    minutes = int(seconds // 60)
    hours = int(minutes // 60)
    minutes = minutes % 60
    sec = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially run ConsolidateResults007.py across generations, with support "
            "for early multi-file raw inputs and later single-file inputs."
        )
    )

    parser.add_argument("--startGen", type=int, default=0)
    parser.add_argument("--endGen", type=int, default=23)

    parser.add_argument(
        "--consolidateScript",
        default="IterativeSimulation/ConsolidateResults007.py",
        help="Path to consolidation script",
    )

    parser.add_argument(
        "--outputTemplate",
        default="IterativeSimulation/ConsolidatedGen{gen}_007.parquet",
        help="Output parquet pattern; must include {gen}",
    )

    parser.add_argument(
        "--multiFileEndGen",
        type=int,
        default=3,
        help="Generations <= this use --multiFilePattern unless overridden",
    )
    parser.add_argument(
        "--multiFilePattern",
        default="",
        help=(
            "Glob for early generations (can include {gen}), e.g. "
            "IterativeSimulation/Gen{gen}Raw/shots_*.parquet"
        ),
    )
    parser.add_argument(
        "--singleFileTemplate",
        default="",
        help=(
            "Template for later generations (usually one raw file per generation), e.g. "
            "IterativeSimulation/RawByGen/Gen{gen}.parquet"
        ),
    )

    parser.add_argument(
        "--inputPlanCsv",
        default="",
        help=(
            "Optional CSV with headers gen,inputPattern. Useful when each generation "
            "needs a distinct raw input pattern."
        ),
    )

    parser.add_argument(
        "--inputOverrides",
        default="",
        help=(
            "Per-generation overrides: 0=pattern;1=pattern;... "
            "Patterns may include {gen}. Overrides take precedence over --inputPlanCsv."
        ),
    )

    parser.add_argument("--minPrevCount", type=int, default=2)
    parser.add_argument("--priorCountCap", type=int, default=50)
    parser.add_argument("--minWinPct", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memoryLimit", default="8GB")
    parser.add_argument("--disableInsertionOrder", action="store_true")

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip generations whose output parquet already exists",
    )
    parser.add_argument(
        "--dryRun",
        action="store_true",
        help="Show resolved plan without running consolidations",
    )
    parser.add_argument(
        "--requireMatches",
        action="store_true",
        help="Fail if a generation input pattern has zero matches",
    )

    args = parser.parse_args()

    if "{gen}" not in args.outputTemplate:
        raise ValueError("--outputTemplate must contain {gen}")

    if args.startGen > args.endGen:
        raise ValueError("--startGen must be <= --endGen")

    consolidate_script = os.path.normpath(args.consolidateScript)
    if not os.path.exists(consolidate_script):
        raise FileNotFoundError(f"Consolidation script not found: {consolidate_script}")

    csv_mappings = _ParseInputPlanCsv(args.inputPlanCsv)
    overrides = _ParseOverrides(args.inputOverrides)

    generation_plan: List[Tuple[int, str, str]] = []
    for generation in range(args.startGen, args.endGen + 1):
        input_pattern = _ResolveInputPattern(
            generation,
            csv_mappings,
            overrides,
            args.multiFileEndGen,
            args.multiFilePattern,
            args.singleFileTemplate,
        )
        output_file = os.path.normpath(args.outputTemplate.format(gen=generation))
        generation_plan.append((generation, input_pattern, output_file))

    print("=== Consolidation 007 Plan ===")
    for generation, input_pattern, output_file in generation_plan:
        matches = _GlobMatches(input_pattern)
        print(
            f"Gen {generation:02d}: input='{input_pattern}' matches={len(matches)} -> output='{output_file}'"
        )
        if args.requireMatches and len(matches) == 0:
            raise FileNotFoundError(
                f"No files matched for generation {generation}: {input_pattern}"
            )

    if args.dryRun:
        print("Dry run complete. No consolidations executed.")
        return 0

    total_start = time.time()
    for generation, input_pattern, output_file in generation_plan:
        if args.resume and os.path.exists(output_file):
            print(f"Skipping Gen {generation:02d} (output exists): {output_file}")
            continue

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        previous_file = ""
        if generation > args.startGen:
            previous_file = os.path.normpath(args.outputTemplate.format(gen=generation - 1))
        elif generation > 0:
            # When starting mid-chain, default previous to prior generation output path.
            previous_file = os.path.normpath(args.outputTemplate.format(gen=generation - 1))

        if previous_file and not os.path.exists(previous_file):
            raise FileNotFoundError(
                f"Previous generation output not found for Gen {generation}: {previous_file}"
            )

        command = _BuildCommand(
            consolidate_script=consolidate_script,
            input_pattern=input_pattern,
            output_file=output_file,
            previous_file=previous_file,
            min_prev_count=args.minPrevCount,
            prior_count_cap=args.priorCountCap,
            min_win_pct=args.minWinPct,
            threads=args.threads,
            memory_limit=args.memoryLimit,
            disable_insertion_order=args.disableInsertionOrder,
        )

        print("\n" + "=" * 80)
        print(f"Starting Gen {generation:02d}")
        print("Command:")
        print(" ".join(command))

        start = time.time()
        result = subprocess.run(command)
        elapsed = time.time() - start

        if result.returncode != 0:
            print(
                f"Gen {generation:02d} FAILED after {_FormatDuration(elapsed)} with code {result.returncode}"
            )
            return result.returncode

        print(f"Gen {generation:02d} completed in {_FormatDuration(elapsed)}")

    total_elapsed = time.time() - total_start
    print("\nAll requested generations completed.")
    print(f"Total runtime: {_FormatDuration(total_elapsed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
