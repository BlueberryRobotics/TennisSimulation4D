#!/usr/bin/env python3
"""
Analyze a gameplay SQLite DB for a specific context in the references table.

Features:
- Count rows for a given context:
  interceptCol, interceptRow, interceptZ, opponentCol, opponentRow
- Filter by bounceCol and bounceRow ranges
- Optional downhillSpeed filter
- Optional CSV export of matching rows

Example:
python IterativeSimulation/AnalyzeSQLiteDB.py \
  --db TrajectoryGenerator/Trajectory007GamePlay30.db \
  --intercept-col 8 --intercept-row 5 --intercept-z 2.7 \
  --opponent-col 5 --opponent-row 23 \
  --downhill-speed 0.0 \
  --export-csv IterativeSimulation/context_8_5_2p7_5_23_0p0.csv
"""

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import List, Sequence, Tuple

# downhillSpeeds are in m/s
AllowedDownhillSpeeds = {0.0, 31.2928, 37.9984, 44.704, 51.405, 58.1152}
ReferencesTableName = '"references"'


def _build_filters(args: argparse.Namespace) -> Tuple[List[str], List[float]]:
    where_parts: List[str] = [
        "interceptCol = ?",
        "interceptRow = ?",
        "opponentCol = ?",
        "opponentRow = ?",
    ]
    parameters: List[float] = [
        int(args.intercept_col),
        int(args.intercept_row),
        int(args.opponent_col),
        int(args.opponent_row),
    ]

    if args.use_tolerance:
        where_parts.append("interceptZ BETWEEN ? AND ?")
        parameters.extend(
            [
                float(args.intercept_z) - float(args.intercept_z_tolerance),
                float(args.intercept_z) + float(args.intercept_z_tolerance),
            ]
        )
    else:
        where_parts.append("interceptZ = ?")
        parameters.append(float(args.intercept_z))

    if args.downhill_speed is not None:
        if args.use_tolerance:
            where_parts.append("COALESCE(downhillSpeed, 0.0) BETWEEN ? AND ?")
            parameters.extend(
                [
                    float(args.downhill_speed) - float(args.downhill_speed_tolerance),
                    float(args.downhill_speed) + float(args.downhill_speed_tolerance),
                ]
            )
        else:
            where_parts.append("COALESCE(downhillSpeed, 0.0) = ?")
            parameters.append(float(args.downhill_speed))

    bounceColMin = int(args.bounce_col_min) if args.bounce_col_min is not None else 5
    bounceColMax = int(args.bounce_col_max) if args.bounce_col_max is not None else 10
    bounceRowMin = int(args.bounce_row_min) if args.bounce_row_min is not None else 5
    bounceRowMax = int(args.bounce_row_max) if args.bounce_row_max is not None else 22

    where_parts.append("bounceCol BETWEEN ? AND ?")
    parameters.extend([bounceColMin, bounceColMax])

    where_parts.append("bounceRow BETWEEN ? AND ?")
    parameters.extend([bounceRowMin, bounceRowMax])

    return where_parts, parameters


def _export_csv(rows: Sequence[sqlite3.Row], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        # Write an empty file with no header when there are no matches.
        csv_path.write_text("", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row[h] for h in headers])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze references table by gameplay context.")
    parser.add_argument("--db", required=True, type=Path, help="Path to SQLite DB file")

    parser.add_argument("--intercept-col", required=True, type=int)
    parser.add_argument("--intercept-row", required=True, type=int)
    parser.add_argument("--intercept-z", required=True, type=float)
    parser.add_argument("--opponent-col", required=True, type=int)
    parser.add_argument("--opponent-row", required=True, type=int)

    parser.add_argument(
        "--bounce-col-min",
        type=int,
        default=None,
        help="Optional minimum bounceCol (default if omitted: 5)",
    )
    parser.add_argument(
        "--bounce-col-max",
        type=int,
        default=None,
        help="Optional maximum bounceCol (default if omitted: 10)",
    )
    parser.add_argument(
        "--bounce-row-min",
        type=int,
        default=None,
        help="Optional minimum bounceRow (default if omitted: 5)",
    )
    parser.add_argument(
        "--bounce-row-max",
        type=int,
        default=None,
        help="Optional maximum bounceRow (default if omitted: 22)",
    )

    parser.add_argument(
        "--use-tolerance",
        action="store_true",
        help="Use tolerance-based numeric filters for interceptZ/downhillSpeed instead of exact equals.",
    )

    parser.add_argument(
        "--intercept-z-tolerance",
        type=float,
        default=0.0001,
        help="Tolerance for interceptZ match when --use-tolerance is set. Default: 0.0001",
    )

    parser.add_argument(
        "--downhill-speed",
        type=float,
        default=None,
        help="Optional downhillSpeed filter. If omitted, downhillSpeed is not filtered.",
    )
    parser.add_argument(
        "--downhill-speed-tolerance",
        type=float,
        default=0.0001,
        help="Tolerance for downhillSpeed match when --downhill-speed and --use-tolerance are set. Default: 0.0001",
    )

    parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Optional path to write matching rows as CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for exported rows (and console preview).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print up to 10 matching rows to console.",
    )

    args = parser.parse_args()

    if not args.db.exists():
        raise FileNotFoundError(f"SQLite DB not found: {args.db}")

    if args.downhill_speed is not None and float(args.downhill_speed) not in AllowedDownhillSpeeds:
        print(
            "Warning: downhillSpeed value is not in expected discrete set "
            f"{sorted(AllowedDownhillSpeeds)}: {args.downhill_speed}"
        )

    bounceColMin = int(args.bounce_col_min) if args.bounce_col_min is not None else 5
    bounceColMax = int(args.bounce_col_max) if args.bounce_col_max is not None else 10
    bounceRowMin = int(args.bounce_row_min) if args.bounce_row_min is not None else 5
    bounceRowMax = int(args.bounce_row_max) if args.bounce_row_max is not None else 22

    if bounceColMin > bounceColMax:
        raise ValueError(
            f"Invalid bounceCol range: min {bounceColMin} is greater than max {bounceColMax}"
        )
    if bounceRowMin > bounceRowMax:
        raise ValueError(
            f"Invalid bounceRow range: min {bounceRowMin} is greater than max {bounceRowMax}"
        )

    where_parts, parameters = _build_filters(args)
    where_sql = " AND ".join(where_parts)

    with sqlite3.connect(str(args.db)) as connection:
        connection.row_factory = sqlite3.Row

        count_sql = f"SELECT COUNT(*) FROM {ReferencesTableName} WHERE {where_sql}"
        matched_count = connection.execute(count_sql, parameters).fetchone()[0]

        print("Context filter:")
        print(
            "  interceptCol=", args.intercept_col,
            " interceptRow=", args.intercept_row,
            " interceptZ=", args.intercept_z,
            " opponentCol=", args.opponent_col,
            " opponentRow=", args.opponent_row,
            sep="",
        )
        if args.use_tolerance:
            print(f"  interceptZ tolerance=+/-{args.intercept_z_tolerance}")
        else:
            print("  interceptZ matching=exact")
        if args.downhill_speed is None:
            print("  downhillSpeed filter=none")
        else:
            if args.use_tolerance:
                print(
                    "  downhillSpeed=", args.downhill_speed,
                    f" (tolerance +/-{args.downhill_speed_tolerance})",
                    sep="",
                )
            else:
                print("  downhillSpeed=", args.downhill_speed, " (exact)", sep="")

        print(f"  bounceCol range={bounceColMin}..{bounceColMax}")
        print(f"  bounceRow range={bounceRowMin}..{bounceRowMax}")

        print(f"Matched rows in references: {matched_count}")

        needs_rows = bool(args.export_csv) or args.preview
        if not needs_rows:
            return

        select_sql = f"SELECT * FROM {ReferencesTableName} WHERE {where_sql} ORDER BY rowid"
        select_params: List[float] = list(parameters)

        if args.limit is not None and args.limit > 0:
            select_sql += " LIMIT ?"
            select_params.append(int(args.limit))

        rows = connection.execute(select_sql, select_params).fetchall()

        if args.preview:
            preview_rows = rows[:10]
            print(f"Preview rows shown: {len(preview_rows)}")
            for index, row in enumerate(preview_rows, start=1):
                print(f"Row {index}: {dict(row)}")

        if args.export_csv:
            _export_csv(rows, args.export_csv)
            print(f"CSV written: {args.export_csv} (rows exported: {len(rows)})")


if __name__ == "__main__":
    main()


# Exact matching (default):
# python AnalyzeSQLiteDB.py --db ../TrajectoryGenerator/Trajectory007GamePlay30.db --intercept-col 8 --intercept-row 5 --intercept-z 2.7 --opponent-col 5 --opponent-row 23 --downhill-speed 0.0

# Tolerance mode (only if needed):
# python AnalyzeSQLiteDB.py --db ../TrajectoryGenerator/Trajectory007GamePlay30.db --intercept-col 8 --intercept-row 5 --intercept-z 2.7 --opponent-col 5 --opponent-row 23 --downhill-speed 0.0 --use-tolerance --intercept-z-tolerance 0.0001 --downhill-speed-tolerance 0.0001

# Blue deuce service box example:
# python AnalyzeSQLiteDB.py --db ../TrajectoryGenerator/Trajectory007GamePlay31.db --intercept-col 8 --intercept-row 5 --intercept-z 2.7 --opponent-col 5 --opponent-row 23 --bounce-col-min 5 --bounce-col-max 7 --bounce-row-min 15 --bounce-row-max 18 --intercept-z-tolerance 0.0001
