#!/usr/bin/env python3
"""
Set winner=True for rows where:
- count is >= 6
- and win rate is 1.0/100%.

Win rate source:
- Prefer winPct if present (supports 1.0 or 100.0 forms)
- Otherwise compute from wins/count when both columns exist

If winner column exists, it is updated in place (existing TRUE values are
preserved). If winner does not exist, it is first added as FALSE for every
row, then qualifying rows are set to TRUE.

The script reads a parquet file with DuckDB, writes updated rows to a temporary
parquet file, then atomically replaces the destination file.
"""

import argparse
import os
from pathlib import Path
from typing import Sequence

import duckdb


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _find_required_column(columns: Sequence[str], target_name: str) -> str:
    lowered = {column_name.lower(): column_name for column_name in columns}
    match = lowered.get(target_name.lower())
    if match is None:
        raise ValueError(f"Input parquet is missing required column: {target_name}")
    return match


def backfill_winner_by_win_pct_and_count(
    input_parquet_path: Path,
    output_parquet_path: Path,
) -> None:
    if not input_parquet_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_parquet_path}")

    output_dir = output_parquet_path.parent
    if str(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    temp_output_path = output_parquet_path.with_suffix(output_parquet_path.suffix + ".tmp")

    connection = duckdb.connect()
    copy_completed = False

    try:
        schema_rows = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [str(input_parquet_path)],
        ).fetchall()
        all_columns = [str(row[0]) for row in schema_rows]

        count_column = _find_required_column(all_columns, "count")
        lower_to_name = {column_name.lower(): column_name for column_name in all_columns}
        win_pct_column = lower_to_name.get("winpct")
        wins_column = lower_to_name.get("wins")
        winner_column = lower_to_name.get("winner")

        if win_pct_column is not None:
            win_rate_is_full_expr = (
                f"TRY_CAST({_quote_ident(win_pct_column)} AS DOUBLE) IN (1.0, 100.0)"
            )
        elif wins_column is not None:
            win_rate_is_full_expr = (
                f"TRY_CAST({_quote_ident(count_column)} AS DOUBLE) > 0 "
                f"AND (TRY_CAST({_quote_ident(wins_column)} AS DOUBLE) "
                f"/ TRY_CAST({_quote_ident(count_column)} AS DOUBLE)) = 1.0"
            )
        else:
            raise ValueError(
                "Input parquet must contain either 'winPct' or both 'wins' and 'count'."
            )

        rule_expr = (
            f"TRY_CAST({_quote_ident(count_column)} AS DOUBLE) >= 25 "
            f"AND ({win_rate_is_full_expr})"
        )

        read_relation_sql = "read_parquet(?)"
        if winner_column is None:
            quoted_source_columns = ",\n        ".join(_quote_ident(column_name) for column_name in all_columns)
            read_relation_sql = (
                "(\n"
                "    WITH base_with_winner AS (\n"
                "        SELECT\n"
                f"        {quoted_source_columns},\n"
                "        FALSE AS winner\n"
                "        FROM read_parquet(?)\n"
                "    )\n"
                "    SELECT * FROM base_with_winner\n"
                ")"
            )
            winner_column = "winner"

        projected_columns = []
        for column_name in all_columns:
            if winner_column is not None and column_name.lower() == winner_column.lower():
                continue
            quoted_column = _quote_ident(column_name)
            projected_columns.append(quoted_column)

        projected_columns.append(
            "CASE "
            f"WHEN {rule_expr} "
            "THEN TRUE "
            f"ELSE COALESCE(TRY_CAST({_quote_ident(winner_column)} AS BOOLEAN), FALSE) "
            "END AS winner"
        )

        select_sql = "SELECT\n    " + ",\n    ".join(projected_columns) + f"\nFROM {read_relation_sql}"
        temp_output_path_sql = str(temp_output_path).replace("\\", "/")

        connection.execute(
            f"""
            COPY (
                {select_sql}
            )
            TO '{temp_output_path_sql}'
            (FORMAT PARQUET, COMPRESSION 'zstd')
            """,
            [str(input_parquet_path)],
        )
        copy_completed = True

    finally:
        connection.close()
        if not copy_completed and temp_output_path.exists():
            temp_output_path.unlink()

    os.replace(temp_output_path, output_parquet_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Set winner=True when count is >= 25 and win rate is 1.0/100, "
            "using winPct if present or wins/count otherwise."
        ),
    )
    parser.add_argument(
        "parquetPath",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("ConsolidatedGen23.parquet"),
        help="Input parquet path (default: ConsolidatedGen23.parquet).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output parquet path. Defaults to in-place update of parquetPath.",
    )

    args = parser.parse_args()

    input_path = args.parquetPath
    output_path = args.output if args.output is not None else input_path

    backfill_winner_by_win_pct_and_count(
        input_parquet_path=input_path,
        output_parquet_path=output_path,
    )

    print(f"Wrote updated parquet: {output_path}")


if __name__ == "__main__":
    main()

# python BackfillWinnerByWinPctAndCount.py Gen23Reference.parquet
# This script creates a winner field if it doesn't exist
# and sets it to true for count >= 25 and winPct = 1.0 
# (which may not always be accurate)
# This script falls short in that it does not set the adjWinPct to 1.0
# which is needed for the Trajectory Explorer to work correctly
# Use the BackfillAdjWinPctByWinner.py to do that after running this one