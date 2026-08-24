#!/usr/bin/env python3
"""
Set adjWinPct to 1.0 for rows where winner is TRUE.

Behavior:
- Requires a winner column.
- If adjWinPct exists, preserve existing values for non-winner rows.
- If adjWinPct does not exist, add it and set non-winner rows to NULL.

The script reads a parquet file with DuckDB, writes updated rows to a
temporary parquet file, then atomically replaces the destination file.
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


def backfill_adj_win_pct_by_winner(
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

        winner_column = _find_required_column(all_columns, "winner")
        lower_to_name = {column_name.lower(): column_name for column_name in all_columns}
        adj_win_pct_column = lower_to_name.get("adjwinpct")

        projected_columns = []
        for column_name in all_columns:
            if adj_win_pct_column is not None and column_name.lower() == adj_win_pct_column.lower():
                continue
            projected_columns.append(_quote_ident(column_name))

        if adj_win_pct_column is not None:
            projected_columns.append(
                "CASE "
                f"WHEN COALESCE(TRY_CAST({_quote_ident(winner_column)} AS BOOLEAN), FALSE) THEN 1.0 "
                f"ELSE TRY_CAST({_quote_ident(adj_win_pct_column)} AS DOUBLE) "
                "END AS "
                f"{_quote_ident(adj_win_pct_column)}"
            )
        else:
            projected_columns.append(
                "CASE "
                f"WHEN COALESCE(TRY_CAST({_quote_ident(winner_column)} AS BOOLEAN), FALSE) THEN 1.0 "
                "ELSE CAST(NULL AS DOUBLE) "
                "END AS adjWinPct"
            )

        select_sql = "SELECT\n    " + ",\n    ".join(projected_columns) + "\nFROM read_parquet(?)"
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
        description="Set adjWinPct=1.0 for winner rows in a GenXXReference parquet.",
    )
    parser.add_argument(
        "parquetPath",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("Gen23Reference.parquet"),
        help="Input parquet path (default: Gen23Reference.parquet).",
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

    backfill_adj_win_pct_by_winner(
        input_parquet_path=input_path,
        output_parquet_path=output_path,
    )

    print(f"Wrote updated parquet: {output_path}")


if __name__ == "__main__":
    main()

# python BackfillAdjWinPctByWinner.py Gen23Reference.parquet
# python BackfillAdjWinPctByWinner.py ConsolidatedGen23.parquet
# Run BackfillWinnerByWInPctAndCount.py before this one to set the winner field
# this one will then set the adjWinPct field based on winnerc