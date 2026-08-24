#!/usr/bin/env python3
"""
Add (or recompute) adjWinPct in a GenXReference parquet file.

Rule:
- count >= 10: adjWinPct = winPct = wins / count
- 1 <= count <= 9: adjWinPct = 0.5 + ((winPct - 0.5) * (count / 10)^2)

The script processes rows in batches, computes adjWinPct row-by-row in Python,
then writes to a temp parquet and atomically replaces the target output path.
"""

import argparse
import math
import os
from pathlib import Path
from typing import List, Sequence

import duckdb


def _quote_ident(identifier: str) -> str:
	# DuckDB supports double-quoted identifiers with escaped quotes.
	return '"' + identifier.replace('"', '""') + '"'


def _compute_adj_win_pct(wins_value, count_value) -> float:
	wins = float(wins_value)
	count = float(count_value)

	if count <= 0.0:
		return 0.5

	win_pct = wins / count
	if count >= 10.0:
		return float(win_pct)

	return float(0.5 + ((win_pct - 0.5) * math.pow((count / 10.0), 2)))


def _find_required_column(columns: Sequence[str], target_name: str) -> str:
	lowered = {c.lower(): c for c in columns}
	if target_name.lower() not in lowered:
		raise ValueError(f"Input parquet is missing required column: {target_name}")
	return lowered[target_name.lower()]


def add_adjusted_win_pct(
	input_parquet_path: Path,
	output_parquet_path: Path,
	batch_size: int = 100_000,
) -> None:
	if not input_parquet_path.exists():
		raise FileNotFoundError(f"Input parquet not found: {input_parquet_path}")

	output_dir = output_parquet_path.parent
	if str(output_dir):
		output_dir.mkdir(parents=True, exist_ok=True)

	temp_output_path = output_parquet_path.with_suffix(output_parquet_path.suffix + ".tmp")

	reader_connection = duckdb.connect()
	writer_connection = duckdb.connect()
	copy_completed = False

	try:
		schema_rows = reader_connection.execute(
			"DESCRIBE SELECT * FROM read_parquet(?)",
			[str(input_parquet_path)],
		).fetchall()
		all_columns = [str(row[0]) for row in schema_rows]

		wins_column = _find_required_column(all_columns, "wins")
		count_column = _find_required_column(all_columns, "count")

		# If adjWinPct already exists, rebuild it from wins/count.
		source_columns: List[str] = [c for c in all_columns if c.lower() != "adjwinpct"]

		if not source_columns:
			raise ValueError("No source columns found in parquet.")

		quoted_source_columns = ", ".join(_quote_ident(c) for c in source_columns)
		source_select_sql = (
			f"SELECT {_quote_ident(wins_column)} AS __wins, "
			f"{_quote_ident(count_column)} AS __count, "
			f"{quoted_source_columns} FROM read_parquet(?)"
		)

		# Build destination table shape from input columns + adjWinPct.
		writer_connection.execute(
			f"""
			CREATE TEMP TABLE output_rows AS
			SELECT {quoted_source_columns}, CAST(NULL AS DOUBLE) AS adjWinPct
			FROM read_parquet(?)
			LIMIT 0
			""",
			[str(input_parquet_path)],
		)

		output_columns = source_columns + ["adjWinPct"]
		quoted_output_columns = ", ".join(_quote_ident(c) for c in output_columns)
		placeholders = ", ".join("?" for _ in output_columns)
		insert_sql = f"INSERT INTO output_rows ({quoted_output_columns}) VALUES ({placeholders})"

		cursor = reader_connection.execute(source_select_sql, [str(input_parquet_path)])

		processed = 0
		while True:
			batch = cursor.fetchmany(int(batch_size))
			if not batch:
				break

			rows_to_insert = []
			for row in batch:
				if len(row) < 2:
					raise ValueError(f"Unexpected row shape from parquet reader: {row}")

				wins_value = row[0]
				count_value = row[1]
				row_values = list(row[2:])
				adj_win_pct = round(_compute_adj_win_pct(wins_value, count_value), 4)
				row_values.append(adj_win_pct)
				rows_to_insert.append(tuple(row_values))

			writer_connection.executemany(insert_sql, rows_to_insert)
			processed += len(rows_to_insert)

			if processed % 10_000 == 0:
				print(f"Processed rows: {processed:,}", flush=True)

		temp_output_path_sql = str(temp_output_path).replace("\\", "/")
		writer_connection.execute(
			f"""
			COPY (SELECT * FROM output_rows)
			TO '{temp_output_path_sql}'
			(FORMAT PARQUET, COMPRESSION 'zstd')
			"""
		)
		copy_completed = True

	finally:
		try:
			reader_connection.close()
		finally:
			try:
				writer_connection.close()
			finally:
				if not copy_completed and temp_output_path.exists():
					temp_output_path.unlink()

	os.replace(temp_output_path, output_parquet_path)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Add/recompute adjWinPct in a parquet file.",
	)
	parser.add_argument(
		"parquetPath",
		type=Path,
		help="Path to input GenXReference parquet file.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Optional output parquet path (defaults to in-place replace of input file).",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=100_000,
		help="Rows fetched per batch (default: 100000).",
	)

	args = parser.parse_args()

	input_path = args.parquetPath
	output_path = args.output if args.output is not None else input_path

	add_adjusted_win_pct(
		input_parquet_path=input_path,
		output_parquet_path=output_path,
		batch_size=max(1, int(args.batch_size)),
	)

	print(f"Wrote parquet with adjWinPct: {output_path}")


if __name__ == "__main__":
	main()
