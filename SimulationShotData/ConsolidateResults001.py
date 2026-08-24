# ConsolidateResultsStreaming.py
#
# Safely consolidate very large Parquet datasets (100M–1B+ rows)
# without loading everything into RAM.
#
# Key properties:
# - Never calls .df()
# - Uses DuckDB streaming + disk spill
# - Processes ALL input files in one pass
# - Writes a single consolidated Parquet output
# - Safe on laptops / limited RAM machines
#
# Usage:
#   python ConsolidateResultsStreaming.py "shots_*.parquet" consolidated.parquet

import duckdb
import argparse
import os


def main(input_pattern: str, output_file: str):
    # ------------------------------------------------------------
    # DuckDB setup: force safe, disk-backed execution
    # ------------------------------------------------------------
    conn = duckdb.connect()

    # Hard memory cap so the OS never gets killed
    conn.execute("PRAGMA memory_limit='4GB'")

    # Force all large intermediates to spill to disk
    tmp_dir = "duckdb_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    conn.execute(f"PRAGMA temp_directory='{tmp_dir}'")

    # Optional but recommended: reduce threads to avoid RAM spikes
    conn.execute("PRAGMA threads=4")

    print("Starting streaming consolidation…")

    # ------------------------------------------------------------
    # Streaming aggregation + write directly to Parquet
    # ------------------------------------------------------------
    conn.execute(f"""
        COPY (
            SELECT
                interceptCol,
                interceptRow,
                interceptZ,
                opponentCol,
                opponentRow,
                defensiveCol,
                defensiveRow,
                bounceCol,
                bounceRow,
                apexHeight,
                spinTopRpm,
                spinSideRpm,
                SUM(count) AS count,
                SUM(wins) AS wins
            FROM read_parquet('{input_pattern}')
            GROUP BY
                interceptCol,
                interceptRow,
                interceptZ,
                opponentCol,
                opponentRow,
                defensiveCol,
                defensiveRow,
                bounceCol,
                bounceRow,
                apexHeight,
                spinTopRpm,
                spinSideRpm
        )
        TO '{output_file}'
        (FORMAT PARQUET)
    """)

    print(f"Consolidation complete: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream-safe consolidation of massive Parquet simulation outputs"
    )
    parser.add_argument(
        "inputPattern",
        help="Glob pattern for input parquet files (e.g. shots_*.parquet)"
    )
    parser.add_argument(
        "outputFile",
        help="Output consolidated parquet file"
    )

    args = parser.parse_args()
    main(args.inputPattern, args.outputFile)
