import duckdb
import argparse
import pandas as pd
import os
import sys

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from CourtPlayerSettings import Court


def ResolveNetBoundaryRow(court: Court) -> int:
    gridRowsValue = int(getattr(court, "gridRows", 26))
    if hasattr(court, "GetRowCenterY") and hasattr(court, "netY"):
        northRows = [
            rowIndex
            for rowIndex in range(1, gridRowsValue + 1)
            if float(court.GetRowCenterY(rowIndex)) < float(court.netY)
        ]
        if northRows:
            return int(max(northRows))
    return int(max(1, gridRowsValue // 2))

def analyze(parquetFile, courtGeometryMode="uniform"):
    conn = duckdb.connect()
    court = Court(geometryMode=courtGeometryMode)
    netBoundaryRow = ResolveNetBoundaryRow(court)
    firstSouthRow = int(netBoundaryRow + 1)

    print("\n=== BASIC COUNTS ===\n")

    # Total rows
    total = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}')"
    ).fetchone()[0]
    print(f"Total rows in file: {total}")

    incorrectPlayerPoints = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE interceptRow <= {netBoundaryRow} AND bounceRow <= {netBoundaryRow}"
    ).fetchone()[0]
    print(f"Player Rows that are incorrect: {incorrectPlayerPoints}")

    incorrectOpponentPoints = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE interceptRow >= {firstSouthRow} AND bounceRow >= {firstSouthRow}"
    ).fetchone()[0]
    print(f"Opponent Rows that are incorrect: {incorrectOpponentPoints}")

    # Number with count = 1
    count1 = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE count = 1"
    ).fetchone()[0]
    print(f"Rows with count = 1: {count1}")

    # Number with count >= 10 
    countgt1 = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE count > 10"
    ).fetchone()[0]
    print(f"Rows with count >= 10: {countgt1}")

    # Average count for rows with count > 1
    avg_dup_count = conn.execute(
        f"SELECT AVG(count) FROM read_parquet('{parquetFile}') WHERE count > 1"
    ).fetchone()[0]
    print(f"Average 'count' for duplicated rows: {avg_dup_count:.3f}\n")

    print("=== DISTINCT VALUES PER KEY FIELD ===\n")

    # Key fields to analyze
    keyFields = [
        "interceptCol","interceptRow","interceptZ",
        "opponentCol","opponentRow",
        "defensiveCol","defensiveRow",
        "bounceCol","bounceRow",
        "apexHeight","spinTopRpm","spinSideRpm"
    ]

    for field in keyFields:
        print(f"\n--- {field} ---")

        # Number of distinct values
        n = conn.execute(
            f"SELECT COUNT(DISTINCT {field}) FROM read_parquet('{parquetFile}')"
        ).fetchone()[0]
        print(f"Distinct count: {n}")

        # Actual distinct values
        values = conn.execute(
            f"SELECT DISTINCT {field} FROM read_parquet('{parquetFile}') ORDER BY {field}"
        ).df()

        print("Values:")
        print(values.to_string(index=False))

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze consolidated parquet file.")
    parser.add_argument("parquetFile", help="Path to consolidated parquet file")
    parser.add_argument(
        "--courtGeometryMode",
        type=str,
        default="uniform",
        choices=["uniform", "short_rows_13_14"],
        help="Court geometry mode used to derive north/south row split.",
    )
    args = parser.parse_args()
    analyze(args.parquetFile, args.courtGeometryMode)