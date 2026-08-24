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

def InspectParquet(parquetPath, limit=20, filterExpr=None, courtGeometryMode="uniform"):
    """
    Inspect rows inside a Parquet shot-results file.

    Args:
        parquetPath: path to a .parquet file or a glob ("shots_*.parquet")
        limit: number of rows to show
        filterExpr: optional SQL WHERE clause (string)
    """

    if not parquetPath.endswith(".parquet") and "*" not in parquetPath:
        raise ValueError("Please provide a .parquet file or a wildcard pattern.")

    print(f"\n=== Inspecting Parquet: {parquetPath} ===\n")

    conn = duckdb.connect()
    court = Court(geometryMode=courtGeometryMode)
    netBoundaryRowBlue = 13 # ResolveNetBoundaryRow(court)
    netBoundaryRowRed = 14

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)       # super wide terminal
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.expand_frame_repr", False)

    # Count rows
    countQuery = f"SELECT COUNT(*) AS row_count FROM read_parquet('{parquetPath}')"
    total = conn.execute(countQuery).fetchone()[0]
    print(f"Total rows: {total}\n")

    incorrect = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{parquetPath}')
        WHERE (
            (interceptRow <= {netBoundaryRowBlue} AND bounceRow <= {netBoundaryRowBlue})
            OR
            (interceptRow >= {netBoundaryRowRed} AND bounceRow >= {netBoundaryRowRed})
        )
        """
    ).fetchone()[0]
    print(f"Rows that have the bounce side incorrect: {incorrect}")

    # Show schema
    schemaQuery = f"DESCRIBE SELECT * FROM read_parquet('{parquetPath}') LIMIT 0"
    print("Schema:")
    print(conn.execute(schemaQuery).df().to_string(index=False))
    print("\n")

    # Show sample rows
    if filterExpr:
        sampleQuery = f"""
            SELECT *
            FROM read_parquet('{parquetPath}')
            WHERE {filterExpr}
            LIMIT {limit}
        """
        print(f"Showing {limit} rows with filter: {filterExpr}\n")
    else:
        sampleQuery = f"""
            SELECT *
            FROM read_parquet('{parquetPath}')
            LIMIT {limit}
        """
        print(f"Showing first {limit} rows:\n")

    df = conn.execute(sampleQuery).df()
    print(df.to_string(index=False))

    print("\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Parquet shot output.")
    parser.add_argument("parquetPath", type=str,
                        help="Path to .parquet file or glob, e.g. shots_20260327_*.parquet")
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of rows to display")
    parser.add_argument("--filter", type=str, default=None,
                        help="Optional SQL WHERE expression")
    parser.add_argument(
        "--courtGeometryMode",
        type=str,
        default="uniform",
        choices=["uniform", "short_rows_13_14"],
        help="Court geometry mode used to derive north/south row split.",
    )

    args = parser.parse_args()
    InspectParquet(args.parquetPath, args.limit, args.filter, args.courtGeometryMode)