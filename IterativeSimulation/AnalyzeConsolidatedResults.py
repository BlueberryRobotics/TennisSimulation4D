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

    # Number with downhillSpeeds
    downhillSpeeds = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE downhillSpeed > 1.0"
    ).fetchone()[0]
    print(f"Rows with downhillSpeeds: {downhillSpeeds}")

    # Number with wins = 0
    wins0 = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE count = 1 AND wins = 0"
    ).fetchone()[0]
    print(f"Rows with wins = 0 and count = 1: {wins0}")

    # Number with count >= 10 AND count < 50
    countgt1 = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE count >= 10 AND count < 50"
    ).fetchone()[0]
    print(f"Rows with count >= 10 and count < 50: {countgt1}")

    # Number with count >= 10 AND count < 50 and winPct >= .5
    countgt10winsgt5 = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE count >= 10 AND count < 50 AND (wins * 1.0 / count) >= 0.5"
    ).fetchone()[0]
    print(f"Rows with count >= 10 and count < 50 and winPct >= .5: {countgt10winsgt5}")

    # Number with count >= 10 and count < 50 and adjWinPct = 1.0
    countgt10winsgt5 = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE count >= 10 AND count < 50 AND adjWinPct = 1.0"
    ).fetchone()[0]
    print(f"Rows with count >= 10 and count < 50 and adjWinPct = 1.0: {countgt10winsgt5}")

    # Number with count >= 50
    countgt1 = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE count >= 50"
    ).fetchone()[0]
    print(f"Rows with count >= 50: {countgt1}")

    # Number with count >= 50 and winPct >= .5
    countgt10winsgt5 = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE count >= 50 AND (wins * 1.0 / count) >= 0.5"
    ).fetchone()[0]
    print(f"Rows with count >= 50 and winPct >= .5: {countgt10winsgt5}")

    # Number with count >= 50 and winPct = 1.0
    countgt10winsgt5 = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE count >= 50 AND adjWinPct = 1.0"
    ).fetchone()[0]
    print(f"Rows with count >= 50 and adjWinPct = 1.0: {countgt10winsgt5}")

    # Number with count >= 12
    countgt12 = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE count >= 12"
    ).fetchone()[0]
    print(f"Rows with count >= 12: {countgt12}")

    # Number with count >= 12 and winPct >= .5
    countgt12winsgt5 = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE count >= 12 AND adjWinPct >= 0.5"
    ).fetchone()[0]
    print(f"Rows with count >= 12 and adjWinPct >= 0.5: {countgt12winsgt5}")

    # Number with winner = True
    countwinnertrue = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE winner = TRUE"
    ).fetchone()[0]
    print(f"Rows with winner = TRUE: {countwinnertrue}")

    # Number with specific serve
    countserve = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE interceptCol = 8 AND interceptRow = 5 AND ROUND(interceptZ, 1) = 2.7 AND opponentRow = 23 AND opponentCol = 5 AND (wins * 1.0 / count) >= 0.5 AND adjWinPct >= 0.5 AND bounceRow BETWEEN 15 AND 18 AND bounceCol BETWEEN 5 AND 7"
    ).fetchone()[0]
    print(f"Rows for serve: {countserve}")

    # Number with initial velocity > 60 mph
    count60mph = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE initialVelocity >= 26.82"
    ).fetchone()[0]
    print(f"Number with velocity > 60 mph: {count60mph}")

    

    # Number with initial velocity > 60 mph and winPct > .5
    count60mph = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE initialVelocity >= 26.82 AND adjWinPct >= 0.5"
    ).fetchone()[0]
    print(f"Number with velocity > 60 mph and adjWinPct > .5: {count60mph}")

    # Number with initial velocity > 80 mph
    count80mph = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE initialVelocity >= 35.76"
    ).fetchone()[0]
    print(f"Number with velocity > 80 mph: {count80mph}")

    # Number with initial velocity > 80 mph and winPct > .5
    count80mph = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE initialVelocity >= 35.76 AND adjWinPct >= 0.5"
    ).fetchone()[0]
    print(f"Number with velocity > 80 mph and adjWinPct > .5: {count80mph}")

    # Number with initial velocity > 100 mph
    count100mph = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE initialVelocity >= 44.7"
    ).fetchone()[0]
    print(f"Number with velocity > 100 mph: {count100mph}")

    # Number with initial velocity > 100 mph and winPct > .5
    count100mph = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE initialVelocity >= 44.7 AND adjWinPct >= 0.5"
    ).fetchone()[0]
    print(f"Number with velocity > 100 mph and adjWinPct > .5: {count100mph}")

    # Number of unique contexts (by intercept/opponent location key) in that qualifying set.
    qualifyingContexts = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                interceptCol,
                interceptRow,
                interceptZ,
                opponentCol,
                opponentRow,
                bounceCol,
                bounceRow
            FROM read_parquet('{parquetFile}')
            WHERE count >= 10
                -- AND (wins * 1.0 / count) >= 0.5
                AND adjWinPct >= 0.5
        ) t
        """
    ).fetchone()[0]
    print(
        f"Unique contexts with count >= 10 and adjWinPct >= .5: {qualifyingContexts}"
    )

    # Number of unique contexts (by intercept/opponent location key) in that qualifying set.
    qualifyingContexts = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                interceptCol,
                interceptRow,
                interceptZ,
                opponentCol,
                opponentRow,
                bounceCol,
                bounceRow,
                downhillSpeed
            FROM read_parquet('{parquetFile}')
            WHERE count >= 10
                -- AND (wins * 1.0 / count) >= 1.0
                AND adjWinPct >= 1.0
        ) t
        """
    ).fetchone()[0]
    print(
        f"Unique contexts with count >= 10 and adjWinPct = 1.0: {qualifyingContexts}"
    )

    # Average count for rows with count > 1
    avg_dup_count = conn.execute(
        f"SELECT AVG(count) FROM read_parquet('{parquetFile}') WHERE count > 1"
    ).fetchone()[0]
    print(f"Average 'count' for duplicated rows: {avg_dup_count:.3f}\n")

    # Number for a specific context with count >= 10 AND count < 50 and winPct >= .5
    specificContext = conn.execute(
        f"""SELECT COUNT(*) FROM read_parquet('{parquetFile}') 
        WHERE 
            interceptCol = 8
            AND interceptRow = 5
            AND ABS(interceptZ - 2.7) < 1e-5
            AND opponentCol = 5
            AND opponentRow = 23
            AND count >= 10 
            -- AND (wins * 1.0 / count) >= 0.5
            AND adjWinPct >= 0.5
        """
    ).fetchone()[0]
    print(f"Winning rows for specific context: {specificContext}")


    # Number for specific first serve 
    specificFirstServe = conn.execute(
        f"""SELECT COUNT(*) FROM read_parquet('{parquetFile}') 
        WHERE 
            (interceptCol = 8)
            AND (interceptRow = 5)
            AND ABS(interceptZ - 3.3) < 1e-5
            AND opponentCol = 5
            AND (opponentRow = 23)
            AND defensiveRow BETWEEN 0 AND 13
        """
    ).fetchone()[0]
    print(f"Winning rows for specific first serve: {specificFirstServe}")


    # Number for first serve without downhillSpeeds
    specificFirstServe = conn.execute(
        f"""SELECT COUNT(*) FROM read_parquet('{parquetFile}') 
        WHERE 
            (interceptCol = 8)
            AND (interceptRow = 5)
            AND ABS(interceptZ - 3.3) < 1e-5
            AND opponentCol = 5
            AND (opponentRow = 23)
            AND downhillSpeed < 1.0
        """
    ).fetchone()[0]
    print(f"Winning rows for specific first serve without downhillSpeed: {specificFirstServe}")


    # Number for first serve with downhillSpeeds
    specificFirstServe = conn.execute(
        f"""SELECT COUNT(*) FROM read_parquet('{parquetFile}') 
        WHERE 
            (interceptCol = 8)
            AND (interceptRow = 5)
            AND ABS(interceptZ - 3.3) < 1e-5
            AND opponentCol = 5
            AND (opponentRow = 23)
            AND downhillSpeed > 1.0
        """
    ).fetchone()[0]
    print(f"Winning rows for specific first serve with downhillSpeed: {specificFirstServe}")

    # Number for first serve with downhillSpeeds and adjWinPct >= .5
    specificFirstServe = conn.execute(
        f"""SELECT COUNT(*) FROM read_parquet('{parquetFile}') 
        WHERE 
            (interceptCol = 8)
            AND (interceptRow = 5)
            AND ABS(interceptZ - 3.3) < 1e-5
            AND opponentCol = 5
            AND (opponentRow = 23)
            AND downhillSpeed > 1.0
            AND adjWinPct >= 0.5
        """
    ).fetchone()[0]
    print(f"Winning rows for specific first serve with downhillSpeed and adjWinPct >= .5: {specificFirstServe}")

    # Number for first serve with downhillSpeeds and adjWinPct >= 1.0
    specificFirstServe = conn.execute(
        f"""SELECT COUNT(*) FROM read_parquet('{parquetFile}') 
        WHERE 
            (interceptCol = 8)
            AND (interceptRow = 5)
            AND ABS(interceptZ - 3.3) < 1e-5
            AND opponentCol = 5
            AND (opponentRow = 23)
            AND downhillSpeed > 1.0
            AND adjWinPct >= 1.0
        """
    ).fetchone()[0]
    print(f"Winning rows for specific first serve with downhillSpeed and adjWinPct >= 1.0: {specificFirstServe}")

    # Number for first serve with downhillSpeeds and adjWinPct >= 1.0
    specificFirstServe = conn.execute(
        f"""SELECT COUNT(*) FROM read_parquet('{parquetFile}') 
        WHERE 
            (interceptCol = 8)
            AND (interceptRow = 5)
            AND ABS(interceptZ - 3.3) < 1e-5
            AND opponentCol = 5
            AND (opponentRow = 23)
            AND downhillSpeed > 1.0
            --AND adjWinPct >= 1.0
            AND winner = TRUE
        """
    ).fetchone()[0]
    print(f"Winning rows for specific first serve with downhillSpeed and winner = TRUE: {specificFirstServe}")


    # Number with winner = TRUE
    countwinnertrue = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquetFile}') WHERE winner = TRUE"
    ).fetchone()[0]
    print(f"Rows with winner = TRUE: {countwinnertrue}")


    # print("=== DISTINCT VALUES PER KEY FIELD ===\n")

    # # Key fields to analyze
    # keyFields = [
    #     "interceptCol","interceptRow","interceptZ",
    #     "opponentCol","opponentRow",
    #     "defensiveCol","defensiveRow",
    #     "bounceCol","bounceRow",
    #     "apexHeight","spinTopRpm","spinSideRpm"
    # ]

    # for field in keyFields:
    #     print(f"\n--- {field} ---")

    #     # Number of distinct values
    #     n = conn.execute(
    #         f"SELECT COUNT(DISTINCT {field}) FROM read_parquet('{parquetFile}')"
    #     ).fetchone()[0]
    #     print(f"Distinct count: {n}")

    #     # Actual distinct values
    #     values = conn.execute(
    #         f"SELECT DISTINCT {field} FROM read_parquet('{parquetFile}') ORDER BY {field}"
    #     ).df()

    #     print("Values:")
    #     print(values.to_string(index=False))

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

    # python AnalyzeConsolidatedResults.py ConsolidatedGen30_008.parquet