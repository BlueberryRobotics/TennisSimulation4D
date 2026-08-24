import duckdb
import argparse
import os

# this version is not in use
# this version has the winShotCount field and averages it across generations


def _HasParquetColumn(connection, parquetPath: str, columnName: str) -> bool:
    try:
        schemaRows = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [parquetPath],
        ).fetchall()
    except Exception:
        return False

    targetName = str(columnName).strip().lower()
    for schemaRow in schemaRows:
        if str(schemaRow[0]).strip().lower() == targetName:
            return True
    return False

def main(
    input_pattern,
    output_file,
    previous_file=None,
    min_prev_count=1,
    prior_count_cap=50,
    min_win_pct=0.2,
    threads=4,
    memory_limit="8GB",
    preserve_insertion_order=True,
):
    conn = duckdb.connect()

    conn.execute(f"PRAGMA memory_limit='{memory_limit}'")
    conn.execute(f"PRAGMA threads={threads}")
    if not preserve_insertion_order:
        conn.execute("PRAGMA preserve_insertion_order=false")

    tmp_dir = "duckdb_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    conn.execute("PRAGMA temp_directory='duckdb_tmp'")

    inputHasWinner = _HasParquetColumn(conn, input_pattern, "winner")
    inputWinnerExpr = "COALESCE(winner, FALSE) AS winner" if inputHasWinner else "FALSE AS winner"

    previousWinnerExpr = "FALSE AS winner"
    if previous_file:
        previousHasWinner = _HasParquetColumn(conn, previous_file, "winner")
        previousWinnerExpr = "COALESCE(winner, FALSE) AS winner" if previousHasWinner else "FALSE AS winner"

    # Court bounds
    MIN_COL = 5
    MAX_COL = 10
    MIN_ROW = 5
    MAX_ROW = 22

    # ----------------------------
    # Pre-aggregate new data (including mirrored variants).
    # This keeps the merge step small and avoids regrouping the full prior dataset.
    # ----------------------------
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE new_mirrored_agg AS
        WITH new_data AS (
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
                1 AS count,
                wins,
                {inputWinnerExpr},
                pointShotCount,
                winShotCount,
                initialVelocity,
                airTravelDistance,
                netClearance
            FROM read_parquet('{input_pattern}')
            WHERE
                bounceRow BETWEEN 5 AND 22
                AND
                NOT (interceptRow <= 13 AND bounceRow <= 13)
                AND NOT (interceptRow >= 14 AND bounceRow >= 14)
        ),

        orig_agg AS (
            SELECT
                interceptCol AS ic,
                interceptRow AS ir,
                interceptZ,
                opponentCol AS oc,
                opponentRow AS orow,
                defensiveCol AS dc,
                defensiveRow AS dr,
                bounceCol AS bc,
                bounceRow AS br,
                apexHeight,
                spinTopRpm,
                spinSideRpm,
                SUM(count) AS count,
                SUM(wins) AS wins,
                MAX(CAST(winner AS INTEGER)) = 1 AS winner,
                SUM(pointShotCount * count) / NULLIF(SUM(count), 0) AS pointShotCount,
                SUM(winShotCount * count) / NULLIF(SUM(count), 0) AS winShotCount,
                SUM(initialVelocity * count) / NULLIF(SUM(count), 0) AS initialVelocity,
                SUM(airTravelDistance * count) / NULLIF(SUM(count), 0) AS airTravelDistance,
                SUM(netClearance * count) / NULLIF(SUM(count), 0) AS netClearance
            FROM new_data
            WHERE
                NOT (interceptRow <= 13 AND bounceRow <= 13)
                AND NOT (interceptRow >= 14 AND bounceRow >= 14)
            GROUP BY
                interceptCol, interceptRow, interceptZ,
                opponentCol, opponentRow,
                defensiveCol, defensiveRow,
                bounceCol, bounceRow,
                apexHeight, spinTopRpm, spinSideRpm
        ),

        lr_agg AS (
            SELECT
                ({MIN_COL} + {MAX_COL} - interceptCol) AS ic,
                interceptRow AS ir,
                interceptZ,
                ({MIN_COL} + {MAX_COL} - opponentCol) AS oc,
                opponentRow AS orow,
                ({MIN_COL} + {MAX_COL} - defensiveCol) AS dc,
                defensiveRow AS dr,
                ({MIN_COL} + {MAX_COL} - bounceCol) AS bc,
                bounceRow AS br,
                apexHeight,
                spinTopRpm,
                spinSideRpm,
                SUM(count) AS count,
                SUM(wins) AS wins,
                MAX(CAST(winner AS INTEGER)) = 1 AS winner,
                SUM(pointShotCount * count) / NULLIF(SUM(count), 0) AS pointShotCount,
                SUM(winShotCount * count) / NULLIF(SUM(count), 0) AS winShotCount,
                SUM(initialVelocity * count) / NULLIF(SUM(count), 0) AS initialVelocity,
                SUM(airTravelDistance * count) / NULLIF(SUM(count), 0) AS airTravelDistance,
                SUM(netClearance * count) / NULLIF(SUM(count), 0) AS netClearance
            FROM new_data
            WHERE
                NOT (interceptRow <= 13 AND bounceRow <= 13)
                AND NOT (interceptRow >= 14 AND bounceRow >= 14)
            GROUP BY
                ({MIN_COL} + {MAX_COL} - interceptCol), interceptRow, interceptZ,
                ({MIN_COL} + {MAX_COL} - opponentCol), opponentRow,
                ({MIN_COL} + {MAX_COL} - defensiveCol), defensiveRow,
                ({MIN_COL} + {MAX_COL} - bounceCol), bounceRow,
                apexHeight, spinTopRpm, spinSideRpm
        ),

        tb_agg AS (
            SELECT
                interceptCol AS ic,
                ({MIN_ROW} + {MAX_ROW} - interceptRow) AS ir,
                interceptZ,
                opponentCol AS oc,
                ({MIN_ROW} + {MAX_ROW} - opponentRow) AS orow,
                defensiveCol AS dc,
                ({MIN_ROW} + {MAX_ROW} - defensiveRow) AS dr,
                bounceCol AS bc,
                ({MIN_ROW} + {MAX_ROW} - bounceRow) AS br,
                apexHeight,
                spinTopRpm,
                spinSideRpm,
                SUM(count) AS count,
                SUM(wins) AS wins,
                MAX(CAST(winner AS INTEGER)) = 1 AS winner,
                SUM(pointShotCount * count) / NULLIF(SUM(count), 0) AS pointShotCount,
                SUM(winShotCount * count) / NULLIF(SUM(count), 0) AS winShotCount,
                SUM(initialVelocity * count) / NULLIF(SUM(count), 0) AS initialVelocity,
                SUM(airTravelDistance * count) / NULLIF(SUM(count), 0) AS airTravelDistance,
                SUM(netClearance * count) / NULLIF(SUM(count), 0) AS netClearance
            FROM new_data
            WHERE
                NOT (interceptRow <= 13 AND bounceRow <= 13)
                AND NOT (interceptRow >= 14 AND bounceRow >= 14)
            GROUP BY
                interceptCol, ({MIN_ROW} + {MAX_ROW} - interceptRow), interceptZ,
                opponentCol, ({MIN_ROW} + {MAX_ROW} - opponentRow),
                defensiveCol, ({MIN_ROW} + {MAX_ROW} - defensiveRow),
                bounceCol, ({MIN_ROW} + {MAX_ROW} - bounceRow),
                apexHeight, spinTopRpm, spinSideRpm
        ),

        both_agg AS (
            SELECT
                ({MIN_COL} + {MAX_COL} - interceptCol) AS ic,
                ({MIN_ROW} + {MAX_ROW} - interceptRow) AS ir,
                interceptZ,
                ({MIN_COL} + {MAX_COL} - opponentCol) AS oc,
                ({MIN_ROW} + {MAX_ROW} - opponentRow) AS orow,
                ({MIN_COL} + {MAX_COL} - defensiveCol) AS dc,
                ({MIN_ROW} + {MAX_ROW} - defensiveRow) AS dr,
                ({MIN_COL} + {MAX_COL} - bounceCol) AS bc,
                ({MIN_ROW} + {MAX_ROW} - bounceRow) AS br,
                apexHeight,
                spinTopRpm,
                spinSideRpm,
                SUM(count) AS count,
                SUM(wins) AS wins,
                MAX(CAST(winner AS INTEGER)) = 1 AS winner,
                SUM(pointShotCount * count) / NULLIF(SUM(count), 0) AS pointShotCount,
                SUM(winShotCount * count) / NULLIF(SUM(count), 0) AS winShotCount,
                SUM(initialVelocity * count) / NULLIF(SUM(count), 0) AS initialVelocity,
                SUM(airTravelDistance * count) / NULLIF(SUM(count), 0) AS airTravelDistance,
                SUM(netClearance * count) / NULLIF(SUM(count), 0) AS netClearance
            FROM new_data
            WHERE
                NOT (interceptRow <= 13 AND bounceRow <= 13)
                AND NOT (interceptRow >= 14 AND bounceRow >= 14)
            GROUP BY
                ({MIN_COL} + {MAX_COL} - interceptCol), ({MIN_ROW} + {MAX_ROW} - interceptRow), interceptZ,
                ({MIN_COL} + {MAX_COL} - opponentCol), ({MIN_ROW} + {MAX_ROW} - opponentRow),
                ({MIN_COL} + {MAX_COL} - defensiveCol), ({MIN_ROW} + {MAX_ROW} - defensiveRow),
                ({MIN_COL} + {MAX_COL} - bounceCol), ({MIN_ROW} + {MAX_ROW} - bounceRow),
                apexHeight, spinTopRpm, spinSideRpm
        ),

        all_variants AS (
            SELECT * FROM orig_agg
            UNION ALL
            SELECT * FROM lr_agg
            UNION ALL
            SELECT * FROM tb_agg
            UNION ALL
            SELECT * FROM both_agg
        )
        SELECT
            ic, ir, interceptZ,
            oc, orow,
            dc, dr,
            bc, br,
            apexHeight,
            spinTopRpm,
            spinSideRpm,
            SUM(count) AS count,
            SUM(wins) AS wins,
            MAX(CAST(winner AS INTEGER)) = 1 AS winner,
            SUM(pointShotCount * count) / NULLIF(SUM(count), 0) AS pointShotCount,
            SUM(winShotCount * count) / NULLIF(SUM(count), 0) AS winShotCount,
            SUM(initialVelocity * count) / NULLIF(SUM(count), 0) AS initialVelocity,
            SUM(airTravelDistance * count) / NULLIF(SUM(count), 0) AS airTravelDistance,
            SUM(netClearance * count) / NULLIF(SUM(count), 0) AS netClearance
        FROM all_variants
        GROUP BY
            ic, ir, interceptZ,
            oc, orow,
            dc, dr,
            bc, br,
            apexHeight,
            spinTopRpm,
            spinSideRpm
    """)

    if previous_file:
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE previous_data AS
            SELECT
                interceptCol AS ic,
                interceptRow AS ir,
                interceptZ,
                opponentCol AS oc,
                opponentRow AS orow,
                defensiveCol AS dc,
                defensiveRow AS dr,
                bounceCol AS bc,
                bounceRow AS br,
                apexHeight,
                spinTopRpm,
                spinSideRpm,
                LEAST(count, {int(prior_count_cap)}) AS count,
                CASE
                    WHEN count > {int(prior_count_cap)} THEN
                        ROUND((wins * {float(prior_count_cap)}) / count, 2)
                    ELSE
                        CAST(wins AS DOUBLE)
                END AS wins,
                {previousWinnerExpr},
                avgPointShotCount AS pointShotCount,
                avgWinShotCount AS winShotCount,
                initialVelocity,
                airTravelDistance,
                netClearance
            FROM read_parquet('{previous_file}')
            WHERE
                count >= {min_prev_count}
                -- Keep prior all-loss rows; dropping them biases carry-forward toward apparent perfect records.
                AND bounceRow BETWEEN 5 AND 22
                AND NOT (interceptRow <= 13 AND bounceRow <= 13)
                AND NOT (interceptRow >= 14 AND bounceRow >= 14)
        """)

        conn.execute("""
            CREATE OR REPLACE TEMP TABLE merged_existing AS
            SELECT
                p.ic AS interceptCol,
                p.ir AS interceptRow,
                p.interceptZ,
                p.oc AS opponentCol,
                p.orow AS opponentRow,
                p.dc AS defensiveCol,
                p.dr AS defensiveRow,
                p.bc AS bounceCol,
                p.br AS bounceRow,
                p.apexHeight,
                p.spinTopRpm,
                p.spinSideRpm,
                (COALESCE(p.winner, FALSE) OR COALESCE(n.winner, FALSE)) AS winner,
                (p.count + COALESCE(n.count, 0)) AS count,
                (p.wins + COALESCE(n.wins, 0)) AS wins,
                (
                    (p.pointShotCount * p.count)
                    + COALESCE(n.pointShotCount * n.count, 0)
                ) / NULLIF(p.count + COALESCE(n.count, 0), 0) AS avgPointShotCount,
                (
                    (p.winShotCount * p.count)
                    + COALESCE(n.winShotCount * n.count, 0)
                ) / NULLIF(p.count + COALESCE(n.count, 0), 0) AS avgWinShotCount,
                (
                    (p.initialVelocity * p.count)
                    + COALESCE(n.initialVelocity * n.count, 0)
                ) / NULLIF(p.count + COALESCE(n.count, 0), 0) AS initialVelocity,
                (
                    (p.airTravelDistance * p.count)
                    + COALESCE(n.airTravelDistance * n.count, 0)
                ) / NULLIF(p.count + COALESCE(n.count, 0), 0) AS airTravelDistance,
                (
                    (p.netClearance * p.count)
                    + COALESCE(n.netClearance * n.count, 0)
                ) / NULLIF(p.count + COALESCE(n.count, 0), 0) AS netClearance
            FROM previous_data p
            LEFT JOIN new_mirrored_agg n
                ON p.ic = n.ic
                AND p.ir = n.ir
                AND p.interceptZ = n.interceptZ
                AND p.oc = n.oc
                AND p.orow = n.orow
                AND p.dc = n.dc
                AND p.dr = n.dr
                AND p.bc = n.bc
                AND p.br = n.br
                AND p.apexHeight = n.apexHeight
                AND p.spinTopRpm = n.spinTopRpm
                AND p.spinSideRpm = n.spinSideRpm
        """)

        conn.execute("""
            CREATE OR REPLACE TEMP TABLE new_only AS
            SELECT
                n.ic AS interceptCol,
                n.ir AS interceptRow,
                n.interceptZ,
                n.oc AS opponentCol,
                n.orow AS opponentRow,
                n.dc AS defensiveCol,
                n.dr AS defensiveRow,
                n.bc AS bounceCol,
                n.br AS bounceRow,
                n.apexHeight,
                n.spinTopRpm,
                n.spinSideRpm,
                COALESCE(n.winner, FALSE) AS winner,
                n.count,
                n.wins,
                n.pointShotCount AS avgPointShotCount,
                n.winShotCount AS avgWinShotCount,
                n.initialVelocity,
                n.airTravelDistance,
                n.netClearance
            FROM new_mirrored_agg n
            LEFT JOIN previous_data p
                ON p.ic = n.ic
                AND p.ir = n.ir
                AND p.interceptZ = n.interceptZ
                AND p.oc = n.oc
                AND p.orow = n.orow
                AND p.dc = n.dc
                AND p.dr = n.dr
                AND p.bc = n.bc
                AND p.br = n.br
                AND p.apexHeight = n.apexHeight
                AND p.spinTopRpm = n.spinTopRpm
                AND p.spinSideRpm = n.spinSideRpm
            WHERE p.ic IS NULL
        """)

        conn.execute(f"""
            COPY (
                SELECT
                    merged.*,
                    ROUND(
                        CASE
                            WHEN merged.winner THEN 1.0
                            WHEN merged.count >= 50 THEN (merged.wins * 1.0 / merged.count)
                            ELSE 0.5 + (((merged.wins * 1.0 / merged.count) - 0.5) * POWER((merged.count * 1.0 / 50.0), 2))
                        END,
                        4
                    ) AS adjWinPct
                FROM (
                    SELECT * FROM merged_existing
                    UNION ALL
                    SELECT * FROM new_only
                ) AS merged
            )
            TO '{output_file}' (FORMAT PARQUET)
        """)
    else:
        conn.execute(f"""
            COPY (
                SELECT
                    ic AS interceptCol,
                    ir AS interceptRow,
                    interceptZ,
                    oc AS opponentCol,
                    orow AS opponentRow,
                    dc AS defensiveCol,
                    dr AS defensiveRow,
                    bc AS bounceCol,
                    br AS bounceRow,
                    apexHeight,
                    spinTopRpm,
                    spinSideRpm,
                    count,
                    wins,
                    winner,
                    pointShotCount AS avgPointShotCount,
                    winShotCount AS avgWinShotCount,
                    initialVelocity,
                    airTravelDistance,
                    netClearance,
                    ROUND(
                        CASE
                            WHEN winner THEN 1.0
                            WHEN count >= 50 THEN (wins * 1.0 / count)
                            ELSE 0.5 + (((wins * 1.0 / count) - 0.5) * POWER((count * 1.0 / 50.0), 2))
                        END,
                        4
                    ) AS adjWinPct
                FROM new_mirrored_agg
            )
            TO '{output_file}' (FORMAT PARQUET)
        """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputPattern")
    parser.add_argument("outputFile")
    parser.add_argument("--previousFile", default=None)
    parser.add_argument("--minPrevCount", type=int, default=2)
    parser.add_argument("--priorCountCap", type=int, default=50)
    parser.add_argument("--minWinPct", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memoryLimit", default="8GB")
    parser.add_argument("--disableInsertionOrder", action="store_true")

    args = parser.parse_args()

    main(
        args.inputPattern,
        args.outputFile,
        args.previousFile,
        args.minPrevCount,
        args.priorCountCap,
        args.minWinPct,
        args.threads,
        args.memoryLimit,
        not args.disableInsertionOrder,
    )


   
    # Usage example:
    # Previous file is previous generation, and minPrevCount filters to only include tactics that had at least that many samples 
    # in the previous generation, to ensure we have a more stable estimate of their win percentage. Adjust as needed.
    # python ConsolidateResults006.py shots_*.parquet ConsolidatedGen2_006.parquet --previousFile ConsolidatedGen1_006.parquet --minPrevCount 1 --minWinPct 0.2
    # for Gen0 there is no previous file, so leave it off

    # Just directly querying Gen1 (not mirrored) the only TrajICs that have more than 10 counts with greater than 50% winpct are serves. 
    # We only start to see non-serve TrajICs at a count of 4 with 1. 
    # At a count of 3 there are 153. 
    # At a count of 2 there are 137,623 non-serves with a greater than 50% winpct. 

    # With the mirrored version querying Gen0 (first you have remove row 22 now, because you have serves from the other end)
    # At a count of 4, there are 41
    # At a count of 3, there are 6,237
    # At a count of 2, there are 1,719,822 non-serves with a greater than 50% winpct.