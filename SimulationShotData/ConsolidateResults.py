import duckdb
import argparse
import os


def main(input_pattern, output_file):
    conn = duckdb.connect()

    conn.execute("PRAGMA memory_limit='4GB'")
    conn.execute("PRAGMA threads=4")

    tmp_dir = "duckdb_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    conn.execute("PRAGMA temp_directory='duckdb_tmp'")

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

                COUNT(*) AS count,
                SUM(wins) AS wins,

                AVG(pointShotCount) FILTER (WHERE wins = 1) AS avgPointShotCount,
                AVG(initialVelocity) FILTER (WHERE wins = 1) AS initialVelocity,
                AVG(airTravelDistance) FILTER (WHERE wins = 1) AS airTravelDistance,
                AVG(netClearance) FILTER (WHERE wins = 1) AS netClearance

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputPattern")
    parser.add_argument("outputFile")
    args = parser.parse_args()
    main(args.inputPattern, args.outputFile)

# LAST USED
# python ConsolidateResults.py shots_*.parquet ConsolidatedResults.010