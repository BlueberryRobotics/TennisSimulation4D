import duckdb
import glob
import os

def add_win_shot_count(input_pattern, output_dir):
    conn = duckdb.connect()

    conn.execute("PRAGMA memory_limit='4GB'")
    conn.execute("PRAGMA threads=4")

    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob(input_pattern)

    print(f"Found {len(files)} files")

    for i, infile in enumerate(files):
        filename = os.path.basename(infile)
        outfile = os.path.join(output_dir, filename)

        print(f"[{i+1}/{len(files)}] Processing: {filename}")

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
                initialVelocity,
                airTravelDistance,
                netClearance,
                wins,
                pointShotCount,

                -- NEW FIELD
                CASE
                    WHEN wins = 1 THEN pointShotCount
                    ELSE 0
                END AS winShotCount

            FROM read_parquet('{infile}')
        )
        TO '{outfile}' (FORMAT PARQUET)
        """)

    print("Done.")

if __name__ == "__main__":
    add_win_shot_count(
        input_pattern="shots_*.parquet",     # adjust path/pattern
        output_dir="shots_with_win_field"    # new directory
    )
