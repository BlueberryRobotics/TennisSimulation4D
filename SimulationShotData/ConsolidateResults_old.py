import duckdb
import argparse
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# Constants / grid helpers (unchanged)
# -------------------------------------------------------------------

GRID_ROWS = 26
GRID_COLS = 14
MID_ROW = GRID_ROWS // 2  # 13
MID_COL = GRID_COLS // 2  # 7


def south(row):
    return row > MID_ROW


# -------------------------------------------------------------------
# Downhill velocity bucketing (NEW)
# -------------------------------------------------------------------

# Canonical downhill speeds in m/s (from trajectory generation)
DOWNHILL_SPEEDS = [
    31.2928,  # 70 mph
    37.9984,  # 85 mph
    44.7040   # 100 mph
]


def downhill_velocity_bucket(row):
    """
    Assign a velocity bucket (m/s) for degenerate downhill shots
    where interceptZ == apexHeight.

    Returns:
        - one of the canonical downhill speeds (m/s)
        - or None for all non-degenerate shots
    """

    interceptZ = row["interceptPoint"][2]
    apexH = row["apexHeight"]

    # Only apply to downhill-degenerate shots
    if abs(interceptZ - apexH) > 1e-6:
        return None

    v = row["initialVelocity"]
    if v is None:
        return None

    # initialVelocity may be scalar or vector
    if isinstance(v, (list, tuple, np.ndarray)):
        speed = float(np.linalg.norm(v))
    else:
        speed = float(v)

    # Bucket with tolerance for floating-point noise
    for s in DOWNHILL_SPEEDS:
        if abs(speed - s) < 0.5:
            return s

    # Should not happen if generation is correct
    return None


# -------------------------------------------------------------------
# Consolidation logic
# -------------------------------------------------------------------

def consolidate(input_pattern, output_file):

    conn = duckdb.connect()

    # Load all parquet files
    df = conn.execute(
        f"""
        SELECT *
        FROM read_parquet('{input_pattern}')
        """
    ).df()

    if len(df) == 0:
        print("No input rows found.")
        return

    # ---------------------------------------------------------------
    # Add downhill velocity bucket
    # ---------------------------------------------------------------
    df["velocityBucket"] = df.apply(downhill_velocity_bucket, axis=1)

    # ---------------------------------------------------------------
    # GROUPING COLUMNS
    # (Original columns preserved + velocityBucket added)
    # ---------------------------------------------------------------
    group_cols = [
        # Original physical specification
        "interceptPoint",
        "bouncePoint",
        "apexHeight",
        "spinTopRpm",
        "spinSideRpm",

        # Original positional / tactical dimensions
        "opponentRow",
        "opponentCol",
        "defensiveRow",
        "defensiveCol",

        # NEW: downhill velocity split (NULL for normal shots)
        "velocityBucket"
    ]

    # ---------------------------------------------------------------
    # Aggregate wins and totals
    # ---------------------------------------------------------------
    agg_df = (
        df
        .groupby(group_cols, dropna=False)
        .agg(
            wins=("win", "sum"),
            total=("win", "count")
        )
        .reset_index()
    )

    # Optional clarity / diagnostics column
    agg_df["isDownhillDegenerate"] = agg_df["velocityBucket"].notna()

    # ---------------------------------------------------------------
    # Save consolidated results
    # ---------------------------------------------------------------
    agg_df.to_parquet(output_file, index=False)

    print(f"Consolidated results written to {output_file}")
    print(f"Total consolidated rows: {len(agg_df)}")


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Consolidate simulation results with downhill velocity separation"
    )

    parser.add_argument(
        "inputPattern",
        help="Glob for simulation parquet files (e.g. shots_*.parquet)"
    )

    parser.add_argument(
        "--out",
        default="ConsolidatedResults005.parquet",
        help="Output parquet file"
    )

    args = parser.parse_args()

    consolidate(args.inputPattern, args.out)