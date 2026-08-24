import duckdb
import argparse

# --------------------------------------------------
# Context vs Option keys
# --------------------------------------------------
# CONTEXT_KEYS:
#   Define the situational context in which options compete.
#   Ranking (top-N pruning) happens within these groups.
#
# OPTION_KEYS:
#   Define distinct physical / tactical shot options.
#   Downhill velocity variants must remain distinct options.
# --------------------------------------------------

CONTEXT_KEYS = [
    "interceptCol", "interceptRow", "interceptZ",
    "opponentCol", "opponentRow"
]

OPTION_KEYS = [
    "defensiveCol", "defensiveRow",
    "bounceCol", "bounceRow",
    "apexHeight",
    "spinTopRpm", "spinSideRpm",
    # NEW: downhill velocity identity
    "velocityBucket"
]


# --------------------------------------------------
# Aggregate and prune
# --------------------------------------------------

def aggregate_and_prune(
    input_pattern,
    output_file,
    min_samples=5,
    min_wins=1,
    top_n=10
):
    conn = duckdb.connect()
    conn.execute("SET memory_limit='8GB'")

    print(f"Loading consolidated files: {input_pattern}")

    # --------------------------------------------------
    # 1. Merge consolidated inputs
    #
    # IMPORTANT:
    #  - velocityBucket is now part of the option identity
    #  - initialVelocity must NOT be averaged across buckets
    # --------------------------------------------------
    merged = conn.execute(
        f"""
        SELECT
            -- Context
            interceptCol, interceptRow, interceptZ,
            opponentCol, opponentRow,

            -- Option identity
            defensiveCol, defensiveRow,
            bounceCol, bounceRow,
            apexHeight,
            spinTopRpm, spinSideRpm,
            velocityBucket,

            -- Aggregates
            SUM(count) AS count,
            SUM(wins) AS wins,

            -- Carry-through physical quantities
            -- (velocity MUST remain distinct per bucket)
            MAX(initialVelocity) AS initialVelocity,

            -- Weighted sums for recomputing averages
            SUM(airTravelDistance * count) AS dist_sum,
            SUM(netClearance * count) AS net_sum

        FROM read_parquet('{input_pattern}')
        GROUP BY
            interceptCol, interceptRow, interceptZ,
            opponentCol, opponentRow,
            defensiveCol, defensiveRow,
            bounceCol, bounceRow,
            apexHeight,
            spinTopRpm, spinSideRpm,
            velocityBucket
        """
    ).df()

    print(f"Merged rows: {len(merged)}")

    if len(merged) == 0:
        print("No data after merge; exiting.")
        return

    # --------------------------------------------------
    # 2. Compute derived metrics
    # --------------------------------------------------

    merged["win_pct"] = merged["wins"] / merged["count"]
    merged["airTravelDistance"] = merged["dist_sum"] / merged["count"]
    merged["netClearance"] = merged["net_sum"] / merged["count"]

    merged.drop(
        columns=["dist_sum", "net_sum"],
        inplace=True
    )

    # --------------------------------------------------
    # 3. Basic pruning: remove meaningless shots
    # --------------------------------------------------

    pruned = merged[
        (merged["wins"] >= min_wins) &
        (merged["count"] >= min_samples)
    ]

    print(f"After basic pruning: {len(pruned)}")

    if len(pruned) == 0:
        print("No rows left after pruning; exiting.")
        return

    # --------------------------------------------------
    # 4. Rank options within each context
    #
    # Each downhill velocity variant competes
    # fairly as its own option.
    # --------------------------------------------------

    pruned["rank"] = (
        pruned
        .groupby(CONTEXT_KEYS)["win_pct"]
        .rank(method="first", ascending=False)
    )

    final = pruned[
        pruned["rank"] <= top_n
    ].drop(columns=["rank"])

    print(f"Final rows after top-{top_n} pruning: {len(final)}")

    # --------------------------------------------------
    # 5. Write output
    # --------------------------------------------------

    conn.register("final_df", final)
    conn.execute(
        f"""
        COPY final_df
        TO '{output_file}'
        (FORMAT PARQUET)
        """
    )

    print(f"✅ Aggregated and pruned file written to: {output_file}")


# --------------------------------------------------
# CLI entry point
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate and prune consolidated results, "
            "preserving downhill velocity variants"
        )
    )

    parser.add_argument(
        "inputPattern",
        help="Glob for consolidated_*.parquet"
    )

    parser.add_argument(
        "--out",
        default="FinalPlaybook.parquet",
        help="Output parquet file"
    )

    parser.add_argument(
        "--min_samples",
        type=int,
        default=5
    )

    parser.add_argument(
        "--min_wins",
        type=int,
        default=1
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=10
    )

    args = parser.parse_args()

    aggregate_and_prune(
        args.inputPattern,
        args.out,
        args.min_samples,
        args.min_wins,
        args.top_n
    )