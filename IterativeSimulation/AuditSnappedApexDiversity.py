import argparse
import duckdb
from collections import defaultdict
from typing import Dict, List, Set, Tuple

CanonicalApexValues = [
    1.00, 1.25, 1.50, 1.80, 2.10, 2.40, 2.70,
    3.00, 3.30, 3.60, 4.50, 6.00, 8.00, 10.00,
]


def _is_canonical_apex(apex_value: float, tolerance: float = 1e-4) -> bool:
    normalized_value = round(float(apex_value), 2)
    for canonical in CanonicalApexValues:
        if abs(normalized_value - round(float(canonical), 2)) <= tolerance:
            return True
    return False


def _build_where_clause(args: argparse.Namespace) -> str:
    filters: List[str] = []

    if args.interceptCol is not None:
        filters.append(f"interceptCol = {int(args.interceptCol)}")
    if args.interceptRow is not None:
        filters.append(f"interceptRow = {int(args.interceptRow)}")
    if args.interceptZ is not None:
        filters.append(f"ROUND(interceptZ, 1) = {float(args.interceptZ):.1f}")
    if args.opponentCol is not None:
        filters.append(f"opponentCol = {int(args.opponentCol)}")
    if args.opponentRow is not None:
        filters.append(f"opponentRow = {int(args.opponentRow)}")

    if not filters:
        return ""

    return "WHERE " + " AND ".join(filters)


def audit_apex_diversity(parquet_file: str, args: argparse.Namespace) -> None:
    conn = duckdb.connect()
    where_clause = _build_where_clause(args)

    rows = conn.execute(
        f"""
        SELECT
            interceptCol,
            interceptRow,
            ROUND(interceptZ, 1) AS interceptZ,
            opponentCol,
            opponentRow,
            bounceCol,
            bounceRow,
            apexHeight
        FROM read_parquet('{parquet_file}')
        {where_clause}
        """
    ).fetchall()

    conn.close()

    grouped_apex_values: Dict[Tuple[int, int, float, int, int, int, int], Set[float]] = defaultdict(set)

    for (
        intercept_col,
        intercept_row,
        intercept_z,
        opponent_col,
        opponent_row,
        bounce_col,
        bounce_row,
        apex_height,
    ) in rows:
        key = (
            int(intercept_col),
            int(intercept_row),
            float(intercept_z),
            int(opponent_col),
            int(opponent_row),
            int(bounce_col),
            int(bounce_row),
        )
        grouped_apex_values[key].add(float(apex_height))

    total_groups = len(grouped_apex_values)
    non_canonical_groups = 0
    fewer_than_three_groups = 0

    print("\n=== Apex Diversity Audit ===\n")
    print(f"Parquet file: {parquet_file}")
    if where_clause:
        print(f"Filter: {where_clause}")
    print(f"Context+bounce groups: {total_groups}")
    print()

    header = (
        "interceptCol,interceptRow,interceptZ,opponentCol,opponentRow,"
        "bounceCol,bounceRow,apexCount,allCanonical,apexValues"
    )
    print(header)

    for key in sorted(grouped_apex_values.keys()):
        apex_values = sorted(grouped_apex_values[key])
        all_canonical = all(_is_canonical_apex(value) for value in apex_values)

        if not all_canonical:
            non_canonical_groups += 1
        if len(apex_values) < 3:
            fewer_than_three_groups += 1

        apex_values_display = "[" + ", ".join(f"{value:.2f}" for value in apex_values) + "]"
        print(
            f"{key[0]},{key[1]},{key[2]:.1f},{key[3]},{key[4]},"
            f"{key[5]},{key[6]},{len(apex_values)},{all_canonical},{apex_values_display}"
        )

    print("\n=== Summary ===\n")
    print(f"Groups with non-canonical apex values: {non_canonical_groups}")
    print(f"Groups with fewer than 3 apex values: {fewer_than_three_groups}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit distinct apex heights per context+bounce cell and verify they "
            "match canonical apex buckets."
        )
    )
    parser.add_argument("parquetFile", type=str, help="Path to reference parquet file")
    parser.add_argument("--interceptCol", type=int, default=None)
    parser.add_argument("--interceptRow", type=int, default=None)
    parser.add_argument("--interceptZ", type=float, default=None)
    parser.add_argument("--opponentCol", type=int, default=None)
    parser.add_argument("--opponentRow", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    audit_apex_diversity(cli_args.parquetFile, cli_args)

# Example:
# python IterativeSimulation/AuditSnappedApexDiversity.py IterativeSimulation/Gen23Reference.parquet --interceptCol 8 --interceptRow 5 --interceptZ 2.7 --opponentCol 5 --opponentRow 23
