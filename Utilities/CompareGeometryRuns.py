import argparse
import csv
from collections import Counter, defaultdict
import os
import sys

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from CourtPlayerSettings import Court
from Trajectory4D.FenceGridIndexer import CellCenter


CONTEXT_KEY_COLUMNS = (
    "interceptCol",
    "interceptRow",
    "interceptZ",
    "opponentCol",
    "opponentRow",
)


def _to_int(value):
    return int(float(value))


def _to_float(value):
    return float(value)


def _read_rows(csv_path):
    with open(csv_path, newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        return list(reader)


def _context_stats(rows):
    stats = defaultdict(lambda: {"wins": 0.0, "count": 0.0})

    for row in rows:
        context_key = (
            _to_int(row["interceptCol"]),
            _to_int(row["interceptRow"]),
            round(_to_float(row["interceptZ"]), 2),
            _to_int(row["opponentCol"]),
            _to_int(row["opponentRow"]),
        )
        stats[context_key]["wins"] += _to_float(row["wins"])
        stats[context_key]["count"] += 1.0

    return stats


def _service_band_counts(rows, court):
    counts = Counter()

    for row in rows:
        bounce_col = _to_int(row["bounceCol"])
        bounce_row = _to_int(row["bounceRow"])
        _, bounce_y = CellCenter(bounce_col, bounce_row, court)

        if court.netY <= bounce_y <= court.opponentServiceLineY:
            counts["northServiceBand"] += 1
        elif court.serviceLineY <= bounce_y <= court.netY:
            counts["southServiceBand"] += 1
        else:
            counts["outsideServiceBands"] += 1

    return counts


def _print_context_overlap(label_a, stats_a, label_b, stats_b):
    keys_a = set(stats_a.keys())
    keys_b = set(stats_b.keys())

    overlap = keys_a & keys_b
    union = keys_a | keys_b

    jaccard = (float(len(overlap)) / float(len(union))) if union else 1.0

    print("\n[COMPARE] Context overlap")
    print(f"{label_a} unique contexts: {len(keys_a):,}")
    print(f"{label_b} unique contexts: {len(keys_b):,}")
    print(f"Overlap contexts: {len(overlap):,}")
    print(f"Jaccard overlap: {jaccard:.4f}")


def _print_winpct_drift(stats_a, stats_b):
    overlap = set(stats_a.keys()) & set(stats_b.keys())
    if not overlap:
        print("\n[COMPARE] winPct drift: no overlapping contexts")
        return

    abs_diffs = []
    weighted_abs_diffs = []
    total_weight = 0.0

    for key in overlap:
        wins_a = stats_a[key]["wins"]
        count_a = stats_a[key]["count"]
        wins_b = stats_b[key]["wins"]
        count_b = stats_b[key]["count"]

        win_pct_a = wins_a / count_a if count_a > 0 else 0.0
        win_pct_b = wins_b / count_b if count_b > 0 else 0.0
        diff = abs(win_pct_a - win_pct_b)
        abs_diffs.append(diff)

        weight = min(count_a, count_b)
        weighted_abs_diffs.append(diff * weight)
        total_weight += weight

    mean_abs = sum(abs_diffs) / len(abs_diffs)
    weighted_mean_abs = (sum(weighted_abs_diffs) / total_weight) if total_weight > 0 else 0.0

    print("\n[COMPARE] winPct drift on overlapping contexts")
    print(f"Mean absolute drift: {mean_abs:.4f}")
    print(f"Weighted mean absolute drift: {weighted_mean_abs:.4f}")


def _print_service_band_delta(label_a, counts_a, label_b, counts_b):
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())

    print("\n[COMPARE] Service-band occupancy")
    for key in ("northServiceBand", "southServiceBand", "outsideServiceBands"):
        count_a = counts_a.get(key, 0)
        count_b = counts_b.get(key, 0)
        pct_a = (100.0 * count_a / total_a) if total_a > 0 else 0.0
        pct_b = (100.0 * count_b / total_b) if total_b > 0 else 0.0
        print(
            f"{key}: {label_a}={count_a:,} ({pct_a:.2f}%) | "
            f"{label_b}={count_b:,} ({pct_b:.2f}%)"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two simulation output CSV files (uniform vs short_rows_13_14)."
        )
    )
    parser.add_argument("uniformCsv", help="CSV produced in uniform geometry mode")
    parser.add_argument("shortCsv", help="CSV produced in short_rows_13_14 mode")
    args = parser.parse_args()

    uniform_rows = _read_rows(args.uniformCsv)
    short_rows = _read_rows(args.shortCsv)

    print("[COMPARE] File stats")
    print(f"uniform rows: {len(uniform_rows):,}")
    print(f"short rows:   {len(short_rows):,}")

    uniform_stats = _context_stats(uniform_rows)
    short_stats = _context_stats(short_rows)

    _print_context_overlap("uniform", uniform_stats, "short", short_stats)
    _print_winpct_drift(uniform_stats, short_stats)

    uniform_court = Court(geometryMode="uniform")
    short_court = Court(geometryMode="short_rows_13_14")

    uniform_bands = _service_band_counts(uniform_rows, uniform_court)
    short_bands = _service_band_counts(short_rows, short_court)
    _print_service_band_delta("uniform", uniform_bands, "short", short_bands)


if __name__ == "__main__":
    main()
