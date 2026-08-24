import argparse
import math
import os
import sys

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from CourtPlayerSettings import Court
from Trajectory4D.FenceGridIndexer import XyToCell, ServiceBoxCells, OpponentHalfRows


def _format_float(value: float) -> str:
    return f"{float(value):.4f}"


def _print_line_positions(court: Court, label: str) -> None:
    print(f"\n[{label}] Line positions (meters)")
    print(f"serverBaselineY      = {_format_float(court.serverBaselineY)}")
    print(f"serviceLineY         = {_format_float(court.serviceLineY)}")
    print(f"netY                 = {_format_float(court.netY)}")
    print(f"opponentServiceLineY = {_format_float(court.opponentServiceLineY)}")
    print(f"receiverBaselineY    = {_format_float(court.receiverBaselineY)}")


def _print_key_boundaries(court: Court, label: str) -> None:
    print(f"\n[{label}] Key row boundaries (meters)")
    for boundary_index in (4, 8, 13, 18, 22):
        if hasattr(court, "RowBoundaryY"):
            boundary_y = court.RowBoundaryY(boundary_index)
        else:
            boundary_y = boundary_index * court.granularity
        print(f"boundary {boundary_index:02d}/{boundary_index + 1:02d}: {_format_float(boundary_y)}")


def _print_row_center_deltas(uniform_court: Court, short_court: Court) -> None:
    print("\n[AB] Row center delta (short - uniform)")
    changed_rows = 0
    for row in range(1, min(uniform_court.gridRows, short_court.gridRows) + 1):
        u = uniform_court.GetRowCenterY(row)
        s = short_court.GetRowCenterY(row)
        delta = s - u
        if abs(delta) > 1e-9:
            changed_rows += 1
            print(
                f"row {row:02d}: uniform={_format_float(u)} short={_format_float(s)} delta={_format_float(delta)}"
            )
    if changed_rows == 0:
        print("No row center deltas found.")


def _print_region_counts(court: Court, label: str) -> None:
    print(f"\n[{label}] Region counts")
    for hitter in ("PLAYER_NORTH", "PLAYER_SOUTH"):
        opponent_rows = OpponentHalfRows(court, forHitter=hitter)
        print(f"opponentHalfRows ({hitter}) = {len(opponent_rows)}")

    for serve_side in ("DEUCE", "AD"):
        north_cells = ServiceBoxCells(court, serve_side, forHitter="PLAYER_NORTH")
        south_cells = ServiceBoxCells(court, serve_side, forHitter="PLAYER_SOUTH")
        print(
            f"serviceBoxCells serveSide={serve_side}: north={len(north_cells)} south={len(south_cells)}"
        )


def _print_mapping_diff(uniform_court: Court, short_court: Court, y_step: float) -> None:
    print("\n[AB] Y->row mapping differences (X fixed at centerLineX)")
    x_value = float(uniform_court.centerLineX)

    y_min = 0.0
    y_max = min(uniform_court.lengthFence, short_court.lengthFence)

    changed = 0
    sample_count = 0
    y_value = y_min
    while y_value <= y_max + 1e-9:
        sample_count += 1
        _, row_uniform = XyToCell(x_value, y_value, uniform_court)
        _, row_short = XyToCell(x_value, y_value, short_court)
        if row_uniform != row_short:
            changed += 1
            print(
                f"y={_format_float(y_value)} uniformRow={row_uniform:02d} shortRow={row_short:02d}"
            )
        y_value += y_step

    print(f"[AB] mapping changed at {changed} of {sample_count} samples")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "A/B validation helper for court geometry modes. "
            "Compares key line positions, row centers, and row mapping behavior."
        )
    )
    parser.add_argument(
        "--yStep",
        type=float,
        default=0.1524,
        help="Sampling step for Y->row mapping comparison (meters). Default 0.1524 (0.5 ft).",
    )

    args = parser.parse_args()

    uniform_court = Court(geometryMode="uniform")
    short_court = Court(geometryMode="short_rows_13_14")

    print("[AB] Court geometry comparison start")
    print(
        f"[AB] lengthFence uniform={_format_float(uniform_court.lengthFence)} "
        f"short={_format_float(short_court.lengthFence)}"
    )

    _print_line_positions(uniform_court, "uniform")
    _print_line_positions(short_court, "short_rows_13_14")

    _print_key_boundaries(short_court, "short_rows_13_14")
    _print_row_center_deltas(uniform_court, short_court)

    _print_region_counts(uniform_court, "uniform")
    _print_region_counts(short_court, "short_rows_13_14")

    y_step = max(0.01, float(args.yStep))
    _print_mapping_diff(uniform_court, short_court, y_step=y_step)

    print("\n[AB] Court geometry comparison end")


if __name__ == "__main__":
    main()
