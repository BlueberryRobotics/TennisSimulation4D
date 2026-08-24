import argparse
import concurrent.futures
import os
import pickle
import sys
import time
from typing import Dict, Any, Tuple, Optional

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from CourtPlayerSettings import Court
from Trajectory4D.FenceGridIndexer import CellCenter
from Trajectory4D.PlayerMovement import PlayerMovement
from Trajectory4D.Trajectory4DCanonical import Trajectory4DCanonical
from Trajectory4D.TransformLayer import TransformLayer


DEFAULT_INTERCEPT_Z_VALUES = [
    0.30, 0.60, 1.00, 1.25, 1.50, 1.80,
    2.10, 2.40, 2.70, 3.00, 3.30,
]

DEFAULT_SPIN_TOP_VALUES = [-3000, -1500, 0, 1500, 2500, 3500]
DEFAULT_SPIN_SIDE_VALUES = [-2000, -1000, 0, 1000, 2000]

DEFAULT_APEX_VALUES = [
    1.00, 1.25, 1.50, 1.80, 2.10, 2.40, 2.70,
    3.00, 3.30, 3.60, 4.50, 6.00, 8.00, 10.00,
]


def _is_north_side_row(court: Court, row: int) -> bool:
    if hasattr(court, "GetRowCenterY") and hasattr(court, "netY"):
        row_center_y = float(court.GetRowCenterY(int(row)))
        return row_center_y < float(court.netY)
    return int(row) <= int(getattr(court, "gridRows", 26) // 2)


def _find_in_play_end_index(z_samples: np.ndarray, bounce_index: int, epsilon: float = 1e-6) -> int:
    sample_count = len(z_samples)
    if sample_count == 0:
        return -1

    bounce_index_clamped = max(0, min(int(bounce_index), sample_count - 1))
    post_bounce_samples = z_samples[bounce_index_clamped:]
    if len(post_bounce_samples) <= 1:
        return sample_count - 1

    apex_relative_index = int(np.argmax(post_bounce_samples))
    apex_index = bounce_index_clamped + apex_relative_index
    if apex_index >= sample_count - 1:
        return sample_count - 1

    second_ground_contact_indices = np.where(z_samples[apex_index + 1:] <= float(epsilon))[0]
    if second_ground_contact_indices.size == 0:
        return sample_count - 1

    return int(apex_index + 1 + second_ground_contact_indices[0])


def _is_trajectory_reachable(
    transformed: Dict[str, Any],
    defender_side: str,
    defender_pos: Tuple[float, float],
    court: Court,
    movement_model: PlayerMovement,
) -> bool:
    x = transformed["fencesX"]
    y = transformed["fencesY"]
    z = transformed["fencesZ"]
    t = transformed["time"]
    bounce_index = int(transformed.get("bounceIndex", 0))

    defender_x = float(defender_pos[0])
    defender_y = float(defender_pos[1])
    net_y = float(court.netY)

    player_velocity = float(getattr(movement_model, "playerSpeed", court.playerSpeed))
    player_reaction_time = float(getattr(movement_model, "reactionTime", court.playerReactionTime))
    reachable_height_min = float(getattr(movement_model, "reachZMin", court.playerReachZMin))
    reachable_height_max = float(getattr(movement_model, "reachZMax", court.playerReachZMax))

    half_mask = (y <= net_y) if (defender_side == "PLAYER_BLUE") else (y >= net_y)

    in_play_end_index = _find_in_play_end_index(z, bounce_index)
    in_play_mask = np.arange(len(z), dtype=int) <= in_play_end_index
    half_mask = half_mask & in_play_mask

    idx_half = np.where(half_mask)[0]
    if idx_half.size == 0:
        return False

    distance_to_defender = np.hypot(x[idx_half] - defender_x, y[idx_half] - defender_y)
    nearest_index_relative = int(np.argmin(distance_to_defender))
    nearest_intercept_point_index = int(idx_half[nearest_index_relative])
    nearest_intercept_t = float(t[nearest_intercept_point_index])

    defender_intercept_radius = max(0.0, nearest_intercept_t - player_reaction_time) * player_velocity

    in_z = (z >= reachable_height_min) & (z <= reachable_height_max)
    distance_to_intercept = np.hypot(x - defender_x, y - defender_y)
    bucket_mask = half_mask & in_z & (distance_to_intercept <= defender_intercept_radius)

    return int(np.count_nonzero(bucket_mask)) > 0


def _build_generator(court: Court, trajectory_library_path: str) -> Trajectory4DCanonical:
    with open(trajectory_library_path, "rb") as fp:
        trajectory_library = pickle.load(fp)

    transform_layer = TransformLayer(debug=False)
    generator = Trajectory4DCanonical(
        trajectoryLibrary=trajectory_library,
        transformLayer=transform_layer,
        interceptZValues=DEFAULT_INTERCEPT_Z_VALUES,
        apexHeight=float(DEFAULT_APEX_VALUES[0]),
        apexValues=DEFAULT_APEX_VALUES,
        spinTopValues=np.array(DEFAULT_SPIN_TOP_VALUES),
        spinSideValues=np.array(DEFAULT_SPIN_SIDE_VALUES),
        bounceDistanceTolerance=0.75,
        debug=False,
        court=court,
    )
    return generator


def _generate_transformed_trajectory(
    row: Dict[str, Any],
    court: Court,
    generator: Trajectory4DCanonical,
) -> Optional[Dict[str, Any]]:
    intercept_col = int(row["interceptCol"])
    intercept_row = int(row["interceptRow"])
    bounce_col = int(row["bounceCol"])
    bounce_row = int(row["bounceRow"])

    intercept_x, intercept_y = CellCenter(intercept_col, intercept_row, court)
    bounce_x, bounce_y = CellCenter(bounce_col, bounce_row, court)

    intercept_point = (
        float(intercept_x),
        float(intercept_y),
        float(row["interceptZ"]),
    )
    bounce_point = (float(bounce_x), float(bounce_y))

    apex_height = float(row["apexHeight"])
    spin_top_rpm = int(row["spinTopRpm"])
    spin_side_rpm = int(row["spinSideRpm"])

    result = generator.generate_by_apex_ladder(
        interceptPoint=intercept_point,
        bouncePoint=bounce_point,
        apexHeight=apex_height,
        apexValues=DEFAULT_APEX_VALUES,
        spinTopRpm=spin_top_rpm,
        spinSideRpm=spin_side_rpm,
        maxNetClearAbove=None,
        landing_tol=0.10,
        net_eps=5e-3,
        maxItersPerApex=600,
    )

    if isinstance(result, dict) and "traj" in result:
        return result["traj"]

    if isinstance(result, dict):
        return result

    return None


def _row_to_dict(columns, row_tuple) -> Dict[str, Any]:
    return {column_name: row_tuple[index] for index, column_name in enumerate(columns)}


def _format_seconds(total_seconds: float) -> str:
    seconds_value = max(0, int(total_seconds))
    hours = seconds_value // 3600
    minutes = (seconds_value % 3600) // 60
    seconds = seconds_value % 60

    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


_WORKER_COLUMN_NAMES = None
_WORKER_COURT = None
_WORKER_MOVEMENT_MODEL = None
_WORKER_GENERATOR = None


def _init_worker(
    column_names,
    trajectory_library_path: str,
    court_geometry_mode: str,
) -> None:
    global _WORKER_COLUMN_NAMES
    global _WORKER_COURT
    global _WORKER_MOVEMENT_MODEL
    global _WORKER_GENERATOR

    _WORKER_COLUMN_NAMES = list(column_names)
    _WORKER_COURT = Court(geometryMode=court_geometry_mode)
    _WORKER_MOVEMENT_MODEL = PlayerMovement(_WORKER_COURT)
    _WORKER_GENERATOR = _build_generator(_WORKER_COURT, trajectory_library_path)


def _process_rows_batch(row_tuples):
    output_batch = []

    rows_processed = 0
    winner_true_before = 0
    winner_true_after = 0
    winner_newly_set = 0
    generation_failures = 0
    reachability_checks = 0

    for row_tuple in row_tuples:
        rows_processed += 1
        row = _row_to_dict(_WORKER_COLUMN_NAMES, row_tuple)

        winner_before = bool(row.get("winner", False))
        if winner_before:
            winner_true_before += 1

        winner_after = winner_before

        # Preserve existing TRUE values without expensive trajectory recompute.
        if not winner_before:
            try:
                transformed = _generate_transformed_trajectory(row, _WORKER_COURT, _WORKER_GENERATOR)
                if transformed is None:
                    raise RuntimeError("No transformed trajectory generated")

                intercept_row = int(row["interceptRow"])
                defender_side = "PLAYER_RED" if _is_north_side_row(_WORKER_COURT, intercept_row) else "PLAYER_BLUE"

                defender_col = int(row["opponentCol"])
                defender_row = int(row["opponentRow"])
                defender_pos = CellCenter(defender_col, defender_row, _WORKER_COURT)

                reachable = _is_trajectory_reachable(
                    transformed=transformed,
                    defender_side=defender_side,
                    defender_pos=(float(defender_pos[0]), float(defender_pos[1])),
                    court=_WORKER_COURT,
                    movement_model=_WORKER_MOVEMENT_MODEL,
                )
                reachability_checks += 1

                if not reachable:
                    winner_after = True
            except Exception:
                generation_failures += 1

        if winner_after and not winner_before:
            winner_newly_set += 1

        row["winner"] = bool(winner_after)
        if winner_after:
            winner_true_after += 1

        output_batch.append(row)

    stats = {
        "rows_processed": rows_processed,
        "winner_true_before": winner_true_before,
        "winner_true_after": winner_true_after,
        "winner_newly_set": winner_newly_set,
        "generation_failures": generation_failures,
        "reachability_checks": reachability_checks,
    }
    return output_batch, stats


def backfill_winner(
    input_parquet_path: str,
    output_parquet_path: str,
    trajectory_library_path: str,
    court_geometry_mode: str,
    batch_size: int,
    progress_every_rows: int,
    num_workers: int,
    max_pending_batches: int,
) -> None:
    connection = duckdb.connect()
    total_expected_rows = int(
        connection.execute("SELECT COUNT(*) FROM read_parquet(?)", [input_parquet_path]).fetchone()[0]
    )
    cursor = connection.execute("SELECT * FROM read_parquet(?)", [input_parquet_path])
    column_names = [description[0] for description in cursor.description]

    temp_output_path = output_parquet_path + ".tmp"
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    writer = None

    total_rows = 0
    winner_true_before = 0
    winner_true_after = 0
    winner_newly_set = 0
    generation_failures = 0
    reachability_checks = 0
    progress_every_rows = max(1, int(progress_every_rows))
    num_workers = max(1, int(num_workers))
    max_pending_batches = max(1, int(max_pending_batches))
    last_progress_rows = 0
    start_time = time.time()

    print(
        "[Config] "
        f"workers={num_workers} "
        f"batchSize={int(batch_size):,} "
        f"maxPendingBatches={max_pending_batches} "
        f"progressEveryRows={progress_every_rows:,} "
        f"expectedRows={total_expected_rows:,}",
        flush=True,
    )

    def _write_output_batch(output_rows_batch):
        nonlocal writer
        if not output_rows_batch:
            return
        table = pa.Table.from_pylist(output_rows_batch)
        if writer is None:
            writer = pq.ParquetWriter(temp_output_path, table.schema, compression="snappy")
        writer.write_table(table)

    def _accumulate_stats(batch_stats):
        nonlocal total_rows
        nonlocal winner_true_before
        nonlocal winner_true_after
        nonlocal winner_newly_set
        nonlocal generation_failures
        nonlocal reachability_checks
        nonlocal last_progress_rows

        total_rows += int(batch_stats["rows_processed"])
        winner_true_before += int(batch_stats["winner_true_before"])
        winner_true_after += int(batch_stats["winner_true_after"])
        winner_newly_set += int(batch_stats["winner_newly_set"])
        generation_failures += int(batch_stats["generation_failures"])
        reachability_checks += int(batch_stats["reachability_checks"])

        if (total_rows - last_progress_rows) >= progress_every_rows:
            elapsed_seconds = max(1e-9, (time.time() - start_time))
            rows_per_second = float(total_rows) / float(elapsed_seconds)
            remaining_rows = max(0, total_expected_rows - total_rows)
            eta_seconds = (float(remaining_rows) / rows_per_second) if rows_per_second > 0 else 0.0
            pct = (100.0 * float(total_rows) / float(total_expected_rows)) if total_expected_rows > 0 else 100.0
            print(
                "[Progress] "
                f"rows={total_rows:,}/{total_expected_rows:,} "
                f"({pct:.2f}%) "
                f"rate={rows_per_second:,.0f} rows/s "
                f"elapsed={_format_seconds(elapsed_seconds)} "
                f"eta={_format_seconds(eta_seconds)}",
                flush=True,
            )
            last_progress_rows = total_rows

    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(column_names, trajectory_library_path, court_geometry_mode),
        ) as executor:
            pending_futures = []

            while True:
                rows = cursor.fetchmany(int(batch_size))
                if not rows:
                    break

                pending_futures.append(executor.submit(_process_rows_batch, rows))

                while len(pending_futures) >= max_pending_batches:
                    done_futures, not_done_futures = concurrent.futures.wait(
                        pending_futures,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    pending_futures = list(not_done_futures)

                    for done_future in done_futures:
                        output_batch, batch_stats = done_future.result()
                        _write_output_batch(output_batch)
                        _accumulate_stats(batch_stats)

            for done_future in concurrent.futures.as_completed(pending_futures):
                output_batch, batch_stats = done_future.result()
                _write_output_batch(output_batch)
                _accumulate_stats(batch_stats)

        if writer is not None:
            writer.close()
            writer = None

        os.replace(temp_output_path, output_parquet_path)

    finally:
        if writer is not None:
            writer.close()
        connection.close()
        if os.path.exists(temp_output_path) and temp_output_path != output_parquet_path:
            try:
                os.remove(temp_output_path)
            except Exception:
                pass

    print("\nBackfill complete")
    print(f"Input:  {input_parquet_path}")
    print(f"Output: {output_parquet_path}")
    print(f"Total rows: {total_rows:,}")
    print(f"Winner true before: {winner_true_before:,}")
    print(f"Winner true after:  {winner_true_after:,}")
    print(f"Winner newly set:   {winner_newly_set:,}")
    print(f"Reachability checks: {reachability_checks:,}")
    print(f"Trajectory generation failures: {generation_failures:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill winner field by regenerating each trajectory and checking whether the defender can reach it. "
            "Rows with unreachable returns are marked winner=true."
        )
    )
    parser.add_argument(
        "inputParquet",
        help="Input consolidated parquet file (for example IterativeSimulation/ConsolidatedGen23.parquet)",
    )
    parser.add_argument(
        "--outputParquet",
        default=None,
        help="Output parquet file path. Default writes <input>_winner_backfilled.parquet",
    )
    parser.add_argument(
        "--trajectoryLibrary",
        default="Trajectory4DLibrary006.pkl",
        help="Path to trajectory library pickle used by canonical generator",
    )
    parser.add_argument(
        "--courtGeometryMode",
        default="short_rows_13_14",
        choices=["uniform", "short_rows_13_14"],
        help="Court geometry mode to use for reachability checks",
    )
    parser.add_argument(
        "--batchSize",
        type=int,
        default=20000,
        help="Rows fetched per batch from parquet (higher is usually faster)",
    )
    parser.add_argument(
        "--progressEveryRows",
        type=int,
        default=50_000_000,
        help="Print progress/ETA after this many additional rows are processed",
    )
    parser.add_argument(
        "--numWorkers",
        type=int,
        default=0,
        help="Worker process count. 0 means auto-tune for local CPU",
    )
    parser.add_argument(
        "--maxPendingBatches",
        type=int,
        default=0,
        help="Max in-flight worker batches. 0 means 2x worker count",
    )

    args = parser.parse_args()

    input_parquet_path = args.inputParquet
    if args.outputParquet:
        output_parquet_path = args.outputParquet
    else:
        root, ext = os.path.splitext(input_parquet_path)
        output_parquet_path = f"{root}_winner_backfilled{ext}"

    detected_cpu_count = os.cpu_count() or 4
    if int(args.numWorkers) > 0:
        num_workers = int(args.numWorkers)
    else:
        # Keep one core free for OS and I/O; cap to avoid oversubscription.
        num_workers = max(1, min(12, detected_cpu_count - 1))

    if int(args.maxPendingBatches) > 0:
        max_pending_batches = int(args.maxPendingBatches)
    else:
        max_pending_batches = max(2, num_workers * 2)

    backfill_winner(
        input_parquet_path=input_parquet_path,
        output_parquet_path=output_parquet_path,
        trajectory_library_path=args.trajectoryLibrary,
        court_geometry_mode=args.courtGeometryMode,
        batch_size=max(1, int(args.batchSize)),
        progress_every_rows=max(1, int(args.progressEveryRows)),
        num_workers=num_workers,
        max_pending_batches=max_pending_batches,
    )


if __name__ == "__main__":
    main()

# This script seemed to make sense...
# but it will take take weeks to run against a consolidated file
# had it run overnight and it had only processed 200 MB of a 23 GB file.
# went back to simply running the simulation
# python BackfillWinnerByReachability.py ConsolidatedGen23.parquet --outputParquet IterativeSimulation/ConsolidatedGen23_win.parquet --progressEveryRows 50000000