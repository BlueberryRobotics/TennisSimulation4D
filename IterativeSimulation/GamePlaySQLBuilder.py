#!/usr/bin/env python3
"""
Create SQLite DB with fixed table names and versioned DB filename.

- Trajectory4DLibraryXXX.pkl/pk1 -> creates a table called "trajectories"
- and a table called "points" which are linked by trajectory id
- GenXXReference.parquet -> creates a table called "references" 
- Default output DB name: TrajectoryXXXReferenceXX.db based on the input
- file names

- Handles list/dict/object-style pickle payloads.
- Writes:
    1) trajectories (one row per trajectory)
    2) points (time-series points per trajectory)
- Optional: import GenXXReference.parquet into references table.

Usage:
    python TrajectoryGenerator/TrajectoryTransformer.py \
            --pkl Trajectory4DLibrary005.pkl \
            --gen-reference-parquet GamePlay/Gen11Reference.parquet
"""

import argparse
import decimal
import math
import pickle
import re
import sqlite3
import time
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------- Generic helpers ----------

def get_attr_or_key(obj: Any, names: Iterable[str], default=None):
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def to_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def to_int(v, default=0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def to_list(v) -> List[Any]:
    if v is None:
        return []
    # numpy/pandas arrays
    if hasattr(v, "tolist"):
        try:
            return v.tolist()
        except Exception:
            pass
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def unpack_point2(v) -> Tuple[float, float]:
    if v is None:
        return 0.0, 0.0
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return to_float(v[0]), to_float(v[1])
    if isinstance(v, dict):
        return to_float(v.get("x", 0.0)), to_float(v.get("y", 0.0))
    if hasattr(v, "X") and hasattr(v, "Y"):
        return to_float(v.X), to_float(v.Y)
    if hasattr(v, "x") and hasattr(v, "y"):
        return to_float(v.x), to_float(v.y)
    return 0.0, 0.0


def unpack_point3(v) -> Tuple[float, float, float]:
    if v is None:
        return 0.0, 0.0, 0.0
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return to_float(v[0]), to_float(v[1]), to_float(v[2])
    if isinstance(v, dict):
        return (
            to_float(v.get("x", 0.0)),
            to_float(v.get("y", 0.0)),
            to_float(v.get("z", 0.0)),
        )
    if hasattr(v, "X") and hasattr(v, "Y") and hasattr(v, "Z"):
        return to_float(v.X), to_float(v.Y), to_float(v.Z)
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
        return to_float(v.x), to_float(v.y), to_float(v.z)
    return 0.0, 0.0, 0.0


# ---------- Normalization ----------

def normalize_entries(payload: Any) -> Iterable[Any]:
    """
    Accept common pickle shapes:
    - list[entry]
    - tuple/list wrapping entries
    - dict[id -> entry]
    - object with .trajectory005Library or .library
    """
    if isinstance(payload, dict):
        return payload.values()
    if isinstance(payload, list):
        return payload
    if isinstance(payload, tuple):
        return payload

    maybe = get_attr_or_key(payload, ["trajectory005Library", "library", "entries"])
    if maybe is not None:
        if isinstance(maybe, dict):
            return maybe.values()
        return maybe

    raise ValueError(f"Unsupported pickle payload type: {type(payload)}")


def read_entry(entry: Any, idx: int) -> Dict[str, Any]:
    # id
    eid = get_attr_or_key(entry, ["id", "Id", "trajectoryId", "key"], default=f"traj-{idx}")

    intercept_raw = get_attr_or_key(entry, ["interceptPoint", "InterceptPoint", "intercept_point"])
    bounce_raw = get_attr_or_key(entry, ["bouncePoint", "BouncePoint", "bounce_point"])

    ix, iy, iz = unpack_point3(intercept_raw)
    bx, by = unpack_point2(bounce_raw)

    apex_height = to_float(get_attr_or_key(entry, ["apex_height", "ApexHeight", "apexHeight"], 0.0))
    spin_top = to_int(get_attr_or_key(entry, ["spin_top_rpm", "SpinTopRpm", "spinTopRpm"], 0))
    spin_side = to_int(get_attr_or_key(entry, ["spin_side_rpm", "SpinSideRpm", "spinSideRpm"], 0))
    distance = to_float(get_attr_or_key(entry, ["distance", "Distance", "forwardDistance"], 0.0))
    initial_velocity = to_float(get_attr_or_key(entry, ["initialVelocity", "InitialVelocity", "initial_velocity"], 0.0))
    air_travel_distance = to_float(get_attr_or_key(entry, ["airTravelDistance", "AirTravelDistance", "air_travel_distance"], 0.0))
    bounce_index = to_int(get_attr_or_key(entry, ["bounceIndex", "BounceIndex", "bounce_index"], 0))
    landing_y = to_float(get_attr_or_key(entry, ["landingY", "LandingY", "landing_y"], 0.0))

    canon_x = to_list(get_attr_or_key(entry, ["canonX", "CanonX", "x"]))
    canon_y = to_list(get_attr_or_key(entry, ["canonY", "CanonY", "y"]))
    canon_z = to_list(get_attr_or_key(entry, ["canonZ", "CanonZ", "z"]))
    time_arr = to_list(get_attr_or_key(entry, ["time", "Time", "t"]))

    if not canon_x or not canon_y or not canon_z:
        trajectory3d = get_attr_or_key(entry, ["trajectory3D", "Trajectory3D", "traj3D"])
        trajectory3d_list = to_list(trajectory3d)
        if trajectory3d_list:
            canon_x = []
            canon_y = []
            canon_z = []
            for point in trajectory3d_list:
                x_val, y_val, z_val = unpack_point3(point)
                canon_x.append(x_val)
                canon_y.append(y_val)
                canon_z.append(z_val)

    n = min(len(canon_x), len(canon_y), len(canon_z))
    if len(time_arr) < n:
        # fallback synthetic time axis if missing
        if n > 1:
            time_arr = [i / (n - 1) for i in range(n)]
        else:
            time_arr = [0.0]
    else:
        time_arr = time_arr[:n]

    canon_x = canon_x[:n]
    canon_y = canon_y[:n]
    canon_z = canon_z[:n]

    return {
        "id": str(eid),
        "intercept_x": ix,
        "intercept_y": iy,
        "intercept_z": iz,
        "bounce_x": bx,
        "bounce_y": by,
        "apex_height": apex_height,
        "spin_top_rpm": spin_top,
        "spin_side_rpm": spin_side,
        "distance": distance,
        "initial_velocity": initial_velocity,
        "air_travel_distance": air_travel_distance,
        "bounce_index": bounce_index,
        "landing_y": landing_y,
        "point_count": n,
        "time": time_arr,
        "canon_x": canon_x,
        "canon_y": canon_y,
        "canon_z": canon_z,
    }


# ---------- SQLite ----------

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=FILE;
PRAGMA cache_size=-32768;
PRAGMA mmap_size=268435456;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS trajectories (
    id TEXT PRIMARY KEY,
    intercept_x REAL NOT NULL,
    intercept_y REAL NOT NULL,
    intercept_z REAL NOT NULL,
    bounce_x REAL NOT NULL,
    bounce_y REAL NOT NULL,
    apex_height REAL NOT NULL,
    spin_top_rpm INTEGER NOT NULL,
    spin_side_rpm INTEGER NOT NULL,
    distance REAL NOT NULL,
    initial_velocity REAL NOT NULL,
    air_travel_distance REAL NOT NULL,
    bounce_index INTEGER NOT NULL,
    landing_y REAL NOT NULL,
    point_count INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_traj_lookup
    ON trajectories(intercept_z, spin_top_rpm, spin_side_rpm, distance, apex_height);

CREATE INDEX IF NOT EXISTS ix_traj_bounce
    ON trajectories(bounce_y, bounce_x);
"""


def create_tables(conn: sqlite3.Connection, trajectory_table_name: str):
    safe_points_table = quote_identifier(trajectory_table_name)
    conn.executescript(DDL)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {safe_points_table} (
            trajectory_id TEXT NOT NULL,
            point_index INTEGER NOT NULL,
            t REAL NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL NOT NULL,
            PRIMARY KEY (trajectory_id, point_index),
            FOREIGN KEY (trajectory_id) REFERENCES trajectories(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS ix_points_traj
            ON {safe_points_table}(trajectory_id, point_index)
        """
    )


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, int(batch_size)))
        if not batch:
            break
        yield batch


def _try_len(value: Any) -> Optional[int]:
    try:
        return int(len(value))
    except Exception:
        return None


def _format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def insert_batch(conn: sqlite3.Connection, rows: List[Dict[str, Any]], trajectory_table_name: str):
    safe_points_table = quote_identifier(trajectory_table_name)
    traj_sql = """
    INSERT OR REPLACE INTO trajectories (
        id, intercept_x, intercept_y, intercept_z, bounce_x, bounce_y,
        apex_height, spin_top_rpm, spin_side_rpm, distance,
        initial_velocity, air_travel_distance, bounce_index, landing_y, point_count
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    pt_sql = f"""
    INSERT OR REPLACE INTO {safe_points_table} (
        trajectory_id, point_index, t, x, y, z
    ) VALUES (?, ?, ?, ?, ?, ?)
    """

    with conn:
        conn.executemany(
            traj_sql,
            [
                (
                    r["id"], r["intercept_x"], r["intercept_y"], r["intercept_z"],
                    r["bounce_x"], r["bounce_y"], r["apex_height"], r["spin_top_rpm"],
                    r["spin_side_rpm"], r["distance"], r["initial_velocity"],
                    r["air_travel_distance"], r["bounce_index"], r["landing_y"],
                    r["point_count"],
                )
                for r in rows
            ],
        )

        points_batch = []
        for r in rows:
            tid = r["id"]
            for i, (t, x, y, z) in enumerate(zip(r["time"], r["canon_x"], r["canon_y"], r["canon_z"])):
                points_batch.append((tid, i, to_float(t), to_float(x), to_float(y), to_float(z)))

        conn.executemany(pt_sql, points_batch)


def iter_read_rows(payload: Any, max_trajectories: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    for i, entry in enumerate(normalize_entries(payload)):
        if max_trajectories is not None and i >= int(max_trajectories):
            break
        yield read_entry(entry, i)


def _duck_to_sqlite_type(duck_type: str) -> str:
    token = (duck_type or "").upper()
    if "INT" in token:
        return "INTEGER"
    if any(name in token for name in ("DOUBLE", "FLOAT", "DECIMAL", "REAL")):
        return "REAL"
    if "BOOL" in token:
        return "INTEGER"
    if "DATE" in token or "TIME" in token:
        return "TEXT"
    return "TEXT"


def quote_identifier(identifier: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Invalid SQLite identifier: {identifier}")
    return f'"{identifier}"'


def extract_trajectory_version_from_pkl(pkl_path: Path) -> str:
    stem = pkl_path.stem
    match = re.search(r"Trajectory4DLibrary(\d+)", stem, re.IGNORECASE)
    if match:
        return match.group(1)
    trailing_digits = re.search(r"(\d+)$", stem)
    if trailing_digits:
        return trailing_digits.group(1)
    raise ValueError(f"Could not derive trajectory version from filename: {pkl_path.name}")


def extract_reference_version_from_parquet(parquet_path: Optional[Path]) -> str:
    if parquet_path is None:
        return "00"
    stem = parquet_path.stem
    match = re.search(r"Gen(\d+)GamePlay", stem, re.IGNORECASE)
    if match:
        return match.group(1)
    trailing_digits = re.search(r"(\d+)$", stem)
    if trailing_digits:
        return trailing_digits.group(1)
    raise ValueError(f"Could not derive reference version from filename: {parquet_path.name}")


def import_gen_reference(
    conn: sqlite3.Connection,
    parquet_path: Path,
    reference_table_name: str,
    batch_size: int = 50000,
):
    try:
        import duckdb
    except ImportError as e:
        raise RuntimeError("duckdb is required for --gen-reference-parquet import") from e

    db = duckdb.connect(database=":memory:")
    try:
        schema_rows = db.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [str(parquet_path)],
        ).fetchall()
        column_names = [str(row[0]) for row in schema_rows]
        column_types = [str(row[1]) for row in schema_rows]

        if not column_names:
            raise RuntimeError(f"No columns found in parquet: {parquet_path}")

        quoted_columns = ", ".join(f'"{name}" {_duck_to_sqlite_type(dtype)}' for name, dtype in zip(column_names, column_types))
        safe_reference_table = quote_identifier(reference_table_name)
        conn.execute(f"DROP TABLE IF EXISTS {safe_reference_table}")
        conn.execute(f"CREATE TABLE {safe_reference_table} ({quoted_columns})")

        select_sql = "SELECT * FROM read_parquet(?)"
        cursor = db.execute(select_sql, [str(parquet_path)])

        insert_columns = ", ".join(f'"{name}"' for name in column_names)
        placeholders = ", ".join("?" for _ in column_names)
        insert_sql = f"INSERT INTO {safe_reference_table} ({insert_columns}) VALUES ({placeholders})"

        tolerant_columns = {"interceptz", "downhillspeed"}

        def _normalize_sqlite_value(column_name: str, value: Any):
            if isinstance(value, decimal.Decimal):
                # SQLite driver cannot bind Decimal directly; preserve integers when exact.
                try:
                    if value == value.to_integral_value():
                        return int(value)
                except Exception:
                    pass
                value = float(value)

            if value is None:
                return None

            if str(column_name).lower() in tolerant_columns:
                try:
                    # Canonicalize to 4 decimal places so exact SQLite equality behaves
                    # like a built-in +/-0.0001 tolerance for these lookup dimensions.
                    return round(float(value), 4)
                except Exception:
                    return value

            return value

        while True:
            batch = cursor.fetchmany(int(batch_size))
            if not batch:
                break
            normalized_batch = [
                tuple(
                    _normalize_sqlite_value(columnName, columnValue)
                    for columnName, columnValue in zip(column_names, row)
                )
                for row in batch
            ]
            with conn:
                conn.executemany(insert_sql, normalized_batch)

        cols = {c.lower() for c in column_names}
        rows_index_name = f"ix_{reference_table_name}_rows"
        match_index_name = f"ix_{reference_table_name}_match"
        with conn:
            if {"interceptrow", "bouncerow"}.issubset(cols):
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {quote_identifier(rows_index_name)} ON {safe_reference_table}(interceptRow, bounceRow)"
                )
            if {"interceptz", "spintoprpm", "spinsiderpm"}.issubset(cols):
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {quote_identifier(match_index_name)} ON {safe_reference_table}(interceptZ, spinTopRpm, spinSideRpm)"
                )
    finally:
        db.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, type=Path, help="Path to Trajectory4DLibrary005.pkl")
    ap.add_argument("--sqlite", default=None, type=Path, help="Output SQLite DB path")
    ap.add_argument(
        "--gen-reference-parquet",
        "--gen6-parquet",
        dest="gen_reference_parquet",
        type=Path,
        default=None,
        help="Optional GenXXReference.parquet path",
    )
    ap.add_argument("--batch-size", type=int, default=1000, help="Rows per insert batch (default: 1000)")
    ap.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Optional cap on trajectories processed (useful for smoke tests).",
    )
    ap.add_argument(
        "--checkpoint-pages",
        type=int,
        default=20000,
        help="Run WAL checkpoint after each batch when page count exceeds this value (default: 20000)",
    )
    args = ap.parse_args()

    with args.pkl.open("rb") as f:
        payload = pickle.load(f)

    trajectory_version = extract_trajectory_version_from_pkl(args.pkl)
    reference_version = extract_reference_version_from_parquet(args.gen_reference_parquet)
    trajectory_table_name = "points"
    reference_table_name = "references"
    if args.sqlite is None:
        args.sqlite = args.pkl.parent / f"Trajectory{trajectory_version}GamePlay{reference_version}.db"

    batch_size = max(1, int(args.batch_size))
    checkpoint_pages = max(0, int(args.checkpoint_pages))
    max_trajectories = None if args.max_trajectories is None else max(0, int(args.max_trajectories))
    entries = normalize_entries(payload)

    total_trajectories = _try_len(entries)
    if max_trajectories is not None:
        if total_trajectories is None:
            total_trajectories = int(max_trajectories)
        else:
            total_trajectories = min(total_trajectories, int(max_trajectories))

    args.sqlite.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.sqlite)
    try:
        create_tables(conn, trajectory_table_name=trajectory_table_name)

        processed = 0
        started = time.time()
        next_progress_percent = 10.0
        for row_batch in batched(
            iter_read_rows(entries, max_trajectories=max_trajectories),
            batch_size=batch_size,
        ):
            insert_batch(conn, row_batch, trajectory_table_name=trajectory_table_name)
            processed += len(row_batch)

            elapsed = max(1e-9, time.time() - started)
            rate = processed / elapsed

            if checkpoint_pages > 0:
                page_count = conn.execute("PRAGMA page_count").fetchone()[0]
                if int(page_count) >= checkpoint_pages:
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

            if total_trajectories and total_trajectories > 0:
                pct = min(100.0, (100.0 * processed) / float(total_trajectories))
                remaining = max(0, int(total_trajectories) - processed)
                eta = (remaining / rate) if rate > 1e-9 else float("inf")
                eta_text = "--:--" if not math.isfinite(eta) else _format_seconds(eta)
                should_report = processed >= int(total_trajectories) or pct >= next_progress_percent
                if should_report:
                    print(
                        f"Progress: {processed}/{total_trajectories} ({pct:.2f}%)"
                        f" | rate={rate:.1f} traj/s | elapsed={_format_seconds(elapsed)} | eta={eta_text}",
                        flush=True,
                    )
                    while next_progress_percent <= pct:
                        next_progress_percent += 10.0
            else:
                print(
                    f"Progress: {processed} trajectories"
                    f" | rate={rate:.1f} traj/s | elapsed={_format_seconds(elapsed)}",
                    flush=True,
                )

        if args.gen_reference_parquet is not None:
            import_gen_reference(
                conn,
                args.gen_reference_parquet,
                reference_table_name=reference_table_name,
            )

        cur = conn.cursor()
        n_traj = cur.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
        n_pts = cur.execute(f"SELECT COUNT(*) FROM {quote_identifier(trajectory_table_name)}").fetchone()[0]
        print(f"Export complete: {args.sqlite}")
        print(f"trajectories={n_traj}, points={n_pts}")
        if args.gen_reference_parquet is not None:
            n_ctx = cur.execute(f"SELECT COUNT(*) FROM {quote_identifier(reference_table_name)}").fetchone()[0]
            print(f"references={n_ctx}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

# Example command:
# python GamePlaySQLBuilder.py --pkl ../TrajectoryGenerator/Trajectory4DLibrary007.pkl --gen-reference-parquet Gen30GamePlay.parquet
# name of output db will be TrajectoryXXXReferenceXX.db
# you can override the db name by specifying --sqlite <db name>