import argparse
import os
import re
import time
from typing import List

import duckdb


CANONICAL_APEX_VALUES = [
    1.00, 1.25, 1.50, 1.80, 2.10, 2.40, 2.70,
    3.00, 3.30, 3.60, 4.50, 6.00, 8.00, 10.00,
]


def _parquet_has_column(connection: duckdb.DuckDBPyConnection, parquetPath: str, columnName: str) -> bool:
    try:
        schemaRows = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{parquetPath}')"
        ).fetchall()
    except Exception:
        return False

    columnNameLower = str(columnName).lower()
    for schemaRow in schemaRows:
        if schemaRow and str(schemaRow[0]).lower() == columnNameLower:
            return True

    return False


def _build_apex_snap_case_expression(columnName: str = "apexHeight") -> str:
    apexValues = [float(value) for value in CANONICAL_APEX_VALUES]
    if not apexValues:
        return f"CAST({columnName} AS DOUBLE)"

    if len(apexValues) == 1:
        return f"CAST({apexValues[0]:.2f} AS DOUBLE)"

    caseClauses: List[str] = ["CASE"]
    for index, currentValue in enumerate(apexValues):
        if index == 0:
            upperBound = (currentValue + apexValues[index + 1]) / 2.0
            caseClauses.append(
                f"WHEN {columnName} < {upperBound:.6f} THEN {currentValue:.2f}"
            )
        elif index == len(apexValues) - 1:
            lowerBound = (apexValues[index - 1] + currentValue) / 2.0
            caseClauses.append(
                f"WHEN {columnName} >= {lowerBound:.6f} THEN {currentValue:.2f}"
            )
        else:
            lowerBound = (apexValues[index - 1] + currentValue) / 2.0
            upperBound = (currentValue + apexValues[index + 1]) / 2.0
            caseClauses.append(
                f"WHEN {columnName} >= {lowerBound:.6f} AND {columnName} < {upperBound:.6f} THEN {currentValue:.2f}"
            )

    caseClauses.append(f"ELSE CAST({columnName} AS DOUBLE)")
    caseClauses.append("END")
    return "\n                    ".join(caseClauses)


def get_gameplay_selection_query(
    sourceParquetPath: str,
    topBounceCellsPerContext: int = 10,
    shotsPerBounceCell: int = 3,
    topServeBounceCellsPerContext: int = 10,
    serveShotsPerBounceCell: int = 5,
    minCountForSelection: int = 1,
    includeAllServeOptions: bool = False,
    hasWinnerColumn: bool = False,
    hasDefensiveColumns: bool = True,
    hasDownhillSpeedColumn: bool = True,
) -> str:
    topBounceCellLimit = max(1, int(topBounceCellsPerContext))
    shotLimit = max(1, int(shotsPerBounceCell))
    serveBounceCellLimit = max(1, int(topServeBounceCellsPerContext))
    serveShotsPerBounceCellLimit = max(1, int(serveShotsPerBounceCell))
    minimumCount = max(1, int(minCountForSelection))
    includeAllServeSqlFlag = "TRUE" if bool(includeAllServeOptions) else "FALSE"
    serveRowLimit = serveBounceCellLimit * serveShotsPerBounceCellLimit
    winnerOverrideClause = (
        "WHEN COALESCE(TRY_CAST(winner AS BOOLEAN), FALSE) THEN 1.0"
        if hasWinnerColumn
        else ""
    )
    winnerSelectColumn = (
        "COALESCE(TRY_CAST(winner AS BOOLEAN), FALSE) AS winner"
        if hasWinnerColumn
        else "FALSE AS winner"
    )
    defensiveSelectColumns = (
        "defensiveCol,\n"
        "            defensiveRow,"
        if hasDefensiveColumns
        else "CAST(NULL AS INTEGER) AS defensiveCol,\n"
             "            CAST(NULL AS INTEGER) AS defensiveRow,"
    )
    apexSnapExpression = _build_apex_snap_case_expression("apexHeight")
    downhillSelectExpression = (
        "COALESCE(TRY_CAST(downhillSpeed AS DOUBLE), 0.0) AS downhillSpeed"
        if hasDownhillSpeedColumn
        else "0.0 AS downhillSpeed"
    )

    return f"""
    WITH base AS (
        SELECT
            source.*,
            {downhillSelectExpression},
            (wins * 1.0 / count) AS rowWinPct,
            CASE
                {winnerOverrideClause}
                WHEN count >= 50 THEN (wins * 1.0 / count)
                ELSE 0.5 + (((wins * 1.0 / count) - 0.5) * POWER((count * 1.0 / 50.0), 2))
            END AS adjWinPctComputed
        FROM read_parquet('{sourceParquetPath}') AS source
        WHERE count >= {minimumCount}
          AND bounceRow BETWEEN 5 AND 22
    ),

    shape_base AS (
        SELECT *
        FROM base
        WHERE wins > 0
    ),

    shape_rows AS (
        SELECT
            sb.*,
            ({apexSnapExpression}) AS snappedApexHeight,
            CASE
                WHEN (
                    sb.interceptCol IN (8, 9)
                    AND sb.interceptRow = 5
                    AND sb.opponentCol = 5
                    AND sb.opponentRow IN (22, 23, 24)
                ) THEN 1
                WHEN (
                    sb.interceptCol IN (6, 7)
                    AND sb.interceptRow = 5
                    AND sb.opponentCol = 10
                    AND sb.opponentRow IN (22, 23, 24)
                ) THEN 2
                WHEN (
                    sb.interceptCol IN (6, 7)
                    AND sb.interceptRow = 22
                    AND sb.opponentCol = 10
                    AND sb.opponentRow IN (3, 4, 5)
                ) THEN 3
                WHEN (
                    sb.interceptCol IN (8, 9)
                    AND sb.interceptRow = 22
                    AND sb.opponentCol = 5
                    AND sb.opponentRow IN (3, 4, 5)
                ) THEN 4
                ELSE 0
            END AS serveCaseKey,
            CASE
                WHEN (
                    sb.interceptCol IN (8, 9)
                    AND sb.interceptRow = 5
                    AND sb.opponentCol = 5
                    AND sb.opponentRow IN (22, 23, 24)
                ) THEN (
                    sb.bounceCol IN (5, 6, 7)
                    AND sb.bounceRow BETWEEN 15 AND 18
                )
                WHEN (
                    sb.interceptCol IN (6, 7)
                    AND sb.interceptRow = 5
                    AND sb.opponentCol = 10
                    AND sb.opponentRow IN (22, 23, 24)
                ) THEN (
                    sb.bounceCol IN (8, 9, 10)
                    AND sb.bounceRow BETWEEN 15 AND 18
                )
                WHEN (
                    sb.interceptCol IN (6, 7)
                    AND sb.interceptRow = 22
                    AND sb.opponentCol = 10
                    AND sb.opponentRow IN (3, 4, 5)
                ) THEN (
                    sb.bounceCol IN (8, 9, 10)
                    AND sb.bounceRow BETWEEN 9 AND 12
                )
                WHEN (
                    sb.interceptCol IN (8, 9)
                    AND sb.interceptRow = 22
                    AND sb.opponentCol = 5
                    AND sb.opponentRow IN (3, 4, 5)
                ) THEN (
                    sb.bounceCol IN (5, 6, 7)
                    AND sb.bounceRow BETWEEN 9 AND 12
                )
                ELSE FALSE
            END AS isServeBoxForCase
        FROM shape_base sb
    ),

    rally_shape_rows AS (
        SELECT *
        FROM shape_rows
    ),

    serve_shape_rows AS (
        SELECT *
        FROM shape_rows
        WHERE serveCaseKey BETWEEN 1 AND 4
          AND isServeBoxForCase
    ),

    bounce_cell_stats_rally AS (
        SELECT
            interceptCol,
            interceptRow,
            interceptZ,
            opponentCol,
            opponentRow,
            downhillSpeed,
            bounceCol,
            bounceRow,
            MAX(CASE WHEN serveCaseKey BETWEEN 1 AND 4 THEN 1 ELSE 0 END) = 1 AS isServeContext,
            MAX(CASE WHEN isServeBoxForCase THEN 1 ELSE 0 END) = 1 AS isServeBox,
            SUM(wins) AS cellWins,
            SUM(count) AS cellCount,
            CASE
                WHEN SUM(count) > 0 THEN SUM(wins) * 1.0 / SUM(count)
                ELSE 0.0
            END AS cellWinPct,
            CASE
                WHEN SUM(count) > 0 THEN SUM(adjWinPctComputed * count) * 1.0 / SUM(count)
                ELSE 0.5
            END AS cellAdjWinPct
        FROM rally_shape_rows
        GROUP BY
            interceptCol,
            interceptRow,
            interceptZ,
            opponentCol,
            opponentRow,
            downhillSpeed,
            bounceCol,
            bounceRow
    ),

    ranked_bounce_cells_rally AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY
                    interceptCol,
                    interceptRow,
                    interceptZ,
                    opponentCol,
                    opponentRow,
                    downhillSpeed
                ORDER BY
                    cellAdjWinPct DESC,
                    cellWinPct DESC,
                    cellWins DESC,
                    cellCount DESC,
                    bounceRow,
                    bounceCol
            ) AS bounce_cell_rank
        FROM bounce_cell_stats_rally
    ),

    selected_bounce_cells_rally AS (
        SELECT *
        FROM ranked_bounce_cells_rally
        WHERE bounce_cell_rank <= {topBounceCellLimit}
           OR (
                {includeAllServeSqlFlag}
                AND isServeContext
                AND isServeBox
           )
    ),

    shape_stats_rally AS (
        SELECT
            sr.interceptCol,
            sr.interceptRow,
            sr.interceptZ,
            sr.opponentCol,
            sr.opponentRow,
            sr.downhillSpeed,
            sr.bounceCol,
            sr.bounceRow,
            sr.snappedApexHeight,
            SUM(wins) AS shapeWins,
            SUM(count) AS shapeCount,
            CASE
                WHEN SUM(count) > 0 THEN SUM(wins) * 1.0 / SUM(count)
                ELSE 0.0
            END AS shapeWinPct,
            CASE
                WHEN SUM(count) > 0 THEN SUM(adjWinPctComputed * count) * 1.0 / SUM(count)
                ELSE 0.5
            END AS shapeAdjWinPct
        FROM rally_shape_rows sr
        JOIN selected_bounce_cells_rally sbc
            ON sr.interceptCol = sbc.interceptCol
            AND sr.interceptRow = sbc.interceptRow
            AND sr.interceptZ = sbc.interceptZ
            AND sr.opponentCol = sbc.opponentCol
            AND sr.opponentRow = sbc.opponentRow
            AND sr.downhillSpeed = sbc.downhillSpeed
            AND sr.bounceCol = sbc.bounceCol
            AND sr.bounceRow = sbc.bounceRow
        GROUP BY
            sr.interceptCol,
            sr.interceptRow,
            sr.interceptZ,
            sr.opponentCol,
            sr.opponentRow,
            sr.downhillSpeed,
            sr.bounceCol,
            sr.bounceRow,
            sr.snappedApexHeight
    ),

    ranked_shapes_rally AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY
                    interceptCol,
                    interceptRow,
                    interceptZ,
                    opponentCol,
                    opponentRow,
                    downhillSpeed,
                    bounceCol,
                    bounceRow
                ORDER BY
                    shapeAdjWinPct DESC,
                    shapeWinPct DESC,
                    shapeWins DESC,
                    shapeCount DESC,
                    snappedApexHeight
            ) AS shape_rank_in_bounce
        FROM shape_stats_rally
    ),

    selected_shapes_rally AS (
        SELECT
            rs.*,
            sbc.cellWinPct,
            sbc.cellAdjWinPct
        FROM ranked_shapes_rally rs
        JOIN selected_bounce_cells_rally sbc
            ON rs.interceptCol = sbc.interceptCol
            AND rs.interceptRow = sbc.interceptRow
            AND rs.interceptZ = sbc.interceptZ
            AND rs.opponentCol = sbc.opponentCol
            AND rs.opponentRow = sbc.opponentRow
            AND rs.downhillSpeed = sbc.downhillSpeed
            AND rs.bounceCol = sbc.bounceCol
            AND rs.bounceRow = sbc.bounceRow
        WHERE rs.shape_rank_in_bounce <= {shotLimit}
    ),

    best_spin_per_shape_rally AS (
        SELECT
            ROW_NUMBER() OVER (
                PARTITION BY
                    sr.interceptCol,
                    sr.interceptRow,
                    sr.interceptZ,
                    sr.opponentCol,
                    sr.opponentRow,
                    sr.downhillSpeed,
                    sr.bounceCol,
                    sr.bounceRow,
                    sr.snappedApexHeight
                ORDER BY
                    sr.wins DESC,
                    sr.count DESC,
                    sr.adjWinPctComputed DESC,
                    sr.spinTopRpm,
                    sr.spinSideRpm
            ) AS spin_rank,
            sr.interceptCol,
            sr.interceptRow,
            sr.interceptZ,
            sr.opponentCol,
            sr.opponentRow,
            sr.downhillSpeed,
            sr.bounceCol,
            sr.bounceRow,
            sr.snappedApexHeight,
            sr.spinTopRpm,
            sr.spinSideRpm,
            sr.wins,
            sr.count,
            sr.rowWinPct,
            sr.adjWinPctComputed AS adjWinPct,
            sr.initialVelocity,
            sr.airTravelDistance,
            sr.netClearance,
            {winnerSelectColumn},
            {defensiveSelectColumns}
            ss.shapeWinPct,
            ss.shapeAdjWinPct,
            ss.cellWinPct,
            ss.cellAdjWinPct
        FROM rally_shape_rows sr
        JOIN selected_shapes_rally ss
            ON sr.interceptCol = ss.interceptCol
            AND sr.interceptRow = ss.interceptRow
            AND sr.interceptZ = ss.interceptZ
            AND sr.opponentCol = ss.opponentCol
            AND sr.opponentRow = ss.opponentRow
            AND sr.downhillSpeed = ss.downhillSpeed
            AND sr.bounceCol = ss.bounceCol
            AND sr.bounceRow = ss.bounceRow
            AND sr.snappedApexHeight = ss.snappedApexHeight
    ),

    selected_spin_rally AS (
        SELECT
            *
        FROM best_spin_per_shape_rally
        WHERE spin_rank = 1
    ),

    bounce_cell_stats_serve AS (
        SELECT
            serveCaseKey,
            bounceCol,
            bounceRow,
            SUM(wins) AS cellWins,
            SUM(count) AS cellCount,
            CASE
                WHEN SUM(count) > 0 THEN SUM(wins) * 1.0 / SUM(count)
                ELSE 0.0
            END AS cellWinPct,
            CASE
                WHEN SUM(count) > 0 THEN SUM(adjWinPctComputed * count) * 1.0 / SUM(count)
                ELSE 0.5
            END AS cellAdjWinPct
        FROM serve_shape_rows
        GROUP BY
            serveCaseKey,
            bounceCol,
            bounceRow
    ),

    ranked_bounce_cells_serve AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY serveCaseKey
                ORDER BY
                    cellAdjWinPct DESC,
                    cellWinPct DESC,
                    cellWins DESC,
                    cellCount DESC,
                    bounceRow,
                    bounceCol
            ) AS bounce_cell_rank
        FROM bounce_cell_stats_serve
    ),

    selected_bounce_cells_serve AS (
        SELECT *
        FROM ranked_bounce_cells_serve
        WHERE bounce_cell_rank <= {serveBounceCellLimit}
    ),

    shape_stats_serve AS (
        SELECT
            sr.serveCaseKey,
            sr.bounceCol,
            sr.bounceRow,
            sr.snappedApexHeight,
            SUM(sr.wins) AS shapeWins,
            SUM(sr.count) AS shapeCount,
            CASE
                WHEN SUM(sr.count) > 0 THEN SUM(sr.wins) * 1.0 / SUM(sr.count)
                ELSE 0.0
            END AS shapeWinPct,
            CASE
                WHEN SUM(sr.count) > 0 THEN SUM(sr.adjWinPctComputed * sr.count) * 1.0 / SUM(sr.count)
                ELSE 0.5
            END AS shapeAdjWinPct,
            MAX(
                CASE
                    WHEN ABS(sr.snappedApexHeight - ROUND(sr.interceptZ, 1)) <= 0.051
                         AND sr.downhillSpeed > 0.0
                    THEN 1
                    ELSE 0
                END
            ) AS hasDownhillAtInterceptApex
        FROM serve_shape_rows sr
        JOIN selected_bounce_cells_serve sbc
            ON sr.serveCaseKey = sbc.serveCaseKey
            AND sr.bounceCol = sbc.bounceCol
            AND sr.bounceRow = sbc.bounceRow
        GROUP BY
            sr.serveCaseKey,
            sr.bounceCol,
            sr.bounceRow,
            sr.snappedApexHeight
    ),

    ranked_shapes_serve AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY serveCaseKey, bounceCol, bounceRow
                ORDER BY
                    hasDownhillAtInterceptApex DESC,
                    shapeAdjWinPct DESC,
                    shapeWinPct DESC,
                    shapeWins DESC,
                    shapeCount DESC,
                    snappedApexHeight
            ) AS shape_rank_in_bounce
        FROM shape_stats_serve
    ),

    selected_shapes_serve AS (
        SELECT
            rs.*,
            sbc.cellWinPct,
            sbc.cellAdjWinPct,
            sbc.bounce_cell_rank
        FROM ranked_shapes_serve rs
        JOIN selected_bounce_cells_serve sbc
            ON rs.serveCaseKey = sbc.serveCaseKey
            AND rs.bounceCol = sbc.bounceCol
            AND rs.bounceRow = sbc.bounceRow
        WHERE rs.shape_rank_in_bounce <= {serveShotsPerBounceCellLimit}
    ),

    best_spin_per_shape_serve AS (
        SELECT
            ROW_NUMBER() OVER (
                PARTITION BY
                    sr.serveCaseKey,
                    sr.bounceCol,
                    sr.bounceRow,
                    sr.snappedApexHeight
                ORDER BY
                    CASE
                        WHEN ABS(sr.snappedApexHeight - ROUND(sr.interceptZ, 1)) <= 0.051
                             AND sr.downhillSpeed > 0.0
                        THEN 0
                        ELSE 1
                    END,
                    sr.wins DESC,
                    sr.count DESC,
                    sr.adjWinPctComputed DESC,
                    sr.spinTopRpm,
                    sr.spinSideRpm
            ) AS spin_rank,
            sr.interceptCol,
            sr.interceptRow,
            sr.interceptZ,
            sr.opponentCol,
            sr.opponentRow,
            sr.downhillSpeed,
            sr.bounceCol,
            sr.bounceRow,
            sr.snappedApexHeight,
            sr.spinTopRpm,
            sr.spinSideRpm,
            sr.wins,
            sr.count,
            sr.rowWinPct,
            sr.adjWinPctComputed AS adjWinPct,
            sr.initialVelocity,
            sr.airTravelDistance,
            sr.netClearance,
            {winnerSelectColumn},
            {defensiveSelectColumns}
            ss.shapeWinPct,
            ss.shapeAdjWinPct,
            ss.cellWinPct,
            ss.cellAdjWinPct,
            ss.serveCaseKey,
            ss.bounce_cell_rank,
            ss.shape_rank_in_bounce
        FROM serve_shape_rows sr
        JOIN selected_shapes_serve ss
            ON sr.serveCaseKey = ss.serveCaseKey
            AND sr.bounceCol = ss.bounceCol
            AND sr.bounceRow = ss.bounceRow
            AND sr.snappedApexHeight = ss.snappedApexHeight
    ),

    selected_spin_serve AS (
        SELECT
            *
        FROM best_spin_per_shape_serve
        WHERE spin_rank = 1
    ),

    capped_selected_spin_serve AS (
        SELECT
            *
        FROM (
            SELECT
                sss.*,
                ROW_NUMBER() OVER (
                    PARTITION BY sss.serveCaseKey
                    ORDER BY
                        sss.bounce_cell_rank,
                        sss.shape_rank_in_bounce,
                        sss.cellAdjWinPct DESC,
                        sss.shapeAdjWinPct DESC,
                        sss.adjWinPct DESC,
                        sss.wins DESC,
                        sss.count DESC
                ) AS serve_case_row_rank
            FROM selected_spin_serve sss
        ) ranked_serve
        WHERE serve_case_row_rank <= {serveRowLimit}
    ),

    serve_overlay_rows AS (
        SELECT
            s.*
        FROM capped_selected_spin_serve s
        WHERE NOT EXISTS (
            SELECT 1
            FROM selected_spin_rally r
            WHERE r.interceptCol = s.interceptCol
              AND r.interceptRow = s.interceptRow
              AND r.interceptZ = s.interceptZ
              AND r.opponentCol = s.opponentCol
              AND r.opponentRow = s.opponentRow
              AND r.downhillSpeed = s.downhillSpeed
              AND r.bounceCol = s.bounceCol
              AND r.bounceRow = s.bounceRow
              AND r.snappedApexHeight = s.snappedApexHeight
              AND r.spinTopRpm = s.spinTopRpm
              AND r.spinSideRpm = s.spinSideRpm
        )
    ),

    selected_spin_all AS (
        SELECT
            interceptCol,
            interceptRow,
            interceptZ,
            opponentCol,
            opponentRow,
            downhillSpeed,
            bounceCol,
            bounceRow,
            snappedApexHeight,
            spinTopRpm,
            spinSideRpm,
            defensiveCol,
            defensiveRow,
            initialVelocity,
            airTravelDistance,
            netClearance,
            wins,
            count,
            winner,
            rowWinPct,
            cellWinPct,
            adjWinPct,
            cellAdjWinPct
        FROM selected_spin_rally
        UNION ALL
        SELECT
            interceptCol,
            interceptRow,
            interceptZ,
            opponentCol,
            opponentRow,
            downhillSpeed,
            bounceCol,
            bounceRow,
            snappedApexHeight,
            spinTopRpm,
            spinSideRpm,
            defensiveCol,
            defensiveRow,
            initialVelocity,
            airTravelDistance,
            netClearance,
            wins,
            count,
            winner,
            rowWinPct,
            cellWinPct,
            adjWinPct,
            cellAdjWinPct
        FROM serve_overlay_rows
    )

    SELECT
        interceptCol,
        interceptRow,
        ROUND(interceptZ, 1) AS interceptZ,
        opponentCol,
        opponentRow,
        downhillSpeed,
        bounceCol,
        bounceRow,
        snappedApexHeight AS apexHeight,
        spinTopRpm,
        spinSideRpm,
        {defensiveSelectColumns}
        initialVelocity,
        airTravelDistance,
        netClearance,
        wins,
        count,
        winner,
        rowWinPct,
        cellWinPct,
        adjWinPct,
        cellAdjWinPct
    FROM selected_spin_all
    """


def build_gameplay_file(
    sourceParquetPath: str,
    outputParquetPath: str,
    topBounceCellsPerContext: int = 10,
    shotsPerBounceCell: int = 3,
    topServeBounceCellsPerContext: int = 10,
    serveShotsPerBounceCell: int = 5,
    minCountForSelection: int = 1,
    includeAllServeOptions: bool = False,
    duckdbThreads: int = 4,
) -> str:
    startedAt = time.time()

    def _elapsed_seconds() -> float:
        return max(0.0, time.time() - startedAt)

    def _log_stage(message: str) -> None:
        print(f"[GamePlayBuilder] {message} | elapsed={_elapsed_seconds():.1f}s")

    outputDir = os.path.dirname(outputParquetPath)
    if outputDir:
        os.makedirs(outputDir, exist_ok=True)

    tempOutputPath = os.path.abspath(outputParquetPath + ".tmp")
    configuredTempDir = os.environ.get("SELECTIVE_PRESSURE_DUCKDB_TEMP_DIR", "").strip()
    if configuredTempDir:
        duckdbTempDir = os.path.abspath(configuredTempDir)
    else:
        duckdbTempDir = os.path.abspath(os.path.join(outputDir if outputDir else ".", "duckdb_tmp"))
    os.makedirs(duckdbTempDir, exist_ok=True)

    connection = duckdb.connect()
    copyCompleted = False
    duckdbTempDirForPragma = duckdbTempDir.replace("\\", "/")
    configuredThreadCount = os.environ.get("SELECTIVE_PRESSURE_DUCKDB_THREADS", "").strip()
    configuredMaxTempDirSize = os.environ.get("SELECTIVE_PRESSURE_DUCKDB_MAX_TEMP_DIR_SIZE", "").strip()
    configuredMemoryLimit = os.environ.get("SELECTIVE_PRESSURE_DUCKDB_MEMORY_LIMIT", "").strip()
    configuredPreserveInsertionOrder = os.environ.get("SELECTIVE_PRESSURE_DUCKDB_PRESERVE_INSERTION_ORDER", "false").strip().lower()
    if configuredThreadCount:
        effectiveThreadCount = max(1, int(configuredThreadCount))
    else:
        effectiveThreadCount = max(1, min(int(duckdbThreads), 2))

    def _is_benign_temp_cleanup_error(errorText: str) -> bool:
        return (
            "Failed to delete file" in errorText
            and "duckdb_temp_storage" in errorText
        )

    try:
        connection.execute(f"PRAGMA threads = {effectiveThreadCount}")
        connection.execute(f"PRAGMA temp_directory='{duckdbTempDirForPragma}'")
        connection.execute(
            "SET preserve_insertion_order = {}".format(
                "true" if configuredPreserveInsertionOrder in ("1", "true", "yes", "on") else "false"
            )
        )
        if configuredMemoryLimit:
            connection.execute(f"SET memory_limit='{configuredMemoryLimit}'")
        if configuredMaxTempDirSize:
            connection.execute(f"PRAGMA max_temp_directory_size='{configuredMaxTempDirSize}'")

        print(
            "[GamePlayBuilder] DuckDB settings: "
            f"threads={effectiveThreadCount}, "
            f"temp_directory={duckdbTempDir}, "
            f"preserve_insertion_order={configuredPreserveInsertionOrder}, "
            f"memory_limit={configuredMemoryLimit or 'default'}, "
            f"max_temp_directory_size={configuredMaxTempDirSize or 'default'}"
        )
        _log_stage("DuckDB session configured")

        _log_stage("Inspecting source schema")
        hasWinnerColumn = _parquet_has_column(connection, sourceParquetPath, "winner")
        hasDefensiveColumns = (
            _parquet_has_column(connection, sourceParquetPath, "defensiveCol")
            and _parquet_has_column(connection, sourceParquetPath, "defensiveRow")
        )
        hasDownhillSpeedColumn = _parquet_has_column(connection, sourceParquetPath, "downhillSpeed")

        sourceRows = connection.execute(
            f"SELECT COUNT(*) FROM read_parquet('{sourceParquetPath}')"
        ).fetchone()[0]
        eligibleRows = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sourceParquetPath}')
            WHERE count >= {max(1, int(minCountForSelection))}
              AND bounceRow BETWEEN 5 AND 22
            """
        ).fetchone()[0]
        positiveWinRows = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sourceParquetPath}')
            WHERE count >= {max(1, int(minCountForSelection))}
              AND bounceRow BETWEEN 5 AND 22
              AND wins > 0
            """
        ).fetchone()[0]

        print(
            "[GamePlayBuilder] Source checkpoints: "
            f"rows={sourceRows:,}, "
            f"eligibleRows={eligibleRows:,}, "
            f"positiveWinRows={positiveWinRows:,}"
        )
        _log_stage("Source checkpoints calculated")

        _log_stage("Constructing gameplay selection SQL")
        query = get_gameplay_selection_query(
            sourceParquetPath=sourceParquetPath,
            topBounceCellsPerContext=topBounceCellsPerContext,
            shotsPerBounceCell=shotsPerBounceCell,
            topServeBounceCellsPerContext=topServeBounceCellsPerContext,
            serveShotsPerBounceCell=serveShotsPerBounceCell,
            minCountForSelection=minCountForSelection,
            includeAllServeOptions=includeAllServeOptions,
            hasWinnerColumn=hasWinnerColumn,
            hasDefensiveColumns=hasDefensiveColumns,
            hasDownhillSpeedColumn=hasDownhillSpeedColumn,
        )

        copySql = f"""
        COPY (
            {query}
        ) TO '{tempOutputPath}'
        (FORMAT PARQUET, COMPRESSION 'zstd');
        """

        _log_stage("Writing output parquet (this is the longest step)")
        connection.execute(copySql)

        outputRows = connection.execute(
            f"SELECT COUNT(*) FROM read_parquet('{tempOutputPath}')"
        ).fetchone()[0]
        print(f"[GamePlayBuilder] Output checkpoints: rows={outputRows:,}")
        _log_stage("Output parquet write complete")
        copyCompleted = True
    except duckdb.IOException as ex:
        if _is_benign_temp_cleanup_error(str(ex)) and os.path.exists(tempOutputPath):
            copyCompleted = True
        else:
            if os.path.exists(tempOutputPath):
                os.remove(tempOutputPath)
            raise
    except Exception:
        if os.path.exists(tempOutputPath):
            os.remove(tempOutputPath)
        raise
    finally:
        try:
            connection.close()
        except duckdb.IOException as ex:
            if not (copyCompleted and _is_benign_temp_cleanup_error(str(ex))):
                raise

    os.replace(tempOutputPath, outputParquetPath)
    return outputParquetPath


def _parse_generation_from_path(pathValue: str):
    fileName = os.path.basename(pathValue)
    match = re.match(r"^ConsolidatedGen(\d+)(?:_.*)?\.parquet$", fileName)
    if not match:
        return None
    return int(match.group(1))


def _resolve_output_path(sourcePath: str, outputPath: str | None) -> str:
    if outputPath:
        return outputPath

    generation = _parse_generation_from_path(sourcePath)
    if generation is None:
        return os.path.join(os.path.dirname(sourcePath), "ManualGamePlay.parquet")

    return os.path.join(os.path.dirname(sourcePath), f"Gen{generation}GamePlay.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic gameplay parquet from a consolidated parquet. "
            "Selection ranks bounce cells per context, keeps the top bounce cells, "
            "then keeps up to N shapes per selected bounce cell using the best spin "
            "variant per selected shape."
        )
    )
    parser.add_argument("sourceParquet", help="Path to ConsolidatedGenXX.parquet")
    parser.add_argument(
        "--outputParquet",
        default=None,
        help="Optional output path (default: GenXXGamePlay.parquet beside source)",
    )
    parser.add_argument(
        "--topBounceCellsPerContext",
        type=int,
        default=10,
        help="Top bounce cells per context (default: 10)",
    )
    parser.add_argument(
        "--shotsPerBounceCell",
        type=int,
        default=3,
        help="Max selected shapes kept per selected bounce cell (default: 3)",
    )
    parser.add_argument(
        "--topServeBounceCellsPerContext",
        type=int,
        default=10,
        help="Top serve bounce cells per serve context (default: 10)",
    )
    parser.add_argument(
        "--serveShotsPerBounceCell",
        type=int,
        default=5,
        help="Max selected serve shapes kept per selected serve bounce cell (default: 5)",
    )
    parser.add_argument(
        "--minCountForSelection",
        type=int,
        default=12,
        help="Minimum consolidated count required for gameplay selection (default: 12)",
    )
    parser.add_argument(
        "--includeAllServeOptions",
        action="store_true",
        help=(
            "Include all serve-box bounce cells for serve contexts "
            "(in addition to topBounceCellsPerContext)."
        ),
    )
    parser.add_argument(
        "--duckdbThreads",
        type=int,
        default=4,
        help="DuckDB thread hint (default: 4; may be clamped by builder)",
    )
    args = parser.parse_args()

    sourcePath = args.sourceParquet
    if not os.path.exists(sourcePath):
        raise FileNotFoundError(f"Source parquet not found: {sourcePath}")

    outputPath = _resolve_output_path(sourcePath, args.outputParquet)

    print("[GamePlay] Building gameplay parquet")
    print(f"[GamePlay] source={sourcePath}")
    print(f"[GamePlay] output={outputPath}")
    print(f"[GamePlay] topBounceCellsPerContext={args.topBounceCellsPerContext}")
    print(f"[GamePlay] shotsPerBounceCell={args.shotsPerBounceCell}")
    print(f"[GamePlay] topServeBounceCellsPerContext={args.topServeBounceCellsPerContext}")
    print(f"[GamePlay] serveShotsPerBounceCell={args.serveShotsPerBounceCell}")
    print(f"[GamePlay] minCountForSelection={args.minCountForSelection}")
    print(f"[GamePlay] includeAllServeOptions={args.includeAllServeOptions}")

    build_gameplay_file(
        sourceParquetPath=sourcePath,
        outputParquetPath=outputPath,
        topBounceCellsPerContext=args.topBounceCellsPerContext,
        shotsPerBounceCell=args.shotsPerBounceCell,
        topServeBounceCellsPerContext=args.topServeBounceCellsPerContext,
        serveShotsPerBounceCell=args.serveShotsPerBounceCell,
        minCountForSelection=args.minCountForSelection,
        includeAllServeOptions=args.includeAllServeOptions,
        duckdbThreads=args.duckdbThreads,
    )

    print(f"[GamePlay] completed: {outputPath}")


if __name__ == "__main__":
    main()

# python GamePlayParquetBuilder.py ConsolidatedGen31.parquet --outputParquet Gen31GamePlay.parquet --topBounceCellsPerContext 10 --shotsPerBounceCell 5 --topServeBounceCellsPerContext 10 --serveShotsPerBounceCell 5 --minCountForSelection 1 --includeAllServeOptions