import duckdb
import random
import os
import numpy as np
from typing import Dict, Tuple, List, Optional, Set
from FenceGridIndexer import XyToCell


BaseContextKey = Tuple[int, int, float, int, int]
ContextKey = Tuple[int, int, float, int, int, float]
BounceKey = Tuple[int, int]
Trajectic = Tuple[float, int, int, int, int, float]


def _ComputeSpeedAtSample(X, Y, Z, T, sampleIndex: int) -> float:
    sampleCount = len(T)
    if sampleCount < 2:
        return 0.0

    if sampleIndex <= 0:
        startIndex, endIndex = 0, 1
    elif sampleIndex >= sampleCount - 1:
        startIndex, endIndex = sampleCount - 2, sampleCount - 1
    else:
        startIndex, endIndex = sampleIndex - 1, sampleIndex + 1

    timeDelta = float(T[endIndex] - T[startIndex])
    if timeDelta <= 1e-9:
        return 0.0

    deltaX = float(X[endIndex] - X[startIndex])
    deltaY = float(Y[endIndex] - Y[startIndex])
    deltaZ = float(Z[endIndex] - Z[startIndex])

    return float(np.sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ) / timeDelta)


def _FindInPlayEndIndex(Z, bounceIndex: int, epsilon: float = 1e-6) -> int:
    """
    Return the last in-play sample index:
    first post-bounce ground contact (z <= 0) after the post-bounce apex,
    or the final sample.
    """
    sampleCount = len(Z)
    if sampleCount == 0:
        return -1

    bounceIndexClamped = max(0, min(int(bounceIndex), sampleCount - 1))
    postBounceSamples = Z[bounceIndexClamped:]
    if len(postBounceSamples) <= 1:
        return sampleCount - 1

    apexRelativeIndex = int(np.argmax(postBounceSamples))
    apexIndex = bounceIndexClamped + apexRelativeIndex
    if apexIndex >= sampleCount - 1:
        return sampleCount - 1

    secondGroundContactIndices = np.where(Z[apexIndex + 1:] <= float(epsilon))[0]
    if secondGroundContactIndices.size == 0:
        return sampleCount - 1

    return int(apexIndex + 1 + secondGroundContactIndices[0])


# A TrajectIC or trajectic is a Trajectory in Context
# A trajectory that starts at a specific interceptPoint on the player's side
# with a specific opponent position on the opponent's side of the court
class TrajecticsSelector:
    CanonicalApexValues = [
        1.00, 1.25, 1.50, 1.80, 2.10, 2.40, 2.70,
        3.00, 3.30, 3.60, 4.50, 6.00, 8.00, 10.00,
    ]

    def __init__(
        self,
        parquetPath: str,
        minimumWinPercentage: float = 0.5,
        minimumCount: int = 3,
        court=None,
        netBoundaryRow: Optional[int] = None,
        isPreFilteredReference: bool = False,
        debug: bool = False,
        debugLevel1: bool = False
    ):
        self.parquetPath = parquetPath
        self.minimumWinPercentage = minimumWinPercentage
        self.minimumCount = minimumCount
        self.court = court
        self.debug = debug
        self.debugLevel1 = debugLevel1
        self.isPreFilteredReference = isPreFilteredReference
        self.trajecticsByContext: Dict[ContextKey, Dict[BounceKey, List[Trajectic]]] = {}
        self.availableDownhillSpeedsByBaseContext: Dict[BaseContextKey, Set[float]] = {}
        self.interceptCellAdjWinPctStatsByOpponentAndIntercept: Dict[
            Tuple[int, int, int, int],
            Tuple[float, int],
        ] = {}
        self.interceptCellMinRowCountByOpponentAndIntercept: Dict[
            Tuple[int, int, int, int],
            int,
        ] = {}

        gridRowsValue = int(getattr(court, "gridRows", 26)) if court is not None else 26
        self.netBoundaryRow = int(netBoundaryRow) if netBoundaryRow is not None else max(1, gridRowsValue // 2)

        if court is not None and hasattr(court, "GetRowCenterY") and hasattr(court, "netY"):
            northRows = [
                rowIndex
                for rowIndex in range(1, gridRowsValue + 1)
                if float(court.GetRowCenterY(rowIndex)) < float(court.netY)
            ]
            if northRows:
                self.netBoundaryRow = int(max(northRows))

        self.LoadTrajectics()

    def _IsNorthSideRow(self, row: int) -> bool:
        if self.court is not None and hasattr(self.court, "GetRowCenterY") and hasattr(self.court, "netY"):
            rowCenterY = float(self.court.GetRowCenterY(int(row)))
            return rowCenterY < float(self.court.netY)
        return int(row) <= int(self.netBoundaryRow)

    def _isBounceAcrossNet(self, interceptRow: int, bounceRow: int) -> bool:
        # Enforce bounce on opposite side of net from intercept side.
        return self._IsNorthSideRow(interceptRow) != self._IsNorthSideRow(bounceRow)

    def _isBounceInsideCourtRows(self, bounceRow: int) -> bool:
        """
        Keep bounce rows whose row centers are between the two baselines.
        Falls back to historical 4..22 rows if court geometry helpers are unavailable.
        """
        rowValue = int(bounceRow)

        if self.court is not None and hasattr(self.court, "GetRowCenterY"):
            try:
                rowCenterY = float(self.court.GetRowCenterY(rowValue))
                minimumY = float(min(self.court.serverBaselineY, self.court.receiverBaselineY))
                maximumY = float(max(self.court.serverBaselineY, self.court.receiverBaselineY))
                epsilon = 1e-6
                return (minimumY - epsilon) <= rowCenterY <= (maximumY + epsilon)
            except Exception:
                pass

        return 5 <= rowValue <= 22

    def _RandomDefensiveCellForOpponentRow(self, opponentRow: int) -> Tuple[int, int]:
        # Keep generated defensive cells in historical singles-style bounds.
        minimumCol = 5
        maximumCol = 10
        minimumRow = 5
        maximumRow = 22

        if minimumCol > maximumCol:
            minimumCol, maximumCol = maximumCol, minimumCol

        targetIsNorth = self._IsNorthSideRow(int(opponentRow))
        candidateRows = [
            rowValue
            for rowValue in range(minimumRow, maximumRow + 1)
            if self._isBounceInsideCourtRows(rowValue)
            and self._IsNorthSideRow(rowValue) == targetIsNorth
        ]

        if not candidateRows:
            candidateRows = [
                rowValue
                for rowValue in range(minimumRow, maximumRow + 1)
                if self._isBounceInsideCourtRows(rowValue)
            ]

        if not candidateRows:
            candidateRows = [int(opponentRow)]

        defensiveRow = int(random.choice(candidateRows))
        defensiveCol = int(random.randint(minimumCol, maximumCol))
        return defensiveCol, defensiveRow

    @staticmethod
    def _NormalizeDownhillSpeed(downhillSpeed: object) -> float:
        try:
            return round(float(downhillSpeed), 4)
        except Exception:
            return 0.0

    @staticmethod
    def BuildReferenceFile(sourceParquetPath: str, outputParquetPath: str, duckdbThreads: int = 4) -> str:
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
                "[SelectivePressure] DuckDB settings: "
                f"threads={effectiveThreadCount}, "
                f"temp_directory={duckdbTempDir}, "
                f"preserve_insertion_order={configuredPreserveInsertionOrder}, "
                f"memory_limit={configuredMemoryLimit or 'default'}, "
                f"max_temp_directory_size={configuredMaxTempDirSize or 'default'}"
            )
            hasWinnerColumn = TrajecticsSelector._ParquetHasColumn(
                connection,
                sourceParquetPath,
                "winner",
            )
            hasDefensiveColumns = (
                TrajecticsSelector._ParquetHasColumn(connection, sourceParquetPath, "defensiveCol")
                and TrajecticsSelector._ParquetHasColumn(connection, sourceParquetPath, "defensiveRow")
            )
            hasDownhillSpeedColumn = TrajecticsSelector._ParquetHasColumn(connection, sourceParquetPath, "downhillSpeed")
            query = TrajecticsSelector.GetReferenceSelectionQuery(
                sourceParquetPath,
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
            connection.execute(copySql)
            copyCompleted = True
        except duckdb.IOException as ex:
            # DuckDB can throw a benign temp cleanup error on Windows after successful copy.
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

    @staticmethod
    def _ParquetHasColumn(connection: duckdb.DuckDBPyConnection, parquetPath: str, columnName: str) -> bool:
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

    @staticmethod
    def _BuildApexSnapCaseExpression(columnName: str = "apexHeight") -> str:
        apexValues = [float(value) for value in TrajecticsSelector.CanonicalApexValues]
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

    @staticmethod
    def GetReferenceSelectionQuery(
        sourceParquetPath: str,
        hasWinnerColumn: bool = False,
        hasDefensiveColumns: bool = True,
        hasDownhillSpeedColumn: bool = True,
    ) -> str:
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
        apexSnapExpression = TrajecticsSelector._BuildApexSnapCaseExpression("apexHeight")
        downhillSelectExpression = (
            "COALESCE(TRY_CAST(downhillSpeed AS DOUBLE), 0.0) AS downhillSpeed"
            if hasDownhillSpeedColumn
            else "0.0 AS downhillSpeed"
        )

        return f"""
        WITH base AS (
            SELECT
                *,
                {downhillSelectExpression},
                (wins * 1.0 / count) AS rowWinPct,
                CASE
                    {winnerOverrideClause}
                    WHEN count >= 50 THEN (wins * 1.0 / count)
                    ELSE 0.5 + (((wins * 1.0 / count) - 0.5) * POWER((count * 1.0 / 50.0), 2))
                END AS adjWinPctComputed
            FROM read_parquet('{sourceParquetPath}')
            WHERE wins > 0
              AND count >= 1
              AND bounceRow BETWEEN 5 AND 22
        ),

        winning_base AS (
            SELECT *
            FROM base
            WHERE rowWinPct >= 0.5
        ),

        context_bounce_stats AS (
            SELECT
                interceptCol,
                interceptRow,
                interceptZ,
                opponentCol,
                opponentRow,
                downhillSpeed,
                bounceCol,
                bounceRow,
                SUM(wins) AS totalWins,
                SUM(count) AS totalCount,
                AVG(adjWinPctComputed) AS adjWinPct
            FROM winning_base
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

        ranked_cells AS (
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
                    ORDER BY adjWinPct DESC, RANDOM()
                ) AS quality_rank
            FROM context_bounce_stats
        ),

        top_cells AS (
            SELECT *
            FROM ranked_cells
            WHERE quality_rank <= 4
        ),

        random_extra_candidates AS (
            SELECT *
            FROM (
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
                        ORDER BY RANDOM()
                    ) AS random_rank
                FROM ranked_cells
                WHERE quality_rank > 4
            )
            WHERE random_rank = 1
        ),

        random_extra_cell AS (
            SELECT
                interceptCol,
                interceptRow,
                interceptZ,
                opponentCol,
                opponentRow,
                downhillSpeed,
                bounceCol,
                bounceRow,
                totalWins,
                totalCount,
                adjWinPct,
                quality_rank
            FROM random_extra_candidates
        ),

        selected_cells AS (
            SELECT * FROM top_cells
            UNION
            SELECT * FROM random_extra_cell
        ),

        joined AS (
            SELECT
                b.*,
                ({apexSnapExpression}) AS snappedApexHeight,
                b.adjWinPctComputed AS adjWinPct
            FROM winning_base b
            JOIN selected_cells c
                ON b.interceptCol = c.interceptCol
                AND b.interceptRow = c.interceptRow
                AND b.interceptZ = c.interceptZ
                AND b.opponentCol = c.opponentCol
                AND b.opponentRow = c.opponentRow
                AND b.downhillSpeed = c.downhillSpeed
                AND b.bounceCol = c.bounceCol
                AND b.bounceRow = c.bounceRow
        ),

        apex_dedup AS (
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
                        bounceRow,
                        snappedApexHeight
                    ORDER BY adjWinPct DESC
                ) AS apex_rank
            FROM joined
        ),

        apex_unique AS (
            SELECT *
            FROM apex_dedup
            WHERE apex_rank = 1
        ),

        ranked AS (
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
                    ORDER BY adjWinPct DESC
                ) AS trajectic_rank
            FROM apex_unique
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
            {winnerSelectColumn},
            adjWinPct
        FROM ranked
        WHERE trajectic_rank <= 3
        """

    def LoadTrajectics(self) -> None:
        connection = duckdb.connect()

        # Keep DuckDB thread count bounded to prevent memory pressure spikes.
        duckdbThreads = max(2, min(12, os.cpu_count() or 4))
        connection.execute(f"PRAGMA threads = {duckdbThreads}")

        hasDefensiveColumns = (
            self._ParquetHasColumn(connection, self.parquetPath, "defensiveCol")
            and self._ParquetHasColumn(connection, self.parquetPath, "defensiveRow")
        )
        hasDownhillSpeedColumn = self._ParquetHasColumn(connection, self.parquetPath, "downhillSpeed")
        downhillSelectExpression = (
            "COALESCE(TRY_CAST(downhillSpeed AS DOUBLE), 0.0) AS downhillSpeed"
            if hasDownhillSpeedColumn
            else "0.0 AS downhillSpeed"
        )

        if self.isPreFilteredReference:
            if hasDefensiveColumns:
                cursor = connection.execute(
                    f"""
                    SELECT
                        interceptCol,
                        interceptRow,
                        ROUND(interceptZ, 1) AS interceptZ,
                        opponentCol,
                        opponentRow,
                        {downhillSelectExpression},
                        bounceCol,
                        bounceRow,
                        apexHeight,
                        spinTopRpm,
                        spinSideRpm,
                        defensiveCol,
                        defensiveRow,
                        initialVelocity,
                        airTravelDistance,
                        netClearance,
                        wins,
                        count,
                        adjWinPct
                    FROM read_parquet('{self.parquetPath}')
                    """
                )
            else:
                cursor = connection.execute(
                    f"""
                    SELECT
                        interceptCol,
                        interceptRow,
                        ROUND(interceptZ, 1) AS interceptZ,
                        opponentCol,
                        opponentRow,
                        {downhillSelectExpression},
                        bounceCol,
                        bounceRow,
                        apexHeight,
                        spinTopRpm,
                        spinSideRpm,
                        CAST(NULL AS INTEGER) AS defensiveCol,
                        CAST(NULL AS INTEGER) AS defensiveRow,
                        initialVelocity,
                        airTravelDistance,
                        netClearance,
                        wins,
                        count,
                        adjWinPct
                    FROM read_parquet('{self.parquetPath}')
                    """
                )
        else:
            hasWinnerColumn = self._ParquetHasColumn(connection, self.parquetPath, "winner")
            cursor = connection.execute(
                self.GetReferenceSelectionQuery(
                    self.parquetPath,
                    hasWinnerColumn=hasWinnerColumn,
                    hasDefensiveColumns=hasDefensiveColumns,
                    hasDownhillSpeedColumn=hasDownhillSpeedColumn,
                )
            )

        batch_size = 100_000   # tune if needed
        total_rows = 0

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            total_rows += len(rows)

            for (
                interceptCol,
                interceptRow,
                interceptZ,
                opponentCol,
                opponentRow,
                downhillSpeed,
                bounceCol,
                bounceRow,
                apexHeight,
                spinTopRpm,
                spinSideRpm,
                defensiveCol,
                defensiveRow,
                initialVelocity,
                airTravelDistance,
                netClearance,
                wins,
                count,
                adjWinPct
            ) in rows:

                def _is_missing_numeric(value: object) -> bool:
                    if value is None:
                        return True
                    try:
                        return bool(np.isnan(value))
                    except Exception:
                        return False

                if _is_missing_numeric(defensiveCol) or _is_missing_numeric(defensiveRow):
                    defensiveColValue, defensiveRowValue = self._RandomDefensiveCellForOpponentRow(int(opponentRow))
                else:
                    defensiveColValue = int(defensiveCol)
                    defensiveRowValue = int(defensiveRow)

                contextKey = (
                    int(interceptCol),
                    int(interceptRow),
                    round(float(interceptZ), 1),
                    int(opponentCol),
                    int(opponentRow),
                    self._NormalizeDownhillSpeed(downhillSpeed),
                )

                baseContextKey = (
                    int(interceptCol),
                    int(interceptRow),
                    round(float(interceptZ), 1),
                    int(opponentCol),
                    int(opponentRow),
                )
                self.availableDownhillSpeedsByBaseContext.setdefault(baseContextKey, set()).add(
                    self._NormalizeDownhillSpeed(downhillSpeed)
                )

                contextEntry = self.trajecticsByContext.setdefault(contextKey, {})
                bounceKey = (int(bounceCol), int(bounceRow))
                trajecticList = contextEntry.setdefault(bounceKey, [])
                trajecticList.append(
                    (
                        round(float(apexHeight), 2),
                        int(spinTopRpm),
                        int(spinSideRpm),
                        defensiveColValue,
                        defensiveRowValue,
                        round(float(downhillSpeed), 4),
                    )
                )

                interceptStatKey = (
                    int(opponentCol),
                    int(opponentRow),
                    int(interceptCol),
                    int(interceptRow),
                )
                priorAdjWinPctSum, priorAdjWinPctCount = self.interceptCellAdjWinPctStatsByOpponentAndIntercept.get(
                    interceptStatKey,
                    (0.0, 0),
                )
                self.interceptCellAdjWinPctStatsByOpponentAndIntercept[interceptStatKey] = (
                    float(priorAdjWinPctSum) + float(adjWinPct),
                    int(priorAdjWinPctCount) + 1,
                )

                priorMinRowCount = self.interceptCellMinRowCountByOpponentAndIntercept.get(interceptStatKey)
                currentRowCount = int(count)
                if priorMinRowCount is None:
                    self.interceptCellMinRowCountByOpponentAndIntercept[interceptStatKey] = currentRowCount
                else:
                    self.interceptCellMinRowCountByOpponentAndIntercept[interceptStatKey] = min(
                        int(priorMinRowCount),
                        currentRowCount,
                    )

        print(f"Rows Count: {total_rows:,}")

        connection.close()

        print(
            f"[TrajecticSelector] Loaded {len(self.trajecticsByContext):,} contexts "
            f"({len(self.availableDownhillSpeedsByBaseContext):,} base contexts)"
        )

    def _GetTopInterceptCellsByWinPercentageFromCache(
        self,
        opponentCol: int,
        opponentRow: int,
        candidateCells: List[Tuple[int, int]],
        topCount: int,
    ) -> List[Tuple[int, int, float]]:
        if not candidateCells:
            return []

        rankedCells: List[Tuple[int, int, float]] = []
        for interceptColValue, interceptRowValue in sorted(set(candidateCells)):
            interceptStatKey = (
                int(opponentCol),
                int(opponentRow),
                int(interceptColValue),
                int(interceptRowValue),
            )
            totalAdjWinPctValue, totalAdjWinPctCount = self.interceptCellAdjWinPctStatsByOpponentAndIntercept.get(
                interceptStatKey,
                (0.0, 0),
            )
            if int(totalAdjWinPctCount) <= 0:
                continue

            winPercentageValue = float(totalAdjWinPctValue) / float(totalAdjWinPctCount)
            rankedCells.append((int(interceptColValue), int(interceptRowValue), float(winPercentageValue)))

        rankedCells.sort(key=lambda item: item[2], reverse=True)
        return rankedCells[: max(1, int(topCount))]

    def _GetLowCountCellSetFromCache(
        self,
        opponentCol: int,
        opponentRow: int,
        candidateCells: List[Tuple[int, int]],
        countThreshold: int = 10,
    ) -> Set[Tuple[int, int]]:
        if not candidateCells:
            return set()

        lowCountCellSet: Set[Tuple[int, int]] = set()
        for interceptColValue, interceptRowValue in set(candidateCells):
            interceptStatKey = (
                int(opponentCol),
                int(opponentRow),
                int(interceptColValue),
                int(interceptRowValue),
            )
            minimumObservedRowCount = self.interceptCellMinRowCountByOpponentAndIntercept.get(interceptStatKey)
            if minimumObservedRowCount is not None and 0 < int(minimumObservedRowCount) < int(countThreshold):
                lowCountCellSet.add((int(interceptColValue), int(interceptRowValue)))

        return lowCountCellSet

    def SampleIntercept(
        self,
        transformed: dict,
        defenderSide: str,
        defenderPos: Tuple[float, float],
        court,
        movementModel,
        opponentContextPos: Optional[Tuple[float, float]],
        topInterceptCellCount: int = 1,
        debug: bool = False,
    ):
        """
        Reachability sampling:
          1) Find nearest sample on defender half.
          2) Compute ToF_perp.
          3) Build reach radius.
          4) Filter legal candidate samples.
          5) Uniformly sample one candidate.

        Returns:
          (selectedInterceptPoint, nearestInterceptPoint) or (None, None)
        """
        try:
            X = transformed["fencesX"]
            Y = transformed["fencesY"]
            Z = transformed["fencesZ"]
            T = transformed["time"]
            bounceIndex = int(transformed.get("bounceIndex", 0))

            defenderX, defenderY = float(defenderPos[0]), float(defenderPos[1])

            netY = court.netY

            playerVelocity = float(getattr(movementModel, "playerSpeed", court.playerSpeed))
            playerReactionTime = float(getattr(movementModel, "reactionTime", court.playerReactionTime))
            reachableHeightMin = float(getattr(movementModel, "reachZMin", court.playerReachZMin))
            reachableHeightMax = float(getattr(movementModel, "reachZMax", court.playerReachZMax))

            half_mask = (Y <= netY) if (defenderSide == "PLAYER_BLUE") else (Y >= netY)

            inPlayEndIndex = _FindInPlayEndIndex(Z, bounceIndex)
            inPlayMask = np.arange(len(Z), dtype=int) <= inPlayEndIndex
            half_mask = half_mask & inPlayMask

            idx_half = np.where(half_mask)[0]
            if idx_half.size == 0:
                return None, None

            trajectoryPointsOnDefenderSide = np.hypot(X[idx_half] - defenderX, Y[idx_half] - defenderY)
            nearestIndexRelative = int(np.argmin(trajectoryPointsOnDefenderSide))
            nearestInterceptPointIndex = int(idx_half[nearestIndexRelative])

            nearestInterceptX = float(X[nearestInterceptPointIndex])
            nearestInterceptY = float(Y[nearestInterceptPointIndex])
            nearestInterceptT = float(T[nearestInterceptPointIndex])
            nearestInterceptSpeed = _ComputeSpeedAtSample(X, Y, Z, T, nearestInterceptPointIndex)

            defenderInterceptRadius = max(0.0, nearestInterceptT - playerReactionTime) * playerVelocity

            in_z = (Z >= reachableHeightMin) & (Z <= reachableHeightMax)
            distanceToIntercept = np.hypot(X - defenderX, Y - defenderY)
            bucket_mask = half_mask & in_z & (distanceToIntercept <= defenderInterceptRadius)
            picks = np.where(bucket_mask)[0]
            rawPickIndices = np.asarray(picks, dtype=int)
            rawPicksCount = int(picks.size)

            if picks.size == 0:
                if debug:
                    print(
                        f"[SampleIntercept] defenderSide={defenderSide} rawPicks=0 "
                        f"opponentCell={XyToCell(defenderX, defenderY, court)}"
                    )
                return None, None

            if opponentContextPos is not None:
                contextX, contextY = float(opponentContextPos[0]), float(opponentContextPos[1])
            else:
                contextX, contextY = defenderX, defenderY

            opponentCol, opponentRow = XyToCell(contextX, contextY, court)
            defenderCol, defenderRow = XyToCell(defenderX, defenderY, court)
            candidateCells = [XyToCell(float(X[sampleIndex]), float(Y[sampleIndex]), court) for sampleIndex in picks]
            topCells = self._GetTopInterceptCellsByWinPercentageFromCache(
                opponentCol=opponentCol,
                opponentRow=opponentRow,
                candidateCells=candidateCells,
                topCount=topInterceptCellCount,
            )

            if topCells:
                topCellSet = {(interceptColValue, interceptRowValue) for interceptColValue, interceptRowValue, _ in topCells}
                filteredPicks = [
                    sampleIndex
                    for sampleIndex in picks
                    if XyToCell(float(X[sampleIndex]), float(Y[sampleIndex]), court) in topCellSet
                ]
                if filteredPicks:
                    picks = np.asarray(filteredPicks, dtype=int)

            if debug:
                rawPickCandidateCells = [
                    XyToCell(float(X[sampleIndex]), float(Y[sampleIndex]), court)
                    for sampleIndex in rawPickIndices
                ]
                lowCountCellSet = self._GetLowCountCellSetFromCache(
                    opponentCol=opponentCol,
                    opponentRow=opponentRow,
                    candidateCells=rawPickCandidateCells,
                    countThreshold=10,
                )
                lowCountPickCount = sum(
                    1 for candidateCell in rawPickCandidateCells if candidateCell in lowCountCellSet
                )
                rawPickCount = len(rawPickCandidateCells)
                lowCountPickPercentage = (
                    (100.0 * float(lowCountPickCount) / float(rawPickCount))
                    if rawPickCount > 0
                    else 0.0
                )

                if topCells:
                    rankedText = ", ".join(
                        [
                            f"({interceptColValue},{interceptRowValue})={winPercentageValue:.3f}"
                            for interceptColValue, interceptRowValue, winPercentageValue in topCells
                        ]
                    )
                else:
                    rankedText = "none"

                sourceName = os.path.basename(self.parquetPath) if self.parquetPath else "none"

                print(
                    f"[SampleIntercept] defenderSide={defenderSide} defenderCell=({defenderCol},{defenderRow}) "
                    f"opponentCell=({opponentCol},{opponentRow}) "
                    f"rawPicks={rawPicksCount} source={sourceName} topCells=[{rankedText}] "
                    f"filteredPicks={int(picks.size)} "
                    f"rawPickCountLt10Pct={lowCountPickPercentage:.1f} "
                    f"({lowCountPickCount}/{rawPickCount})"
                )

            selectedSampleIndex = int(np.random.choice(picks))

            interceptPickX = float(X[selectedSampleIndex])
            interceptPickY = float(Y[selectedSampleIndex])
            interceptPickZ = float(Z[selectedSampleIndex])
            interceptPickSpeed = _ComputeSpeedAtSample(X, Y, Z, T, selectedSampleIndex)

            distanceToInterceptPick = np.hypot(interceptPickX - defenderX, interceptPickY - defenderY)
            selectedInterceptPoint = (
                float(X[selectedSampleIndex]),
                float(Y[selectedSampleIndex]),
                float(Z[selectedSampleIndex]),
                float(T[selectedSampleIndex]),
                float(interceptPickSpeed),
            )
            nearestInterceptPoint = (
                nearestInterceptX,
                nearestInterceptY,
                nearestInterceptT,
                float(nearestInterceptSpeed),
            )

            if (
                interceptPickZ <= reachableHeightMax
                and interceptPickZ >= reachableHeightMin
                and distanceToInterceptPick <= defenderInterceptRadius
            ):
                return selectedInterceptPoint, nearestInterceptPoint

            return None, None

        except Exception:
            return None, None


    def SampleTrajectic(
        self,
        interceptCol: int,
        interceptRow: int,
        interceptZ: float,
        opponentCol: int,
        opponentRow: int,
        apexValues: List[float],
        downhillSpeedPreferred: Optional[float] = None,
        allowDownhillUnionFallback: bool = True,
        bounceFilter=None,
    ) -> Optional[Dict]:

        baseContextKey = (
            int(interceptCol),
            int(interceptRow),
            round(float(interceptZ), 1),
            int(opponentCol),
            int(opponentRow),
        )

        candidateContextEntries: List[Tuple[str, Dict[BounceKey, List[Trajectic]]]] = []
        visitedContextKeys: Set[ContextKey] = set()

        preferredSpeedNormalized = None
        if downhillSpeedPreferred is not None:
            preferredSpeedNormalized = self._NormalizeDownhillSpeed(downhillSpeedPreferred)

        def _try_add_context(contextKey: ContextKey, sourceTag: str) -> None:
            if contextKey in visitedContextKeys:
                return
            contextEntryValue = self.trajecticsByContext.get(contextKey)
            if contextEntryValue is None:
                return
            visitedContextKeys.add(contextKey)
            candidateContextEntries.append((sourceTag, contextEntryValue))

        if preferredSpeedNormalized is not None and preferredSpeedNormalized > 0.0:
            _try_add_context(baseContextKey + (preferredSpeedNormalized,), "preferred")

        _try_add_context(baseContextKey + (0.0,), "zero")

        if allowDownhillUnionFallback and not candidateContextEntries:
            availableDownhillSpeeds = sorted(self.availableDownhillSpeedsByBaseContext.get(baseContextKey, set()))
            for downhillSpeedValue in availableDownhillSpeeds:
                _try_add_context(baseContextKey + (float(downhillSpeedValue),), "union")

        if not candidateContextEntries:
            return None

        bounceStats: Dict[BounceKey, List[Trajectic]] = {}
        for _, contextEntry in candidateContextEntries:
            for bounceKey, trajectics in contextEntry.items():
                bounceStats.setdefault(bounceKey, []).extend(trajectics)

        # Collect all trajectics across bounce cells
        allTrajectics = []

        for bounceKey, trajectics in bounceStats.items():
            bounceCol, bounceRow = bounceKey

            if not self._isBounceInsideCourtRows(int(bounceRow)):
                continue

            if not self._isBounceAcrossNet(int(interceptRow), int(bounceRow)):
                continue

            # Optional external filter
            if bounceFilter is not None and not bounceFilter((bounceCol, bounceRow)):
                continue

            if not trajectics:
                continue

            for t in trajectics:
                # store bounce info with trajectic
                allTrajectics.append((bounceKey, t))

        if not allTrajectics:
            return None

        # # PURE RANDOM SELECTION (correct for current system state)
        # (bounceCol, bounceRow), trajectic = random.choice(allTrajectics)

        # (
        #     apexHeight,
        #     spinTopRpm,
        #     spinSideRpm,
        #     defensiveCol,
        #     defensiveRow,
        # ) = trajectic

           #     if len(trajectics) < 2:
    #         print("Less than 2 Trajectics")
    #         return None

        # selecting 2 trajectICs and recombining characteristics
        
        # To test random only 
        # if True:
        #     return None
        
        if len(allTrajectics) < 2:
            return None

        firstItem, secondItem = random.sample(allTrajectics, 2)

        (firstBounceCol, firstBounceRow), firstTrajectic = firstItem
        (secondBounceCol, secondBounceRow), secondTrajectic = secondItem

        bounceCol, bounceRow = random.choice([(firstBounceCol, firstBounceRow), (secondBounceCol, secondBounceRow)])

        # apexHeight options, random, random exponential for lower trajectories, less than for slightly lower trajectories
        # apexHeight = random.choice([firstTrajectic[0], secondTrajectic[0]])
        apexProbabilities = [0.133, 0.124, 0.114, 0.104, 0.095, 0.085, 0.076, 0.066, 0.057, 0.048, 0.038, 0.029, 0.019, 0.012 ]
        # apexProbabilities = [0.1930, 0.1665, 0.1419, 0.1192, 0.0985, 0.0798, 0.0631, 0.0483, 0.0355, 0.0246, 0.0158, 0.0089, 0.0039, 0.001]
        apexHeight = float(np.random.choice(self.CanonicalApexValues, p=apexProbabilities)) if self.CanonicalApexValues else 1.0
        # apexHeight = firstTrajectic[0] if firstTrajectic[0] < secondTrajectic[0] else secondTrajectic[0]
        
        spinTopRpm = random.choice([firstTrajectic[1], secondTrajectic[1]])
        spinSideRpm = random.choice([firstTrajectic[2], secondTrajectic[2]])

        defensiveCol, defensiveRow = random.choice(
            [
                (firstTrajectic[3], firstTrajectic[4]),
                (secondTrajectic[3], secondTrajectic[4]),
            ]
        )

        if self.debugLevel1:
            print("Selected Trajectic: " + str(interceptCol)+ ", " + str(interceptRow) + ", " + str(interceptZ) + ", " + str(opponentCol) + ", " + str(opponentRow) + ", " + str(bounceCol) + ", " + str(bounceRow) + ", " + str(apexHeight) + ", " + str(spinTopRpm) + ", " + str(spinSideRpm) + ", " + str(defensiveCol) + ", " + str(defensiveRow))

        return {
            "interceptCol": interceptCol,
            "interceptRow": interceptRow,
            "interceptZ": interceptZ,
            "opponentCol": opponentCol,
            "opponentRow": opponentRow,

            "bounceCol": bounceCol,
            "bounceRow": bounceRow,

            "apexHeight": apexHeight,
            "spinTopRpm": spinTopRpm,
            "spinSideRpm": spinSideRpm,
            "downhillSpeed": round(float(random.choice([firstTrajectic[5], secondTrajectic[5]])), 4),

            "defensiveCol": defensiveCol,
            "defensiveRow": defensiveRow,
        }

        
    def debugPrintBounceStats(self, bounceStats, bounceFilter=None):
        rows = []

        for bounceKey, trajectics in bounceStats.items():
            if bounceFilter and not bounceFilter(bounceKey):
                continue

            numTrajectics = len(trajectics)

            rows.append((bounceKey, numTrajectics))

        # Sort by descending density
        rows.sort(key=lambda x: x[1], reverse=True)

        print("\n--- Bounce Cell Distribution ---")
        totalCells = len(rows)
        nonEmpty = sum(1 for _, n in rows if n > 0)
        validCells = sum(1 for _, n in rows if n >= 2)

        print(f"Total cells (after filter): {totalCells}")
        print(f"Non-empty cells: {nonEmpty}")
        print(f"Cells with >=2 trajectics: {validCells}")
        print()

        for (col, row), n in rows:
            status = ""
            if n == 0:
                status = "EMPTY"
            elif n == 1:
                status = "WEAK (<2)"
            else:
                status = "OK"

            print(f"Cell ({col}, {row}) -> {n:4d} trajectics   [{status}]")

        print("--- End Bounce Distribution ---\n")



# Backward-compatible export used across runners and scripts.
SelectivePressure = TrajecticsSelector
