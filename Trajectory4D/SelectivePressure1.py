import duckdb
import random
import os
import shutil
import uuid
import numpy as np
from typing import Dict, Tuple, List, Optional, Set
from FenceGridIndexer import XyToCell


ContextKey = Tuple[int, int, float, int, int]
BounceKey = Tuple[int, int]
Trajectic = Tuple[float, int, int, int, int]


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

# In evolutionary computation, Selective Pressure, refers to selection factors 
# chosen to increase the success of individuals (trajectics in this case)
# and apply pressure to make successful traits increasingly prevail over
# less successful traits as the generations iterate (adaptive evolution).

class SelectivePressure:
    def __init__(
        self,
        parquetPath: str,
        minimumWinPercentage: float = 0.5,
        minimumCount: int = 3,
        court=None,
        netBoundaryRow: Optional[int] = None,
        isPreFilteredReference: bool = False,
        debug: bool = False
    ):
        self.parquetPath = parquetPath
        self.minimumWinPercentage = minimumWinPercentage
        self.minimumCount = minimumCount
        self.court = court
        self.debug = debug
        self.debugLevel1 = True
        self.isPreFilteredReference = isPreFilteredReference
        self.trajecticsByContext: Dict[ContextKey, Dict[BounceKey, List[Trajectic]]] = {}
        self.interceptCellStatsByOpponentAndIntercept: Dict[
            Tuple[int, int, int, int],
            Tuple[int, int],
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

    @staticmethod
    def BuildReferenceFile(sourceParquetPath: str, outputParquetPath: str, duckdbThreads: int = 4) -> str:
        outputDir = os.path.dirname(outputParquetPath)
        if outputDir:
            os.makedirs(outputDir, exist_ok=True)

        tempOutputPath = os.path.abspath(outputParquetPath + ".tmp")
        duckdbTempRoot = os.path.abspath(os.path.join(outputDir if outputDir else ".", "duckdb_tmp"))
        os.makedirs(duckdbTempRoot, exist_ok=True)
        duckdbTempDir = os.path.join(duckdbTempRoot, f"run_{os.getpid()}_{uuid.uuid4().hex[:8]}")
        os.makedirs(duckdbTempDir, exist_ok=True)

        query = SelectivePressure.GetReferenceSelectionQuery(sourceParquetPath)
        copySql = f"""
        COPY (
            {query}
        ) TO '{tempOutputPath}'
        (FORMAT PARQUET, COMPRESSION 'zstd');
        """

        connection = duckdb.connect()
        copyCompleted = False
        duckdbTempDirForPragma = duckdbTempDir.replace("\\", "/")

        def _is_benign_temp_cleanup_error(errorText: str) -> bool:
            return (
                "Failed to delete file" in errorText
                and "duckdb_temp_storage" in errorText
            )

        try:
            connection.execute(f"PRAGMA threads = {duckdbThreads}")
            connection.execute(f"PRAGMA temp_directory='{duckdbTempDirForPragma}'")
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
            finally:
                # Best-effort cleanup of per-run temp directory. Files can remain on Windows
                # when the OS still holds transient handles; that is safe for correctness.
                try:
                    shutil.rmtree(duckdbTempDir, ignore_errors=True)
                except Exception:
                    pass

        os.replace(tempOutputPath, outputParquetPath)

        return outputParquetPath

    @staticmethod
    def GetReferenceSelectionQuery(sourceParquetPath: str) -> str:
        return f"""
        -- Reference selection strategy:
        -- 1) Winner-cell quality prefers trusted rows (count >= 10),
        --    with capped influence per row to avoid legacy lock-in.
        --    If trusted rows are absent for a cell/context, fallback uses all rows
        --    so context coverage does not collapse.
        -- 2) Winner/loser bounce cells are sampled per context to preserve exploration.
        -- 3) Within each selected cell, keep a quality-filtered top-10 candidate set,
        --    then randomly pick 3 trajectics so reference composition changes by generation.
        WITH base AS (
            SELECT *
            FROM read_parquet('{sourceParquetPath}')
                        WHERE wins > 0
                            AND count >= 1
                            AND bounceRow BETWEEN 5 AND 22
        ),

        trusted_base AS (
            SELECT
                *,
                LEAST(count, 10) AS effectiveCount,
                (wins * 1.0 / count) AS rowWinPct
            FROM base
            WHERE count >= 10
        ),

        context_bounce_stats_trusted AS (
            SELECT
                interceptCol,
                interceptRow,
                interceptZ,
                opponentCol,
                opponentRow,
                bounceCol,
                bounceRow,
                SUM(rowWinPct * effectiveCount) AS totalWins,
                SUM(effectiveCount) AS totalCount,
                SUM(rowWinPct * effectiveCount) * 1.0 / SUM(effectiveCount) AS winPct
            FROM trusted_base
            GROUP BY
                interceptCol, interceptRow, interceptZ,
                opponentCol, opponentRow,
                bounceCol, bounceRow
        ),

        context_bounce_stats_all AS (
            SELECT
                interceptCol,
                interceptRow,
                interceptZ,
                opponentCol,
                opponentRow,
                bounceCol,
                bounceRow,
                SUM(wins) AS totalWins,
                SUM(count) AS totalCount,
                SUM(wins) * 1.0 / SUM(count) AS winPct
            FROM base
            GROUP BY
                interceptCol, interceptRow, interceptZ,
                opponentCol, opponentRow,
                bounceCol, bounceRow
        ),

        winner_cell_stats AS (
            SELECT
                allStats.interceptCol,
                allStats.interceptRow,
                allStats.interceptZ,
                allStats.opponentCol,
                allStats.opponentRow,
                allStats.bounceCol,
                allStats.bounceRow,
                -- Coverage safety: determine winner/loser eligibility from all data.
                -- Trusted rows are still preferred later via requireTrusted/isTrusted.
                allStats.totalWins AS totalWins,
                allStats.totalCount AS totalCount,
                allStats.winPct AS winPct,
                CASE WHEN trustedStats.totalCount IS NULL THEN 0 ELSE 1 END AS requireTrusted
            FROM context_bounce_stats_all allStats
            LEFT JOIN context_bounce_stats_trusted trustedStats
                ON allStats.interceptCol = trustedStats.interceptCol
                AND allStats.interceptRow = trustedStats.interceptRow
                AND allStats.interceptZ = trustedStats.interceptZ
                AND allStats.opponentCol = trustedStats.opponentCol
                AND allStats.opponentRow = trustedStats.opponentRow
                AND allStats.bounceCol = trustedStats.bounceCol
                AND allStats.bounceRow = trustedStats.bounceRow
        ),

        winner_tier_a AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        interceptCol, interceptRow, interceptZ,
                        opponentCol, opponentRow
                    ORDER BY requireTrusted DESC, RANDOM()
                ) AS tier_rank
            FROM winner_cell_stats
            WHERE totalCount >= 10 AND winPct >= 0.5
        ),

        winner_tier_b AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        interceptCol, interceptRow, interceptZ,
                        opponentCol, opponentRow
                    ORDER BY requireTrusted DESC, RANDOM()
                ) AS tier_rank
            FROM winner_cell_stats
            WHERE totalCount >= 3 AND totalCount < 10 AND winPct >= 0.5
        ),

        winner_tier_c AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        interceptCol, interceptRow, interceptZ,
                        opponentCol, opponentRow
                    ORDER BY requireTrusted DESC, RANDOM()
                ) AS tier_rank
            FROM winner_cell_stats
            WHERE totalCount >= 1 AND totalCount < 3 AND winPct >= 0.5
        ),

        winner_tier_counts AS (
            SELECT
                interceptCol,
                interceptRow,
                interceptZ,
                opponentCol,
                opponentRow,
                COALESCE(MAX(aCount), 0) AS tierACount,
                COALESCE(MAX(bCount), 0) AS tierBCount,
                COALESCE(MAX(cCount), 0) AS tierCCount
            FROM (
                SELECT
                    interceptCol,
                    interceptRow,
                    interceptZ,
                    opponentCol,
                    opponentRow,
                    COUNT(*) AS aCount,
                    0 AS bCount,
                    0 AS cCount
                FROM winner_tier_a
                GROUP BY interceptCol, interceptRow, interceptZ, opponentCol, opponentRow

                UNION ALL

                SELECT
                    interceptCol,
                    interceptRow,
                    interceptZ,
                    opponentCol,
                    opponentRow,
                    0 AS aCount,
                    COUNT(*) AS bCount,
                    0 AS cCount
                FROM winner_tier_b
                GROUP BY interceptCol, interceptRow, interceptZ, opponentCol, opponentRow

                UNION ALL

                SELECT
                    interceptCol,
                    interceptRow,
                    interceptZ,
                    opponentCol,
                    opponentRow,
                    0 AS aCount,
                    0 AS bCount,
                    COUNT(*) AS cCount
                FROM winner_tier_c
                GROUP BY interceptCol, interceptRow, interceptZ, opponentCol, opponentRow
            ) tierCounts
            GROUP BY interceptCol, interceptRow, interceptZ, opponentCol, opponentRow
        ),

        winner_candidates_a_only AS (
            SELECT
                a.interceptCol,
                a.interceptRow,
                a.interceptZ,
                a.opponentCol,
                a.opponentRow,
                a.bounceCol,
                a.bounceRow,
                a.totalWins,
                a.totalCount,
                a.winPct,
                a.requireTrusted
            FROM winner_tier_a a
            JOIN winner_tier_counts tierCounts
                ON a.interceptCol = tierCounts.interceptCol
                AND a.interceptRow = tierCounts.interceptRow
                AND a.interceptZ = tierCounts.interceptZ
                AND a.opponentCol = tierCounts.opponentCol
                AND a.opponentRow = tierCounts.opponentRow
            WHERE tierCounts.tierACount > 5
                AND a.tier_rank <= 10
        ),

        winner_candidates_a_sparse AS (
            SELECT
                a.interceptCol,
                a.interceptRow,
                a.interceptZ,
                a.opponentCol,
                a.opponentRow,
                a.bounceCol,
                a.bounceRow,
                a.totalWins,
                a.totalCount,
                a.winPct,
                a.requireTrusted
            FROM winner_tier_a a
            JOIN winner_tier_counts tierCounts
                ON a.interceptCol = tierCounts.interceptCol
                AND a.interceptRow = tierCounts.interceptRow
                AND a.interceptZ = tierCounts.interceptZ
                AND a.opponentCol = tierCounts.opponentCol
                AND a.opponentRow = tierCounts.opponentRow
            WHERE tierCounts.tierACount <= 5
                AND a.tier_rank <= 10
        ),

        winner_candidates_b_fill AS (
            SELECT
                b.interceptCol,
                b.interceptRow,
                b.interceptZ,
                b.opponentCol,
                b.opponentRow,
                b.bounceCol,
                b.bounceRow,
                b.totalWins,
                b.totalCount,
                b.winPct,
                b.requireTrusted
            FROM winner_tier_b b
            JOIN winner_tier_counts tierCounts
                ON b.interceptCol = tierCounts.interceptCol
                AND b.interceptRow = tierCounts.interceptRow
                AND b.interceptZ = tierCounts.interceptZ
                AND b.opponentCol = tierCounts.opponentCol
                AND b.opponentRow = tierCounts.opponentRow
            WHERE tierCounts.tierACount <= 5
                AND b.tier_rank <= GREATEST(0, 10 - tierCounts.tierACount)
        ),

        winner_candidates_c_fill AS (
            SELECT
                c.interceptCol,
                c.interceptRow,
                c.interceptZ,
                c.opponentCol,
                c.opponentRow,
                c.bounceCol,
                c.bounceRow,
                c.totalWins,
                c.totalCount,
                c.winPct,
                c.requireTrusted
            FROM winner_tier_c c
            JOIN winner_tier_counts tierCounts
                ON c.interceptCol = tierCounts.interceptCol
                AND c.interceptRow = tierCounts.interceptRow
                AND c.interceptZ = tierCounts.interceptZ
                AND c.opponentCol = tierCounts.opponentCol
                AND c.opponentRow = tierCounts.opponentRow
            WHERE tierCounts.tierACount <= 5
                AND c.tier_rank <= GREATEST(
                    0,
                    10
                    - tierCounts.tierACount
                    - LEAST(tierCounts.tierBCount, GREATEST(0, 10 - tierCounts.tierACount))
                )
        ),

        winner_pool AS (
            SELECT * FROM winner_candidates_a_only
            UNION ALL
            SELECT * FROM winner_candidates_a_sparse
            UNION ALL
            SELECT * FROM winner_candidates_b_fill
            UNION ALL
            SELECT * FROM winner_candidates_c_fill
        ),

        winning_cells AS (
            -- Winner side: final pick is up to 5 bounce cells per context.
            SELECT
                interceptCol,
                interceptRow,
                interceptZ,
                opponentCol,
                opponentRow,
                bounceCol,
                bounceRow,
                requireTrusted,
                'winner' AS selectionSource
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            interceptCol, interceptRow, interceptZ,
                            opponentCol, opponentRow
                        ORDER BY RANDOM()
                    ) AS rand_rank
                FROM winner_pool
            )
            WHERE rand_rank <= 5
        ),

        losing_cells AS (
            -- Loser side remains conservative: up to 1 bounce cell per context.
            SELECT
                interceptCol,
                interceptRow,
                interceptZ,
                opponentCol,
                opponentRow,
                bounceCol,
                bounceRow,
                0 AS requireTrusted,
                'loser' AS selectionSource
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            interceptCol, interceptRow, interceptZ,
                            opponentCol, opponentRow
                        ORDER BY RANDOM()
                    ) AS rand_rank
                FROM context_bounce_stats_all
                WHERE winPct <= 0.5
            )
            WHERE rand_rank <= 1
        ),

        selected_cells AS (
            SELECT * FROM winning_cells
            UNION ALL
            SELECT * FROM losing_cells
        ),

        joined AS (
            -- Pull trajectics from selected cells.
            -- Coverage safety: do not hard-gate winner cells to trusted rows only.
            -- Trusted rows are still prioritized later by ranking (isTrusted, effectiveCount).
            SELECT
                b.*,
                (b.wins * 1.0 / b.count) AS winPct,
                LEAST(b.count, 10) AS effectiveCount,
                CASE WHEN b.count >= 10 THEN 1 ELSE 0 END AS isTrusted
            FROM base b
            JOIN selected_cells c
                ON b.interceptCol = c.interceptCol
                AND b.interceptRow = c.interceptRow
                AND b.interceptZ = c.interceptZ
                AND b.opponentCol = c.opponentCol
                AND b.opponentRow = c.opponentRow
                AND b.bounceCol = c.bounceCol
                AND b.bounceRow = c.bounceRow
        ),

        apex_dedup AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        interceptCol, interceptRow, interceptZ,
                        opponentCol, opponentRow,
                        bounceCol, bounceRow,
                        apexHeight
                    ORDER BY winPct DESC, count DESC
                ) AS apex_rank
            FROM joined
        ),

        apex_unique AS (
            SELECT *
            FROM apex_dedup
            WHERE apex_rank = 1
        ),

        ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        interceptCol, interceptRow, interceptZ,
                        opponentCol, opponentRow,
                        bounceCol, bounceRow
                    ORDER BY isTrusted DESC, winPct DESC, effectiveCount DESC
                ) AS quality_rank
            FROM apex_unique
        ),

        top_candidates AS (
            SELECT *
            FROM ranked
            WHERE quality_rank <= 10
        ),

        randomized AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        interceptCol, interceptRow, interceptZ,
                        opponentCol, opponentRow,
                        bounceCol, bounceRow
                    ORDER BY RANDOM()
                ) AS trajectic_rank
            FROM top_candidates
        )

        SELECT
            interceptCol,
            interceptRow,
            ROUND(interceptZ, 1) AS interceptZ,
            opponentCol,
            opponentRow,
            bounceCol,
            bounceRow,
            ROUND(apexHeight, 1) AS apexHeight,
            spinTopRpm,
            spinSideRpm,
            defensiveCol,
            defensiveRow,
            wins,
            count
        FROM randomized
        WHERE trajectic_rank <= 3
        """

    def LoadTrajectics(self) -> None:
        connection = duckdb.connect()

        # Keep DuckDB thread count bounded to prevent memory pressure spikes.
        duckdbThreads = max(2, min(12, os.cpu_count() or 4))
        connection.execute(f"PRAGMA threads = {duckdbThreads}")

        if self.isPreFilteredReference:
            cursor = connection.execute(
                f"""
                SELECT
                    interceptCol,
                    interceptRow,
                    ROUND(interceptZ, 1) AS interceptZ,
                    opponentCol,
                    opponentRow,
                    bounceCol,
                    bounceRow,
                    ROUND(apexHeight, 1) AS apexHeight,
                    spinTopRpm,
                    spinSideRpm,
                    defensiveCol,
                    defensiveRow,
                    wins,
                    count
                FROM read_parquet('{self.parquetPath}')
                """
            )
        else:
            cursor = connection.execute(self.GetReferenceSelectionQuery(self.parquetPath))

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
                bounceCol,
                bounceRow,
                apexHeight,
                spinTopRpm,
                spinSideRpm,
                defensiveCol,
                defensiveRow,
                wins,
                count
            ) in rows:

                contextKey = (
                    int(interceptCol),
                    int(interceptRow),
                    round(float(interceptZ), 1),
                    int(opponentCol),
                    int(opponentRow),
                )

                contextEntry = self.trajecticsByContext.setdefault(contextKey, {})
                bounceKey = (int(bounceCol), int(bounceRow))
                trajecticList = contextEntry.setdefault(bounceKey, [])
                trajecticList.append(
                    (
                        round(float(apexHeight), 1),
                        int(spinTopRpm),
                        int(spinSideRpm),
                        int(defensiveCol),
                        int(defensiveRow),
                    )
                )

                interceptStatKey = (
                    int(opponentCol),
                    int(opponentRow),
                    int(interceptCol),
                    int(interceptRow),
                )
                priorWins, priorCount = self.interceptCellStatsByOpponentAndIntercept.get(interceptStatKey, (0, 0))
                self.interceptCellStatsByOpponentAndIntercept[interceptStatKey] = (
                    int(priorWins) + int(wins),
                    int(priorCount) + int(count),
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

        print(f"[TrajecticSelector] Loaded {len(self.trajecticsByContext):,} contexts")

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
            totalWinsValue, totalCountValue = self.interceptCellStatsByOpponentAndIntercept.get(interceptStatKey, (0, 0))
            if int(totalCountValue) <= 0:
                continue

            winPercentageValue = float(totalWinsValue) / float(totalCountValue)
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

    # A TrajectIC or trajectic is a Trajectory in Context
    # A trajectory that starts at a specific interceptPoint on the player's side
    # with a specific opponent position on the opponent's side of the court
    def SampleTrajectic(
        self,
        interceptCol: int,
        interceptRow: int,
        interceptZ: float,
        opponentCol: int,
        opponentRow: int,
        apexValues: List[float],
        bounceFilter=None,
    ) -> Optional[Dict]:

        contextKey = (
            int(interceptCol),
            int(interceptRow),
            round(float(interceptZ), 1),
            int(opponentCol),
            int(opponentRow),
        )

        contextEntry = self.trajecticsByContext.get(contextKey)

        if contextEntry is None:
            return None

        bounceStats = contextEntry

        # Collect all trajectics across bounce cells
        allTrajectics = []

        for bounceKey, trajectics in bounceStats.items():
            bounceCol, bounceRow = bounceKey

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
        
        if len(allTrajectics) < 2:
            return None

        firstItem, secondItem = random.sample(allTrajectics, 2)

        (firstBounceCol, firstBounceRow), firstTrajectic = firstItem
        (secondBounceCol, secondBounceRow), secondTrajectic = secondItem

        bounceCol, bounceRow = random.choice([(firstBounceCol, firstBounceRow), (secondBounceCol, secondBounceRow)])
        apexHeight = random.choice([firstTrajectic[0], secondTrajectic[0]])
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
