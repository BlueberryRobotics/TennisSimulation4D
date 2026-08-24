# API Migration Note

This repository completed a strict API naming migration to PascalCase method/function names and removed compatibility wrappers.

## Scope

The migration applies to the simulation pipeline and grid-indexing APIs in:

- `Trajectory4D/FenceGridIndexer.py`
- `Trajectory4D/ServeShotRunner.py`
- `Trajectory4D/RallyShotRunner.py`
- `Trajectory4D/PointRunner.py`
- `Trajectory4D/ShotValueTracker.py`
- `Trajectory4D/TrajecticsSelector.py`
- `RunSimulation4D.py`
- `RunSImulation4DMP.py`

## Entry-Point Migrations

- `pointRunner.playPoint(...)` -> `pointRunner.PlayPoint(...)`
- `tracker.processPoint(...)` -> `tracker.ProcessPoint(...)`
- `serveRunner.hitServe(...)` -> `serveRunner.HitServe(...)`
- `rallyRunner.hitRallyShot(...)` -> `rallyRunner.HitRallyShot(...)`

## Serve/Rally Helper Migrations

- `_computeServeInterceptionPoint(...)` -> `_ComputeServeInterceptionPoint(...)`
- `_resolveHitter(...)` -> `_ResolveHitter(...)`
- `_fail_with_intended(...)` -> `_FailWithIntended(...)`

## PointRunner Helper Migrations

- `_starting_positions(...)` -> `_StartingPositions(...)`
- `serverPoseForServe(...)` -> `ServerPoseForServe(...)`
- `receiverPoseForServe(...)` -> `ReceiverPoseForServe(...)`
- `applyDefensiveMove(...)` -> `ApplyDefensiveMove(...)`
- `winnerAfterServe(...)` -> `WinnerAfterServe(...)`
- `winnerAfterRally(...)` -> `WinnerAfterRally(...)`

## ShotValueTracker Helper Migrations

- `xyToCell(...)` -> `XyToCell(...)`
- `_roundX(...)` -> `_RoundX(...)`
- `_roundY(...)` -> `_RoundY(...)`
- `_roundZ(...)` -> `_RoundZ(...)`
- `_roundAngle(...)` -> `_RoundAngle(...)`
- `_convertShot(...)` -> `_ConvertShot(...)`

## FenceGridIndexer API Migrations

- `cellCenter(...)` -> `CellCenter(...)`
- `xyToCell(...)` -> `XyToCell(...)`
- `snapXyToCellCenter(...)` -> `SnapXyToCellCenter(...)`
- `snapXyToQuarterCenter(...)` -> `SnapXyToQuarterCenter(...)`
- `columnsInRange(...)` -> `ColumnsInRange(...)`
- `rowsInRange(...)` -> `RowsInRange(...)`
- `columnsInsideCourt(...)` -> `ColumnsInsideCourt(...)`
- `rowsInsideCourt(...)` -> `RowsInsideCourt(...)`
- `opponentHalfRows(...)` -> `OpponentHalfRows(...)`
- `singlesOpponentRegion(...)` -> `SinglesOpponentRegion(...)`
- `serviceBoxCells(...)` -> `ServiceBoxCells(...)`
- `quarterCenters(...)` -> `QuarterCenters(...)`
- `opponentRowsSortedNearNet(...)` -> `OpponentRowsSortedNearNet(...)`
- `cullFrontRows(...)` -> `CullFrontRows(...)`
- `_grid_settings(...)` -> `_GetGridSettings(...)`

## TrajecticsSelector Helper Migrations

- `_top_intercept_cells_by_win_pct(...)` -> `_GetTopInterceptCellsByWinPercentage(...)`
- `_speed_at_sample(...)` -> `_ComputeSpeedAtSample(...)`
- `_in_play_end_index(...)` -> `_FindInPlayEndIndex(...)`

## Removed Compatibility Layer

The old lowercase compatibility wrappers and aliases have been removed.

If you maintain external scripts that import these modules directly, update all calls to the new names listed above.

## Intentionally Unchanged

`CourtPlayerSettings.py` still contains methods such as `serverPoseForServe(...)` and `receiverPoseForServe(...)`. Those are part of that class API and were not changed by this migration.
