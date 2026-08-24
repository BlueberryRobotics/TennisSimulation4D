# Trajectory4D/FenceGridIndexer.py
from typing import Tuple, List
import math
from CourtPlayerSettings import Court

_defaultCourt = Court()

# Backward-compatible module defaults sourced from Court settings.
Granularity = float(getattr(_defaultCourt, "granularity", 1.3716))
GridColumns = int(getattr(_defaultCourt, "gridColumns", 14))
GridRows = int(getattr(_defaultCourt, "gridRows", 26))


def _GetGridSettings(court=None) -> Tuple[float, int, int]:
    if court is None:
        return Granularity, GridColumns, GridRows

    granularityValue = float(getattr(court, "granularity", Granularity))
    gridColumnsValue = int(getattr(court, "gridColumns", GridColumns))
    gridRowsValue = int(getattr(court, "gridRows", GridRows))
    return granularityValue, gridColumnsValue, gridRowsValue


def _GetRowCenterY(court, row: int, granularityValue: float) -> float:
    if court is not None and hasattr(court, "GetRowCenterY"):
        return float(court.GetRowCenterY(int(row)))
    return (float(row) - 0.5) * granularityValue


def _GetRowHeight(court, row: int, granularityValue: float) -> float:
    if court is not None and hasattr(court, "GetRowHeight"):
        return float(court.GetRowHeight(int(row)))
    return float(granularityValue)


def CellCenter(column: int, row: int, court=None) -> Tuple[float, float]:
    """
    Return (x,y) center (meters) of 1-indexed grid cell (col,row).
    col: 1..14 left->right; row: 1..26 near->far from server (north->south).
    """
    granularityValue, gridColumnsValue, gridRowsValue = _GetGridSettings(court)
    if not (1 <= column <= gridColumnsValue and 1 <= row <= gridRowsValue):
        raise ValueError(f"column,row out of range: {column},{row}")
    xValue = (column - 0.5) * granularityValue
    yValue = _GetRowCenterY(court, row, granularityValue)
    return (xValue, yValue)


def XyToCell(xValue: float, yValue: float, court=None) -> Tuple[int, int]:
    """
    Nearest (col,row) for fences (x,y) in meters; clamped to grid.
    """
    granularityValue, gridColumnsValue, gridRowsValue = _GetGridSettings(court)
    column = min(gridColumnsValue, max(1, int(math.floor(xValue / granularityValue) + 1)))

    if court is not None and hasattr(court, "YToRow"):
        row = int(court.YToRow(float(yValue)))
    else:
        row = int(math.floor(yValue / granularityValue) + 1)

    row = min(gridRowsValue, max(1, int(row)))
    return (column, row)


def SnapXyToCellCenter(xValue: float, yValue: float, court=None) -> Tuple[float, float]:
    """
    Snap fences (x,y) to the center of the nearest 1-yard grid cell.
    """
    column, row = XyToCell(xValue, yValue, court)
    return CellCenter(column, row, court)


def SnapXyToQuarterCenter(xValue: float, yValue: float, court=None) -> Tuple[float, float]:
    """
    Snap fences (x,y) to the center of the nearest quarter within its 1-yard cell.
    """
    granularityValue, _, _ = _GetGridSettings(court)
    column, row = XyToCell(xValue, yValue, court)
    centerXValue, centerYValue = CellCenter(column, row, court)
    quarterOffset = granularityValue / 4.0
    quarterOffsetY = _GetRowHeight(court, row, granularityValue) / 4.0
    quarterXValue = centerXValue + (quarterOffset if xValue >= centerXValue else -quarterOffset)
    quarterYValue = centerYValue + (quarterOffsetY if yValue >= centerYValue else -quarterOffsetY)
    return (quarterXValue, quarterYValue)

# ---------------- Court-aware region builders ----------------

def ColumnsInRange(court, xMin: float, xMax: float) -> List[int]:
    columnsInRangeList: List[int] = []
    epsilon = 1e-6
    _, gridColumnsValue, _ = _GetGridSettings(court)
    for column in range(1, gridColumnsValue + 1):
        centerXValue, _ = CellCenter(column, 1, court)
        if xMin - epsilon <= centerXValue <= xMax + epsilon:
            columnsInRangeList.append(column)
    return columnsInRangeList

def RowsInRange(court, yMin: float, yMax: float) -> List[int]:
    rowsInRangeList: List[int] = []
    epsilon = 1e-6
    _, _, gridRowsValue = _GetGridSettings(court)
    for row in range(1, gridRowsValue + 1):
        _, centerYValue = CellCenter(1, row, court)
        if yMin - epsilon <= centerYValue <= yMax + epsilon:
            rowsInRangeList.append(row)
    return rowsInRangeList

def ColumnsInsideCourt(court) -> List[int]:
    """Columns whose centers lie inside singles width."""
    return ColumnsInRange(court, court.singlesLeftX, court.singlesRightX)

def RowsInsideCourt(court) -> List[int]:
    """Rows whose centers lie between the two baselines (inclusive)."""
    minimumY = min(court.serverBaselineY, court.receiverBaselineY)
    maximumY = max(court.serverBaselineY, court.receiverBaselineY)
    return RowsInRange(court, minimumY, maximumY)

def OpponentHalfRows(court, forHitter: str) -> List[int]:
    """
    Rows (by center) on the opponent half of the court (strictly beyond the net).
    """
    rowList: List[int] = []
    netYValue = float(court.netY)
    _, _, gridRowsValue = _GetGridSettings(court)
    if (forHitter or "").upper() == "PLAYER_BLUE":
        # South half: row centers strictly beyond net plane.
        rowList = [
            row
            for row in range(1, gridRowsValue + 1)
            if CellCenter(1, row, court)[1] > netYValue
        ]
    else:
        # North half: row centers strictly beyond net plane.
        rowList = [
            row
            for row in range(1, gridRowsValue + 1)
            if CellCenter(1, row, court)[1] < netYValue
        ]
    return rowList

def SinglesOpponentRegion(court, forHitter: str) -> List[Tuple[int, int]]:
    """
    9 x 26 center region, restricted to the opponent half rows (by center).
    Returns list of (col,row) cells.
    """
    columnList = ColumnsInsideCourt(court)
    allRows = RowsInsideCourt(court)
    opponentHalfRowSet = set(OpponentHalfRows(court, forHitter))
    opponentRows = [row for row in allRows if row in opponentHalfRowSet]
    cellList: List[Tuple[int, int]] = [(column, row) for column in columnList for row in opponentRows]
    return cellList

def ServiceBoxCells(court, serveSide: str, forHitter: str) -> List[Tuple[int, int]]:
    """
    Grid cells (by center) inside the opponent service box for the given serve side.
    Uses court geometry (netY and service line).
    """
    if (forHitter or "").upper() == "PLAYER_BLUE":
        # Build geometric service box first, then cull one net-adjacent opponent row.
        minimumY = float(court.netY)
        maximumY = float(court.opponentServiceLineY)
        if (serveSide or "").upper() == "DEUCE":
            minimumX, maximumX = float(court.singlesLeftX), float(court.centerLineX)
        else:
            minimumX, maximumX = float(court.centerLineX), float(court.singlesRightX)
        rowList = RowsInRange(court, minimumY, maximumY)
        columnList = ColumnsInRange(court, minimumX, maximumX)
        blockedRows = set(OpponentRowsSortedNearNet(court, forHitter)[:1])
        rowList = [row for row in rowList if row not in blockedRows]
    else:
        # PlayerRed serves toward -Y (mirror box on north half)
        # Build geometric service box first, then cull one net-adjacent opponent row.
        maximumY = float(court.netY)
        minimumY = float(court.serviceLineY)
        if (serveSide or "").upper() == "DEUCE":
            # From South's POV, deuce is court's RIGHT half ⇒ x in [center, right singles]
            minimumX, maximumX = float(court.centerLineX), float(court.singlesRightX)
        else:
            minimumX, maximumX = float(court.singlesLeftX), float(court.centerLineX)
        rowList = RowsInRange(court, minimumY, maximumY)
        columnList = ColumnsInRange(court, minimumX, maximumX)
        blockedRows = set(OpponentRowsSortedNearNet(court, forHitter)[:1])
        rowList = [row for row in rowList if row not in blockedRows]

    return [(column, row) for column in columnList for row in rowList]

def QuarterCenters(column: int, row: int, court=None) -> dict:
    """
    Return centers (x,y) of 4 equal quarters inside a 1.5-yd cell: TL, TR, BL, BR.
    BL = lower Y; TL = higher Y; L/R by X.
    """
    granularityValue, _, _ = _GetGridSettings(court)
    centerXValue, centerYValue = CellCenter(column, row, court)
    quarterOffsetX = granularityValue / 4.0
    quarterOffsetY = _GetRowHeight(court, row, granularityValue) / 4.0
    return {
        "TL": (centerXValue - quarterOffsetX, centerYValue + quarterOffsetY),
        "TR": (centerXValue + quarterOffsetX, centerYValue + quarterOffsetY),
        "BL": (centerXValue - quarterOffsetX, centerYValue - quarterOffsetY),
        "BR": (centerXValue + quarterOffsetX, centerYValue - quarterOffsetY),
    }

# --- Row helpers for opponent half (by row centers) ---

def OpponentRowsSortedNearNet(court, forHitter: str) -> list[int]:
    """
    Return opponent-half row numbers sorted by ascending distance from the net plane.
    """
    netYValue = float(court.netY)
    # rows on opponent half
    opponentHalfRowList = OpponentHalfRows(court, forHitter)
    # sort by |y_center - netY|
    rowsSorted = sorted(opponentHalfRowList, key=lambda row: abs(CellCenter(1, row, court)[1] - netYValue))
    return rowsSorted

def CullFrontRows(court, forHitter: str, rows: list[int], frontRowCount: int) -> list[int]:
    """
    Remove the first n_front rows nearest to the net (on the opponent half).
    """
    if frontRowCount <= 0:
        return rows
    frontRowsSorted = OpponentRowsSortedNearNet(court, forHitter)
    blockedRows = set(frontRowsSorted[:frontRowCount])
    return [row for row in rows if row not in blockedRows]