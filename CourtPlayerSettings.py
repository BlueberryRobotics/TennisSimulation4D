from typing import Tuple, Optional, List
import random
import bisect

# ---------------------------------------------------------
# Court geometry (unchanged)
# ---------------------------------------------------------
class Court:
    class _BoxPolygon:
        def __init__(self, corners):
            self._corners = corners
        def edges(self):
            pts = self._corners
            return list(zip(pts, pts[1:] + pts[:1]))

    # granularity is 1.5 yards
    def __init__(self, granularity=1.3716, geometryMode: str = "uniform"):

        # -------------------------------------------------
        # Player parameters - determines winners and errors
        # -------------------------------------------------

        # reaction time and speed determine what shots can be reached
        self.playerReactionTime = 0.0   # seconds
        self.playerSpeed = 4.5          # m/s 
        self.playerReachZMin = 0.3
        self.playerReachZMax = 3.3

        # optimal precision is how accurate the player is when hitting the ball
        # measured radially from target under baseline conditions
        # 3 degrees means they spray the ball between +- 3 degrees around target
        self.optimalPrecisionDegrees= 3.0

        # error gains determine how fast the player's imprecision
        # grows as incoming and outgoing velocity increase
        self.incomingVelocityErrorGain=0.5
        self.outgoingVelocityErrorGain=0.5

        # Luck weights scale per-shot luck cost contributions.
        # Incoming/outgoing velocity are attribution metrics only.
        # shotLuckCost = weightedNetClearanceError
        #              + weightedInBoundsError
        self.proximityToNetLuckWeight=1.0
        self.proximityToBoundaryLuckWeight=1.0
        # Linear incoming-speed scaling uses 0..maxIncomingVelocity.
        # 130 mph = 58.1152 m/s.
        self.maxIncomingVelocity=58.1152

        # error accumulates for both players during a point
        # and if one or the other does not win the point outright 
        # then it will ultimately hit the threshold for one of them
        # and the player that hits the threshold first is given an error 
        self.executionFailureThreshold=1.0

        # -------------------------------------------------
        # Court geometry parameters 
        # the reference geometry has 0,0 at the intersection of the left and bottom fences
        # the court is then offset inside that area
        # -------------------------------------------------

        # granularity is 1.5 yards in meters
        # court is divided into 1.5 yard cells or squares
        self.granularity = granularity
        self.geometryMode = (geometryMode or "uniform").strip().lower()

        if self.geometryMode not in ("uniform", "short_rows_13_14"):
            raise ValueError(
                "geometryMode must be 'uniform' or 'short_rows_13_14'"
            )
        
        # fence to fence is 63 feet wide or 14 - 1.5 yard columns
        self.widthFence = 19.2024
        # fence to fence is 117 feet long in uniform mode.
        # fence to fence is 114 feet long In short_rows_13_14 mode
        # since rows 13 and 14 are only3 ft each.
        # self.lengthFence = 35.6616 # uniform mode
        self.lengthFence = 34.7472 # short rows 13_14 mode

        self.xMin = 0.0
        self.xMax = self.widthFence
        self.yMin = 0.0
        self.yMax = self.lengthFence

        self.gridColumns = int(self.widthFence/self.granularity)

        self.rowHeights = self._BuildRowHeights()
        self.gridRows = len(self.rowHeights)
        self.lengthFence = float(sum(self.rowHeights))
        self.yMax = self.lengthFence
        self.rowBoundaryY = self._BuildRowBoundaries()
        self.rowCenterY = [
            0.5 * (self.rowBoundaryY[rowIndex - 1] + self.rowBoundaryY[rowIndex])
            for rowIndex in range(1, self.gridRows + 1)
        ]

        # print("Grid Columns " + str(self.gridColumns))
        # print("Grid Rows " + str(self.gridRows))

        # each sideline is 18 feet or 4 - 1.5 yard squares in from the fence
        self.singlesLeftX = 5.4864
        self.singlesRightX = self.singlesLeftX + 8.23

        if self.geometryMode == "short_rows_13_14":
            # Baselines and service lines align to requested row boundaries.
            # North baseline: boundary between rows 4 and 5.
            self.serverBaselineY = self.RowBoundaryY(4)
            # South baseline: boundary between rows 22 and 23.
            self.receiverBaselineY = self.RowBoundaryY(22)
            self.opponentBaselineY = self.receiverBaselineY

            # Service lines on boundaries 8/9 and 18/19.
            self.serviceLineY = self.RowBoundaryY(8)
            self.opponentServiceLineY = self.RowBoundaryY(18)

            # Net on boundary 13/14.
            self.netY = self.RowBoundaryY(13)
            self.centerLineY = 0.5 * (self.serverBaselineY + self.receiverBaselineY)
        else:
            self.serverBaselineY = 5.9436
            self.receiverBaselineY = 29.718
            self.opponentBaselineY = self.receiverBaselineY

            # Service line is 18 ft from each baseline (and 21 ft from net).
            serviceLineOffsetFromBaseline = 18.0 * 0.3048
            self.opponentServiceLineY = self.receiverBaselineY - serviceLineOffsetFromBaseline
            self.serviceLineY = self.serverBaselineY + serviceLineOffsetFromBaseline
            self.centerLineY = self.serverBaselineY + 11.88

            self.netY = self.lengthFence / 2.0
        # setting the net height at 1m or 39 inches, a compromise between sides and center
        self.netHeight = 1  

        self.centerLineX = (self.singlesLeftX) + (8.23 / 2.0)

        # self.serverStartPos = (self.centerLineX, self.serverBaselineY)

        self.playerSpeed = 4.5

    def _BuildRowHeights(self) -> List[float]:
        if self.geometryMode == "short_rows_13_14":
            rowHeights = [float(self.granularity) for _ in range(26)]
            # Rows are 1-indexed; rows 13 and 14 become 3 feet.
            shortRowHeight = 3.0 * 0.3048
            rowHeights[12] = shortRowHeight
            rowHeights[13] = shortRowHeight
            return rowHeights

        # Uniform legacy geometry.
        return [float(self.granularity) for _ in range(26)]

    def _BuildRowBoundaries(self) -> List[float]:
        boundaries = [0.0]
        for rowHeight in self.rowHeights:
            boundaries.append(boundaries[-1] + float(rowHeight))
        return boundaries

    def RowBoundaryY(self, boundaryIndex: int) -> float:
        if boundaryIndex < 0 or boundaryIndex > self.gridRows:
            raise ValueError(f"boundaryIndex out of range: {boundaryIndex}")
        return float(self.rowBoundaryY[boundaryIndex])

    def GetRowHeight(self, row: int) -> float:
        if row < 1 or row > self.gridRows:
            raise ValueError(f"row out of range: {row}")
        return float(self.rowHeights[row - 1])

    def GetRowCenterY(self, row: int) -> float:
        if row < 1 or row > self.gridRows:
            raise ValueError(f"row out of range: {row}")
        return float(self.rowCenterY[row - 1])

    def YToRow(self, yValue: float) -> int:
        # On exact boundaries use the row above the boundary (bisect_right).
        if yValue <= self.yMin:
            return 1
        if yValue >= self.yMax:
            return self.gridRows

        boundaryIndex = bisect.bisect_right(self.rowBoundaryY, float(yValue)) - 1
        boundaryIndex = max(0, min(self.gridRows - 1, int(boundaryIndex)))
        return boundaryIndex + 1

    # def serverPoseForServe(self, serveSide: str) -> Tuple[float, float]:
    #     g = float(self.granularity)
    #     if (serveSide or "").upper() == "DEUCE":
    #         x = self.centerLineX + (g / 2)
    #     else:  # AD
    #         x = self.centerLineX - (g / 2)
    #     y = self.receiverBaselineY - (g / 2)

    #     return (x, y)

    def serveInterceptionPoseForServe(self, serveSide: str) -> Tuple[float, float]:
        """Canonical serve interception XY used by serve context keys."""
        g = float(self.granularity)
        if (serveSide or "").upper() == "DEUCE":
            # DEUCE context alternates between cells (8,5) and (9,5).
            x = self.centerLineX + (0.5 * g) if random.random() < 0.5 else self.centerLineX + (1.5 * g)
        else:  # AD
            # AD context alternates between cells (6,5) and (7,5).
            x = self.centerLineX - (0.5 * g) if random.random() < 0.5 else self.centerLineX - (1.5 * g)
        # Keep serve interception cell-oriented: sample from row 4 or row 5 centers,
        # never on the baseline boundary between those rows.
        y = self.GetRowCenterY(4) if random.random() < 0.5 else self.GetRowCenterY(5)
        return (x, y)

    def receiverPoseCandidatesForServe(self, serveSide: str, depthCells: int = 3) -> List[Tuple[float, float]]:
        """Canonical receiver pose candidates for serve-side context sampling."""
        g = float(self.granularity)
        if (serveSide or "").upper() == "DEUCE":
            x = self.singlesLeftX + (g / 2)
        else:  # AD
            x = self.singlesRightX - (g / 2)

        candidatePoses: List[Tuple[float, float]] = []
        # Include one half-cell inside baseline first (row 22), then move behind baseline.
        startOffset = -0.5
        for offset in range(max(1, int(depthCells))):
            y = self.receiverBaselineY + ((startOffset + offset) * g)
            candidatePoses.append((float(x), float(y)))
        return candidatePoses

    def receiverPoseForServe(self, serveSide: str) -> Tuple[float, float]:
        candidates = self.receiverPoseCandidatesForServe(serveSide, depthCells=1)
        return candidates[0]

    # Receiver (Player South) serve-side starting pose:
    #  - 1 yard behind receiver baseline
    #  - 1 yard in from the appropriate singles sideline
    # def receiverPoseForServe(self, serveSide: str) -> Tuple[float, float]:
    #     g = float(self.granularity)
    #     if (serveSide or "").upper() == "DEUCE":
    #         x = self.singlesLeftX + (g / 2)
    #     else:  # AD
    #         x = self.singlesRightX - (g / 2)
    #     # "Behind" the receiver baseline = toward the back fence (in +Y for Player South)
    #     y = self.receiverBaselineY + (g / 2)
    #     return (x, y)


    # def getServeBoxPolygon(self, serveSide):
    #     y1 = self.netY
    #     y2 = self.opponentServiceLineY
    #     if serveSide.upper() == "DEUCE":
    #         x1, x2 = self.singlesLeftX, self.centerLineX
    #     else:
    #         x1, x2 = self.centerLineX, self.singlesRightX
    #     return Court._BoxPolygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    def isInServeBox(self, x, y, serveSide):
        eps = 1e-6
        if not (self.netY - eps <= y <= self.opponentServiceLineY + eps):
            return False
        if serveSide.upper() == "DEUCE":
            return self.singlesLeftX - eps <= x <= self.centerLineX + eps
        return self.centerLineX - eps <= x <= self.singlesRightX + eps

    def isInBounds(self, x, y):
        return (
            self.singlesLeftX <= x <= self.singlesRightX
            and self.serverBaselineY <= y <= self.receiverBaselineY
        )
    