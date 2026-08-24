import math
from Trajectory4D.FenceGridIndexer import XyToCell

class ShotValueTracker:
    """
    Converts raw shot packages from PointRunner into
    shot-value training data rows.

    Responsibilities:
      - Mirror OPP shots into PLAYER coordinate frame
      - Discretize (round) positions
      - Attach final point outcome (WIN/LOSS)
    """

    def __init__(self, court, interceptZValues, apexValues):
        # self.shotKeyGenerator = shotKeyGenerator
        self.court = court
        self.interceptZValues = interceptZValues
        self.apexValues = apexValues

        # Precompute mirror centers
        self.centerX = (court.singlesLeftX + court.singlesRightX) / 2.0
        self.centerY = (court.serverBaselineY + court.opponentBaselineY) / 2.0

    # ------------------------------------------------------------
    # Discretization helpers
    # ------------------------------------------------------------
    def XyToCell(self, xValue: float, yValue: float, granularityValue: float) -> tuple[int, int]:
        """
        Convert world coordinates (x, y) in meters into 1-indexed
        grid column and row positions based on grid cell size GRANULARITY.
        """
        columnIndex = int(xValue // granularityValue) + 1
        rowIndex = int(yValue // granularityValue) + 1
        return columnIndex, rowIndex


    def _RoundX(self, xValue):
        return round(xValue * 2) / 2.0   # nearest 0.5 m

    def _RoundY(self, yValue):
        return round(yValue * 1) / 1.0   # nearest 1.0 m

    def _RoundZ(self, zValue):
        return round(zValue * 2) / 2.0   # nearest 0.5 m

    def _RoundAngle(self, angleValue):
        return round(angleValue * 2) / 2.0   # nearest 0.5°

    def ConvertFencesXYToCell(self, xValue: float, yValue: float, cellSizeMeters: float):
        """Convert world coordinates (x, y) to 1-indexed grid column/row."""
        columnIndex, rowIndex = XyToCell(xValue, yValue, self.court)

        #print("Before Clamping " + str(columnIndex) + ", " + str(rowIndex))

        fencesColIndex = max(1, min(self.court.gridColumns, columnIndex))
        fencesRowIndex = max(1, min(self.court.gridRows, rowIndex))
        #print("After Clamping " + str(fencesColIndex) + ", " + str(fencesRowIndex))

        return fencesColIndex, fencesRowIndex
    
    def ConvertBounceXYToCell(self, xValue: float, yValue: float, cellSizeMeters: float):
        # print("Pre Confined X,Y " + str(xValue) + ", " + str(yValue))
        """Convert world coordinates (x, y) to 1-indexed grid column/row."""
        columnIndex, rowIndex = XyToCell(xValue, yValue, self.court)
        # print("Pre Clamped Bounce Col, Row " + str(columnIndex) + ", " + str(rowIndex))

        # Bounce must be INSIDE the entire playable singles court
        # We mask the opposite side of the court elsewhere
        validBounceColMin = 5
        validBounceColMax = 10   # inclusive, gives 6 columns inside court
        validBounceRowMin = 3
        validBounceRowMax = 26  
        
        # clamp values due to some rare bounces on the lines ending up
        # in cells outside the court
        bounceColIndex = max(validBounceColMin, min(validBounceColMax, columnIndex))
        bounceRowIndex = max(validBounceRowMin, min(validBounceRowMax, rowIndex))

        # print("Post Clamped Bounce Col, Row " + str(columnIndex) + ", " + str(rowIndex))

        return bounceColIndex, bounceRowIndex

    def SnapZToAllowedHeights(self, zValue: float, allowedHeights):
        """Snap intercept height to nearest allowed canonical Z height."""
        return min(allowedHeights, key=lambda h: abs(h - zValue))
    
    def SnapApexToAllowedHeights(self, apexValue: float, allowedApexValues):
        if apexValue is not None and apexValue != 0.0:
            """Snap apex height to nearest allowed canonical apex height."""
            return min(allowedApexValues, key=lambda h: abs(h - apexValue))
        else:
            return 0.0

    # ------------------------------------------------------------
    # Convert a single shot into a training row
    # ------------------------------------------------------------
    def _ConvertShot(self, shot, finalWinner, pointShotCount):
        """
        Returns a dict representing one row of training data.
        """
        # print(f"[DEBUG] ConvertShot: shotType={shot.get('shotType')} "
        #     f"outcome={shot.get('outcome')} reason={shot.get('reason')} "
        #     f"intended={'Present' if shot.get('intendedEntry') is not None else 'None'}")

        assert shot.get("intendedEntry") is not None, (
            f"Tracker received a non-shot or failure: "
            f"shotType={shot.get('shotType')} outcome={shot.get('outcome')} reason={shot.get('reason')}"
        )

        # Extract fields
        interceptX, interceptY, interceptZ = shot["interceptPoint"][:3]

        # print("interceptZ " + str(interceptZ))
        if shot["defensivePosX"] is not None:
            defensivePosX = shot["defensivePosX"]
            defensivePosY = shot["defensivePosY"]
        else:
            defensivePosX = interceptX
            defensivePosY = interceptY

        # Opponent position may be None (e.g., serves)
        if shot["opponentPosition"] is not None:
            opponentX, opponentY = shot["opponentPosition"]
        else:
            opponentX, opponentY = (0.0, 0.0)  # neutral placeholder

        bounceX = shot["bounceX"]
        bounceY = shot["bounceY"]
        apexHeight = shot["apexHeight"]
        downhillSpeed = float(shot.get("downhillSpeed", 0.0) or 0.0)

        # Discretize
        ix_r = self._RoundX(interceptX)
        iy_r = self._RoundY(interceptY)
        iz_r = round(interceptZ,2)
        ox_r = self._RoundX(opponentX)
        oy_r = self._RoundY(opponentY)
        tx_r = self._RoundX(bounceX)
        ty_r = self._RoundY(bounceY)

        if apexHeight is None:
            # every one in a while about 1 in 5000 points, a shot gets through 
            # generate by apex ladder without a trajectory 
            # in that event, all of these other values will also be None
            # this entire point is discarded in RunSimulation4DMP in the pointIndex loop 
            # print("APEX VALUE NONE. Bounce Point: " + str(tx_r) + ", " + str(ty_r) + " TopSpinRpm: " + str(shot["spinTopRpm"]) )
            apexHeight = 0.0
            shot["apexHeight"] = 0.0
            shot["initialVelocity"] = 0.0
            shot["airTravelDistance"] = 0.0
            shot["netClearance"] = 0.0

        #  snap values, should be unnecessary
        # print("InterceptZ before snap: " + str(iz_r))
        snappedInterceptZ = self.SnapZToAllowedHeights(iz_r, self.interceptZValues)
        # print("Intercept Z after snap: " + str(snappedInterceptZ))
        snappedApex = self.SnapApexToAllowedHeights(apexHeight, self.apexValues)

        # Map to cells: Intercept, Bounce, Opponent, defensive position
        interceptCol, interceptRow = self.ConvertFencesXYToCell(ix_r, iy_r, self.court.granularity)
        bounceCol, bounceRow = self.ConvertBounceXYToCell(bounceX, bounceY, self.court.granularity)
        opponentCol, opponentRow = self.ConvertFencesXYToCell(opponentX, opponentY, self.court.granularity)
        defensiveCol, defensiveRow = self.ConvertFencesXYToCell(defensivePosX, defensivePosY, self.court.granularity)

        # Compute winValue per rule:
        #   - If PLAYER (PLAYER_BLUE) wins the point, all PLAYER_BLUE shots get 1, PLAYER_RED get 0
        #   - If OPPONENT (PLAYER_RED) wins, all PLAYER_RED shots get 1, PLAYER_BLUE shots get 0
        playerSideRowCount = int(self.court.gridRows // 2)

        if interceptRow <= playerSideRowCount:
            hitter = "PLAYER"
        elif interceptRow > playerSideRowCount:
            hitter = "OPPONENT"

        playerWon = False
        if finalWinner == "PLAYER":
            playerWon = True

        if hitter == "PLAYER":
            winLabel = 1 if playerWon else 0
        elif hitter == "OPPONENT":
            winLabel = 0 if playerWon else 1
        else:
            # Failsafe: unknown hitter → treat as loss
            winLabel = 0

        def ToFloat(value):
            try:
                return float(value.item())
            except Exception:
                return float(value)

        def ToInt(value):
            try:
                return int(value.item())
            except Exception:
                return int(value)

        return {
            "interceptCol": ToInt(interceptCol),
            "interceptRow": ToInt(interceptRow),
            "interceptZ":   ToFloat(snappedInterceptZ),

            "opponentCol": ToInt(opponentCol),
            "opponentRow": ToInt(opponentRow),

            "defensiveCol": ToInt(defensiveCol),
            "defensiveRow": ToInt(defensiveRow),

            "bounceCol": ToInt(bounceCol),
            "bounceRow": ToInt(bounceRow),

            "apexHeight": ToFloat(snappedApex),

            "spinTopRpm": ToInt(shot["spinTopRpm"]),
            "spinSideRpm": ToInt(shot["spinSideRpm"]),

            "initialVelocity": ToFloat(round(shot["initialVelocity"],2)),
            "airTravelDistance": ToFloat(round(shot["airTravelDistance"],2)),
            "netClearance": ToFloat(shot["netClearance"]),
            "downhillSpeed": ToFloat(round(downhillSpeed, 4)),

            "wins": ToInt(winLabel),
            "winner": shot["winner"],
            "pointShotCount": ToInt(pointShotCount),
            "winShotCount": ToInt(pointShotCount if ToInt(winLabel) == 1 else 0)
        }

    # ------------------------------------------------------------
    # Convert entire point into training rows
    # ------------------------------------------------------------
    def ProcessPoint(self, pointResult, pointShotCount):
        finalWinner = pointResult["winningPlayer"]
        rows = []

        for shot in pointResult["shots"]:
            # print("Shot in pointResult: " + str(shot["interceptPoint"]))
            row = self._ConvertShot(shot, finalWinner, pointShotCount)
            # print("Row: " + str(row))
            rows.append(row)

        return rows
