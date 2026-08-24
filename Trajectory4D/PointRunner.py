# Trajectory2D3Db/PointRunner.py
from typing import Tuple, Optional
from ExecutionProbability import ExecutionProbability
import random
import numpy as np
from FenceGridIndexer import CellCenter

class PointRunner:
    """
    Orchestrates a single point with movement-agnostic runners and ToF-based reachability.

    Flow:
      - North serves; South returns; then alternate rally shots.
      - After each shot, the hitter's defensive move is chosen by PlayerMovement.
      - Intercept selection for the defender uses your spec:
          1) Find the perpendicular/nearest XY sample on defender's half.
          2) Compute ToF_perp (time of that sample).
          3) Travel radius: R = playerSpeed * max(0, ToF_perp - reactionTime).
          4) Bucket = all samples on the defender's half that are within singles width,
             height band [reachZMin, reachZMax], and within R of defender XY.
          5) Uniformly sample one from the bucket. If none -> point ends.

    Invariants:
      - Every shot dict includes 'opponentPosition' (serve -> receiver pose).
      - 'maxShots' is an optional cap for debugging; None disables it.
    """

    def __init__(
        self,
        court,
        serveRunner,
        rallyRunner,
        movementModel,
        maxShots: Optional[int] = None,
        enableExecutionErrors: bool = True,
    ):
        self.court = court
        self.serveRunner = serveRunner
        self.rallyRunner = rallyRunner
        self.movementModel = movementModel
        self.maxShots = maxShots  # None => no cap
        self.enableExecutionErrors = bool(enableExecutionErrors)

        self.executionModel = ExecutionProbability(court) if self.enableExecutionErrors else None
        self.trajecticsSelector = getattr(self.rallyRunner, "trajecticSelector", None)
        if self.trajecticsSelector is None:
            self.trajecticsSelector = getattr(self.serveRunner, "trajecticSelector", None)
        self.trajecticsParquetPath = (
            getattr(self.trajecticsSelector, "parquetPath", None)
            if self.trajecticsSelector is not None
            else None
        )
        self.interceptSelectionDebug = bool(
            getattr(self.trajecticsSelector, "debug", False)
        )

    def _ComputeExecutionDetails(
        self,
        bounceX,
        bounceY,
        airTravelDistance,
        initialVelocity,
        intendedNetClearance,
        incomingInterceptVelocity,
        shotType,
        serveSide=None,
    ) -> dict:
        if self.executionModel is None:
            return {
                "executionProbability": 1.0,
                "shotLuckCost": 0.0,
            }

        return self.executionModel.computeExecutionDetails(
            bounceX=bounceX,
            bounceY=bounceY,
            airTravelDistance=airTravelDistance,
            initialVelocity=initialVelocity,
            intendedNetClearance=intendedNetClearance,
            incomingInterceptVelocity=incomingInterceptVelocity,
            shotType=shotType,
            serveSide=serveSide,
        )

    # ------------------------------------------------------------
    # Helper: starting positions per side
    # ------------------------------------------------------------
    def _StartingPositions(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        cx = self.court.centerLineX
        posNorth = (cx, self.court.serverBaselineY)
        posSouth = (cx, self.court.receiverBaselineY)
        return posNorth, posSouth
    
    def ServerPoseForServe(self, serveSide: str) -> Tuple[float, float]:
        return self.court.serverPoseForServe(serveSide)

    def ServeInterceptionPoseForServe(self, serveSide: str) -> Tuple[float, float]:
        return self.court.serveInterceptionPoseForServe(serveSide)

    # Receiver (Player South) serve-side starting pose:
    #  - 1 yard behind receiver baseline
    #  - 1 yard in from the appropriate singles sideline
    def ReceiverPoseForServe(self, serveSide: str) -> Tuple[float, float]:
        candidates = self.court.receiverPoseCandidatesForServe(serveSide, depthCells=3)
        if not candidates:
            return self.court.receiverPoseForServe(serveSide)
        return candidates[int(np.random.randint(len(candidates)))]

    def ApplyDefensiveMove(self, shotData, currentPos, playerSide):

        import numpy as np

        # -----------------------------------------
        # 1) Validate necessary shot data
        # -----------------------------------------
        if "transformed" not in shotData:
            return None

        transformed = shotData["transformed"]
        if "trajectory3D" not in transformed or "bounceIndex" not in transformed:
            return None

        traj = transformed["trajectory3D"]
        bounceIdx = transformed["bounceIndex"]

        if traj is None or bounceIdx is None:
            return None

        if bounceIdx <= 0 or bounceIdx >= len(traj):
            return None

        if "time" not in transformed:
            return None

        T = transformed["time"]
        timeToBounce = float(T[bounceIdx])

        # -----------------------------------------
        # 2) Compute max run distance
        # -----------------------------------------
        playerSpeed = float(getattr(self.movementModel, "playerSpeed", self.court.playerSpeed))
        reactionTime = float(getattr(self.movementModel, "reactionTime", 0.0))

        timeAvailable = max(0.0, timeToBounce - reactionTime)
        maxRunDistance = playerSpeed * timeAvailable

        defenderX, defenderY = float(currentPos[0]), float(currentPos[1])

        playerSideNormalized = str(playerSide or "").upper()
        if playerSideNormalized not in {"PLAYER_BLUE", "PLAYER_RED"}:
            playerSideNormalized = "PLAYER_BLUE" if defenderY <= self.court.netY else "PLAYER_RED"

        preferredDefensivePos = None
        try:
            preferredX = shotData.get("defensivePosX")
            preferredY = shotData.get("defensivePosY")
            if preferredX is not None and preferredY is not None:
                preferredDefensivePos = np.array([float(preferredX), float(preferredY)])
        except Exception:
            preferredDefensivePos = None

        playerPos = np.array([defenderX, defenderY])

        # -----------------------------------------
        # 3) Define court grid restrictions
        # -----------------------------------------
        gridCols = self.court.gridColumns
        gridRows = self.court.gridRows
        cellSize = self.court.granularity

        colStart = 3
        colEnd = 12

        if playerSideNormalized == "PLAYER_BLUE":
            validRows = [
                row
                for row in range(1, gridRows + 1)
                if float(self.court.GetRowCenterY(row)) < float(self.court.netY)
            ]
        else:
            validRows = [
                row
                for row in range(1, gridRows + 1)
                if float(self.court.GetRowCenterY(row)) > float(self.court.netY)
            ]

        if not validRows:
            return (defenderX, defenderY)

        # -----------------------------------------
        # 4) Define play axis
        # -----------------------------------------
        # A = bounce point (proxy for opponent-side interaction)
        bounceX = float(traj[bounceIdx][0])
        bounceY = float(traj[bounceIdx][1])
        A = np.array([bounceX, bounceY])

        # B = service/center intersection on defender side
        centerX = (gridCols * cellSize) / 2

        if playerSideNormalized == "PLAYER_BLUE":
            centerY = 0.5 * (float(self.court.serverBaselineY) + float(self.court.netY))
        else:
            centerY = 0.5 * (float(self.court.netY) + float(self.court.receiverBaselineY))

        B = np.array([centerX, centerY])

        AB = B - A
        AB_len_sq = np.dot(AB, AB)

        if AB_len_sq == 0:
            return None

        def distance_to_line(P):
            # Use explicit 2D determinant for NumPy 2.x compatibility.
            ap = P - A
            cross_2d = (AB[0] * ap[1]) - (AB[1] * ap[0])
            return abs(cross_2d) / np.sqrt(AB_len_sq)

        def projection_param(P):
            return np.dot(P - A, AB) / AB_len_sq

        # Direction of play (for forward constraint)
        playDir = A - playerPos
        playDir_norm = np.linalg.norm(playDir)
        if playDir_norm > 0:
            playDir /= playDir_norm

        # Band width
        MAX_DEVIATION = 3 * cellSize

        # -----------------------------------------
        # 5) Enumerate VALID defensive cells
        # -----------------------------------------
        validCells = []

        for col in range(colStart, colEnd + 1):
            for row in validRows:

                cellX, cellY = CellCenter(col, row, self.court)
                P = np.array([cellX, cellY])

                # Reachability
                dist = np.hypot(cellX - defenderX, cellY - defenderY)
                if dist > maxRunDistance:
                    continue

                # Axis band constraint
                if distance_to_line(P) > MAX_DEVIATION:
                    continue

                # Segment constraint
                t = projection_param(P)
                if t < 0 or t > 1:
                    continue

                # Forward-only movement (prevents backward nonsense)
                moveVec = P - playerPos
                move_norm = np.linalg.norm(moveVec)
                if move_norm > 0:
                    moveVec /= move_norm
                    if np.dot(moveVec, playDir) < 0:
                        continue

                validCells.append((cellX, cellY))

        # -----------------------------------------
        # 6) Uniform selection with fallback
        # -----------------------------------------
        if not validCells:
            # Fallback: relax ONLY the axis constraint, keep reachability + court bounds

            fallbackCells = []

            for col in range(colStart, colEnd + 1):
                for row in validRows:

                    cellX, cellY = CellCenter(col, row, self.court)

                    dist = np.hypot(cellX - defenderX, cellY - defenderY)
                    if dist <= maxRunDistance:
                        fallbackCells.append((cellX, cellY))

            if not fallbackCells:
                # absolute fallback: stay in place (safe, non-breaking)
                return (defenderX, defenderY)

            if preferredDefensivePos is not None:
                return min(
                    fallbackCells,
                    key=lambda candidate: float(
                        np.hypot(candidate[0] - preferredDefensivePos[0], candidate[1] - preferredDefensivePos[1])
                    ),
                )

            return min(
                fallbackCells,
                key=lambda candidate: distance_to_line(np.array([candidate[0], candidate[1]])),
            )

        # Normal case
        if preferredDefensivePos is not None:
            return min(
                validCells,
                key=lambda candidate: float(
                    np.hypot(candidate[0] - preferredDefensivePos[0], candidate[1] - preferredDefensivePos[1])
                ),
            )

        idx = np.random.randint(len(validCells))
        return validCells[idx]


    
    # ------------------------------------------------------------
    # Winner helpers
    # ------------------------------------------------------------
    @staticmethod
    def WinnerAfterServe(serveRes: dict):
        outc = serveRes.get("outcome")
        if outc == "NET":
            return ("OPP", "SERVE_NET")
        if outc == "OUT":
            return ("OPP", "SERVE_OUT")
        return ("UNKNOWN", "SERVE_UNKNOWN")

    @staticmethod
    def WinnerAfterRally(rallyRes: dict, hitterSide: str):
        outc = rallyRes.get("outcome")
        if outc == "NET":
            # Hitter hit the net -> defender wins
            return (("PLAYER" if hitterSide == "PLAYER_RED" else "OPP"), "RALLY_NET")
        if outc == "OUT":
            # Hitter missed long/wide -> defender wins
            return (("PLAYER" if hitterSide == "PLAYER_RED" else "OPP"), "RALLY_OUT")
        return ("UNKNOWN", "RALLY_UNKNOWN")

    # ------------------------------------------------------------
    # Main entry: play one point (DEUCE or AD serve)
    # ------------------------------------------------------------
    def PlayPoint(self, serveSide: str):
        shots = []
        cumulativeCurrentHitterLuckCost = {
            "PLAYER_BLUE": 0.0,
            "PLAYER_RED": 0.0,
        }
        cumulativeLuckCost = {
            "PLAYER_BLUE": 0.0,
            "PLAYER_RED": 0.0,
        }
        luckCostThreshold = float(
            getattr(self.court, "executionFailureThreshold", 1.0)
        )

        # player north is facing north or up, player south facing south or down
        positionPlayerBlue, positionPlayerRed = self._StartingPositions()
        # positionPlayerBlue = self.serverPoseForServe(serveSide)

        # Receiver starts serve-side-aware (1 yard in from sideline, 1 yard behind baseline)
        positionPlayerRed = self.ReceiverPoseForServe(serveSide)

        # ===== 1) Serve (North serves) =====
        serveResult = self.serveRunner.HitServe(serveSide, positionPlayerRed)
        # print("Serve Result InterceptPoint: " + str(serveResult["interceptPoint"]))

        # Invariant: every shot has 'opponentPosition' (serve -> receiver pose)

        # print("Serve Result Opponent Position " + str(serveResult["opponentPosition"]))
        serveResult["opponentPosition"] = positionPlayerRed  # receiver pose at serve time
        # A serve has no incoming shot velocity.
        serveResult["incomingInterceptVelocity"] = 0.0

        # Current-shot perspective (server's serve): incoming is naturally zero.
        serveResult["shotExecutionFactors"] = self._ComputeExecutionDetails(
            bounceX=serveResult.get("bounceX"),
            bounceY=serveResult.get("bounceY"),
            airTravelDistance=serveResult.get("airTravelDistance"),
            initialVelocity=serveResult.get("initialVelocity"),
            intendedNetClearance=serveResult.get("netClearance"),
            incomingInterceptVelocity=serveResult.get("incomingInterceptVelocity"),
            shotType=serveResult.get("type"),
            serveSide=serveResult.get("serveSide", serveSide),
        )
        serveShotLuckCost = float(serveResult["shotExecutionFactors"].get("shotLuckCost", 0.0))
        cumulativeCurrentHitterLuckCost["PLAYER_BLUE"] += serveShotLuckCost
        serveResult["currentHitterLuckCostIncrement"] = float(serveShotLuckCost)
        serveResult["currentHitterLuckCostCumulative"] = float(
            cumulativeCurrentHitterLuckCost["PLAYER_BLUE"]
        )

        shots.append(serveResult)
        assert "opponentPosition" in serveResult, "Missing opponentPosition on serve result"

        if serveResult.get("outcome") != "IN":
            winningPlayer, reason = self.WinnerAfterServe(serveResult)
            return {"shots": shots, "winningPlayer": winningPlayer, "reason": reason}

        # Track total shots in the point (serve counts as 1)
        total_shots = 1
        if self.maxShots is not None and total_shots >= self.maxShots:
            return {"shots": shots, "winningPlayer": "UNKNOWN", "reason": "SHOT_LIMIT_REACHED"}

        # Apply defensive move for the server (from North's own pose)
        newPos = self.ApplyDefensiveMove(serveResult, positionPlayerBlue, "PLAYER_BLUE")
        # print("ServePlayerBlue newPos: " + str(newPos))
        if newPos is not None:
            positionPlayerBlue = newPos
            serveResult["defensivePosX"] = float(newPos[0]) if newPos else self.court.centerLineX
            serveResult["defensivePosY"] = float(newPos[1]) if newPos else self.court.baselineY
        # print("Opponent Intercept: " + str(oppIntercept))

        # Determine opponent intercept for the return (South defender, ToF-based reach)
        # using the server's post-shot defensive position as opponent context.
        if self.trajecticsSelector is not None:
            oppIntercept, nearestIntercept = self.trajecticsSelector.SampleIntercept(
                transformed=serveResult["transformed"],
                defenderSide="PLAYER_RED",
                defenderPos=positionPlayerRed,
                opponentContextPos=positionPlayerBlue,
                court=self.court,
                movementModel=self.movementModel,
                topInterceptCellCount=1,
                debug=self.interceptSelectionDebug,
            )
        else:
            oppIntercept, nearestIntercept = None, None

        if oppIntercept is not None and len(oppIntercept) >= 5:
            serveResult["opponentInterceptVelocity"] = float(oppIntercept[4])
            serveResult["opponentInterceptPoint"] = (
                float(oppIntercept[0]),
                float(oppIntercept[1]),
                float(oppIntercept[2]),
                float(oppIntercept[3]),
            )

        if oppIntercept is None:
            # Could not return serve
            serveResult["winner"] = True
            return {"shots": shots, "winningPlayer": "PLAYER", "reason": "OPP_COULD_NOT_REACH_SERVE"}

        # The next hitter after a legal serve is Player South; include this in cumulative failure.
        serveIncomingInterceptVelocity = None
        if oppIntercept is not None and len(oppIntercept) >= 5:
            serveIncomingInterceptVelocity = float(oppIntercept[4])

        serveReturnDetails = self._ComputeExecutionDetails(
            bounceX=serveResult["bounceX"],
            bounceY=serveResult["bounceY"],
            airTravelDistance=serveResult["airTravelDistance"],
            initialVelocity=serveResult["initialVelocity"],
            intendedNetClearance=serveResult["netClearance"],
            incomingInterceptVelocity=serveIncomingInterceptVelocity,
            shotType=serveResult.get("type"),
            serveSide=serveResult.get("serveSide", serveSide),
        )

        serveReturnExecutionProbability = float(serveReturnDetails["executionProbability"])
        serveReturnErrorIncrement = max(0.0, 1.0 - serveReturnExecutionProbability)
        serveReturnLuckCost = float(serveReturnDetails.get("shotLuckCost", 0.0))

        cumulativeLuckCost["PLAYER_RED"] += serveReturnLuckCost
        serveResult["nextHitterExecutionProbability"] = serveReturnExecutionProbability
        serveResult["nextHitterExecutionErrorIncrement"] = float(serveReturnErrorIncrement)
        serveResult["nextHitterExecutionCumulative"] = float(
            cumulativeLuckCost["PLAYER_RED"]
        )
        serveResult["nextHitterLuckCostIncrement"] = float(serveReturnLuckCost)
        serveResult["nextHitterLuckCostCumulative"] = float(
            cumulativeLuckCost["PLAYER_RED"]
        )
        serveResult["nextHitterExecutionFactors"] = dict(serveReturnDetails)

        if self.enableExecutionErrors and cumulativeLuckCost["PLAYER_RED"] >= luckCostThreshold:
            return {"shots": shots, "winningPlayer": "PLAYER", "reason": "OPPONENT_ERROR"}

        # ===== 2) Rally loop =====
        hitterSide = "PLAYER_RED"  # opponent hits next
        while True:
            # This is the incoming speed for the shot that is about to be struck.
            currentIncomingInterceptVelocity = None
            if oppIntercept is not None and len(oppIntercept) >= 5:
                currentIncomingInterceptVelocity = float(oppIntercept[4])

            # Pass the defender's current pose to RallyShotRunner (3D generation context)
            opponentPosition = positionPlayerBlue if hitterSide == "PLAYER_RED" else positionPlayerRed

            rallyResult = self.rallyRunner.HitRallyShot(
                hitter=("OPP" if hitterSide == "PLAYER_RED" else "PLAYER"),
                interceptPoint=oppIntercept,
                opponentPosition=opponentPosition
            )
            # Persist incoming velocity for THIS shot (used by visualizations and diagnostics).
            rallyResult["incomingInterceptVelocity"] = currentIncomingInterceptVelocity

            # Current-shot perspective for this striker.
            rallyResult["shotExecutionFactors"] = self._ComputeExecutionDetails(
                bounceX=rallyResult.get("bounceX"),
                bounceY=rallyResult.get("bounceY"),
                airTravelDistance=rallyResult.get("airTravelDistance"),
                initialVelocity=rallyResult.get("initialVelocity"),
                intendedNetClearance=rallyResult.get("netClearance"),
                incomingInterceptVelocity=rallyResult.get("incomingInterceptVelocity"),
                shotType=rallyResult.get("type"),
            )
            currentShotLuckCost = float(
                rallyResult["shotExecutionFactors"].get("shotLuckCost", 0.0)
            )
            cumulativeCurrentHitterLuckCost[hitterSide] += currentShotLuckCost
            rallyResult["currentHitterLuckCostIncrement"] = float(currentShotLuckCost)
            rallyResult["currentHitterLuckCostCumulative"] = float(
                cumulativeCurrentHitterLuckCost[hitterSide]
            )

            shots.append(rallyResult)
            total_shots += 1
            assert "opponentPosition" in rallyResult, "Missing opponentPosition on rally result"

            # Optional per-point cap
            if self.maxShots is not None and total_shots >= self.maxShots:
                return {"shots": shots, "winningPlayer": "UNKNOWN", "reason": "SHOT_LIMIT_REACHED"}

            # Terminal by geometry at the hitter's shot
            if rallyResult.get("outcome") in ("NET", "OUT"):
                winningPlayer, reason = self.WinnerAfterRally(rallyResult, hitterSide)
                return {"shots": shots, "winningPlayer": winningPlayer, "reason": reason}
            
            # Apply defensive move for the *hitter* who just struck
            if hitterSide == "PLAYER_RED":
                newPos = self.ApplyDefensiveMove(rallyResult, positionPlayerRed, "PLAYER_RED")
                if newPos is not None:
                    positionPlayerRed = newPos
                    # print("RallyPlayerRed newPos: " + str(newPos))
                    rallyResult["defensivePosX"] = float(newPos[0]) if newPos else self.court.centerLineX
                    rallyResult["defensivePosY"] = float(newPos[1]) if newPos else self.court.receiverBaselineY
            else:
                newPos = self.ApplyDefensiveMove(rallyResult, positionPlayerBlue, "PLAYER_BLUE")
                if newPos is not None:
                    positionPlayerBlue = newPos
                    # print("RallyPlayerBlue newPos: " + str(newPos))
                    rallyResult["defensivePosX"] = float(newPos[0]) if newPos else self.court.centerLineX
                    rallyResult["defensivePosY"] = float(newPos[1]) if newPos else self.court.serverBaselineY

            # Next exchange: swap hitter, select defender intercept via ToF-based reach
            hitterSide = "PLAYER_BLUE" if hitterSide == "PLAYER_RED" else "PLAYER_RED"

            nextDefenderPos = positionPlayerBlue if hitterSide == "PLAYER_BLUE" else positionPlayerRed
            opponentContextPos = positionPlayerRed if hitterSide == "PLAYER_BLUE" else positionPlayerBlue

            if self.trajecticsSelector is not None:
                oppIntercept, nearestIntercept = self.trajecticsSelector.SampleIntercept(
                    transformed=rallyResult["transformed"],
                    defenderSide=hitterSide,     # the side that will now hit
                    defenderPos=nextDefenderPos,
                    opponentContextPos=opponentContextPos,
                    court=self.court,
                    movementModel=self.movementModel,
                    topInterceptCellCount=1,
                    debug=self.interceptSelectionDebug,
                )
            else:
                oppIntercept, nearestIntercept = None, None

            incomingInterceptVelocity = None
            if oppIntercept is not None and len(oppIntercept) >= 5:
                incomingInterceptVelocity = float(oppIntercept[4])
                rallyResult["opponentInterceptVelocity"] = incomingInterceptVelocity
                rallyResult["opponentInterceptPoint"] = (
                    float(oppIntercept[0]),
                    float(oppIntercept[1]),
                    float(oppIntercept[2]),
                    float(oppIntercept[3]),
                )

            executionDetails = self._ComputeExecutionDetails(
                bounceX=rallyResult["bounceX"],
                bounceY=rallyResult["bounceY"],
                airTravelDistance=rallyResult["airTravelDistance"],
                initialVelocity=rallyResult["initialVelocity"],
                intendedNetClearance=rallyResult["netClearance"],
                incomingInterceptVelocity=incomingInterceptVelocity,
                shotType=rallyResult.get("type"),
            )

            executionProbability = float(executionDetails["executionProbability"])
            executionErrorIncrement = max(0.0, 1.0 - executionProbability)
            shotLuckCost = float(executionDetails.get("shotLuckCost", 0.0))

            cumulativeLuckCost[hitterSide] += shotLuckCost
            rallyResult["nextHitterExecutionProbability"] = executionProbability
            rallyResult["nextHitterExecutionErrorIncrement"] = float(executionErrorIncrement)
            rallyResult["nextHitterExecutionCumulative"] = float(
                cumulativeLuckCost[hitterSide]
            )
            rallyResult["nextHitterLuckCostIncrement"] = float(shotLuckCost)
            rallyResult["nextHitterLuckCostCumulative"] = float(
                cumulativeLuckCost[hitterSide]
            )
            rallyResult["nextHitterExecutionFactors"] = dict(executionDetails)

            execution = cumulativeLuckCost[hitterSide] < luckCostThreshold

            # print("PositionPlayerBlue " + str(positionPlayerBlue) + " PositionPlayerRed: " + str(positionPlayerRed))

            if oppIntercept is None:
                # Defender cannot reach any intercept -> winner by forcing shot.
                winningPlayer = "OPP" if hitterSide == "PLAYER_BLUE" else "PLAYER"
                reason = "PLAYER_WINNER" if winningPlayer == "PLAYER" else "OPPONENT_WINNER"
                rallyResult["winner"] = True
                return {"shots": shots, "winningPlayer": winningPlayer, "reason": reason}

            if self.enableExecutionErrors and not execution:
                # Retained for compatibility when execution errors are enabled.
                winningPlayer = "OPP" if hitterSide == "PLAYER_BLUE" else "PLAYER"
                reason = (
                    "PLAYER_ERROR"
                    if winningPlayer == "PLAYER"
                    else "OPPONENT_ERROR"
                )
                return {"shots": shots, "winningPlayer": winningPlayer, "reason": reason}

