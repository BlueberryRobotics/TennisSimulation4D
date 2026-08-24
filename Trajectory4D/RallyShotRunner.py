# Trajectory4D/RallyShotRunner.py

import json
import numpy as np
from typing import List
from Trajectory4D.Trajectory4DDynamicGenerator import Trajectory4DDynamicGenerator
from Trajectory4D.SelectivePressure import SelectivePressure

from Trajectory4D.FenceGridIndexer import (
    CellCenter,
    XyToCell,
    SinglesOpponentRegion,
    OpponentRowsSortedNearNet,
    Granularity,
)

# print("[VER] RallyShotRunner4D: grid-targeted rally; "
#       "supports dynamic or canonical generators (bundle-safe)")


class RallyShotRunner:
    def __init__(
        self,
        court,
        generator4D: Trajectory4DDynamicGenerator,
        spinTopValues: List[int],
        spinSideValues: List[int],
        apexValues: List[float],
        interceptZValues: List[float],
        trajecticSelector,
        debug: bool = False,
        debugLevel1: bool = False
    ):
        self.court = court
        self.gen4D = generator4D
        self.spinTopList = list(spinTopValues)
        self.spinSideList = list(spinSideValues)
        self.apexValues = list(apexValues)
        self.interceptZValues = list(interceptZValues)
        self.trajecticSelector = trajecticSelector
        self.debug = debug
        self.debugLevel1 = debugLevel1
        self.g = float(getattr(court, "granularity", 1.3716))

    def _IsNorthSideRow(self, row: int) -> bool:
        if hasattr(self.court, "GetRowCenterY") and hasattr(self.court, "netY"):
            rowCenterY = float(self.court.GetRowCenterY(int(row)))
            return rowCenterY < float(self.court.netY)
        return int(row) <= int(getattr(self.court, "gridRows", 26) // 2)

    # -----------------------------------------------------------
    # Main rally execution
    # -----------------------------------------------------------
    def HitRallyShot(self, hitter, interceptPoint, opponentPosition):
        
        maxNetClearAbove = None

        # Normalize intercept point
        try:
            x0 = float(interceptPoint[0])
            y0 = float(interceptPoint[1])
            z0 = float(interceptPoint[2])
        except Exception:
            raise ValueError(f"interceptPoint must be (x,y,z[,t]); got {interceptPoint!r}")

        canonHitter = self._ResolveHitter(hitter, y0)

        if self.debug:
            print(
                f"[DEBUG] RallyShot: label={hitter}, "
                f"resolved={canonHitter}, "
                f"strike=({x0:.6f},{y0:.6f},{z0:.3f})"
            )

        ##################################

        # -----------------------------------------
        # Try trajectic-based selection first
        # -----------------------------------------
        trajectic = None
        interceptCol, interceptRow = XyToCell(x0, y0, self.court)
        interceptZ = self.SnapZToAllowedZValues(zValue=z0, allowedZValues=self.interceptZValues)
        opponentCol, opponentRow = XyToCell(opponentPosition[0], opponentPosition[1], self.court)
        if self.debug:
            print("Intercept: " + str(x0) + " " + str(y0) + " " + str(z0))
            print("Opponent: " + str(opponentPosition[0]) + " " + str(opponentPosition[1]))

        # tactic selects two by intercept and opponent pos with 50%+ win percentage
        # does random sampling between the two for all other attributes
        if self.trajecticSelector is not None:
            trajectic = self.trajecticSelector.SampleTrajectic(
                interceptCol=interceptCol,
                interceptRow=interceptRow,
                interceptZ=interceptZ,
                opponentCol=opponentCol,
                opponentRow=opponentRow,
                apexValues=self.apexValues
            )

        if trajectic is not None:
            candidateBounceRow = int(trajectic["bounceRow"])
            if self._IsNorthSideRow(interceptRow) == self._IsNorthSideRow(candidateBounceRow):
                trajectic = None
        
        defensivePosX = None
        defensivePosY = None
        downhillSpeedPreferred = None

        if trajectic is not None:
            if self.debug:
                print("Tactic")
            # Use tactic-selected parameters
            col = trajectic["bounceCol"]
            row = trajectic["bounceRow"]
            xb, yb = CellCenter(col, row, self.court)

            apexHeight = trajectic["apexHeight"]
            spinTopRpm = trajectic["spinTopRpm"]
            spinSideRpm = trajectic["spinSideRpm"]
            downhillSpeedPreferred = float(trajectic.get("downhillSpeed", 0.0))

            defensiveCol = trajectic["defensiveCol"]
            defensiveRow = trajectic["defensiveRow"]

            defensivePosX, defensivePosY = CellCenter(defensiveCol, defensiveRow, self.court)
            if self.debug:
                print("Defensive: " + str(defensivePosX) + " " + str(defensivePosY))
                print(
                    "[DEBUG] RallyShot: using trajecticSelector "
                    f"bounce=({col},{row}) "
                    f"apex={apexHeight:.2f} "
                    f"spin=({spinTopRpm},{spinSideRpm}) "
                    f"defensive=({defensiveCol},{defensiveRow})"
                )

        else:
            # -----------------------------------------
            # FALLBACK: random selection logic
            # -----------------------------------------

            # Select valid singles-region cells for the hitter
            allCells = SinglesOpponentRegion(self.court, forHitter=canonHitter)

            # Row-culling if behind baseline
            if canonHitter == "PLAYER_BLUE":
                baselineY = float(self.court.serverBaselineY)
                distanceBehindBaseline = float(baselineY - y0)
            else:
                baselineY = float(self.court.receiverBaselineY)
                distanceBehindBaseline = float(y0 - baselineY)

            frontRowsToCull = 2 if (distanceBehindBaseline >= 4.572) else (1 if distanceBehindBaseline > 0.0 else 0)

            if frontRowsToCull > 0:
                opponentRowsSorted = OpponentRowsSortedNearNet(
                    self.court, forHitter=canonHitter
                )
                blockedRows = set(opponentRowsSorted[:frontRowsToCull])
                cells = [(colValue, rowValue) for (colValue, rowValue) in allCells if rowValue not in blockedRows]
            else:
                cells = allCells

            if not cells:
                return self._FailWithIntended(
                    interceptPoint=interceptPoint,
                    bouncePoint=(xb, yb),
                    spinTopRpm=spinTopRpm,
                    spinSideRpm=spinSideRpm,
                    apexHeight=apexHeight,
                    reason=reason,
                    failureDetails=None,
                )

            col, row = cells[np.random.randint(len(cells))]
            xb, yb = CellCenter(col, row, self.court)

            # random choices from all possible values
            spinTopRpm = int(np.random.choice(self.spinTopList)) if self.spinTopList else 0
            spinSideRpm = int(np.random.choice(self.spinSideList)) if self.spinSideList else 0

            # weighting apexHeights to emphasize lower trajectories
            apexProbabilities = [0.133, 0.124, 0.114, 0.104, 0.095, 0.085, 0.076, 0.066, 0.057, 0.048, 0.038, 0.029, 0.019, 0.012 ]
            # apexProbabilities = [0.1930, 0.1665, 0.1419, 0.1192, 0.0985, 0.0798, 0.0631, 0.0483, 0.0355, 0.0246, 0.0158, 0.0089, 0.0039, 0.001]
            apexHeight = float(np.random.choice(self.apexValues, p=apexProbabilities)) if self.apexValues else 1.0

            if self.debugLevel1:
                print("Random Trajectic: " + str(interceptCol)+ ", " + str(interceptRow) + ", " + str(interceptZ) + ", " + str(opponentCol) + ", " + str(opponentRow) + ", " + str(col) + ", " + str(row) + ", " + str(apexHeight) + ", " + str(spinTopRpm) + ", " + str(spinSideRpm))

        #####################################
        # # Select valid singles-region cells for the hitter
        # cells_all = singlesOpponentRegion(self.court, for_hitter=canonHitter)

        # # Row-culling if behind baseline
        # if canonHitter == "PLAYER_BLUE":
        #     baselineY = float(self.court.serverBaselineY)
        #     dist_behind = float(baselineY - y0)
        # else:
        #     baselineY = float(self.court.receiverBaselineY)
        #     dist_behind = float(y0 - baselineY)

        # n_front = 2 if (dist_behind >= 4.572) else (1 if dist_behind > 0.0 else 0)

        # if n_front > 0:
        #     opp_rows_sorted = opponentRowsSortedNearNet(
        #         self.court, for_hitter=canonHitter
        #     )
        #     blocked_rows = set(opp_rows_sorted[:n_front])
        #     cells = [(c, r) for (c, r) in cells_all if r not in blocked_rows]
        # else:
        #     cells = cells_all

        # if not cells:
        #     return self._fail_with_intended(
        #         interceptPoint=interceptPoint,
        #         bouncePoint=(xb, yb),
        #         spinTopRpm=spinTopRpm,
        #         spinSideRpm=spinSideRpm,
        #         apexHeight=apexHeight,
        #         reason=reason,
        #         failureDetails=None,
        #     )

        # Select a single cell center as target
        # col, row = cells[np.random.randint(len(cells))]
        # xb, yb = cellCenter(col, row)

            # this may be used in dynamic
            # Near-net clearance cap
            dy_from_net = abs(float(yb) - float(self.court.netY))
            if dy_from_net <= 1.0 * Granularity:
                maxNetClearAbove = 2.0
            elif dy_from_net <= 2.0 * Granularity:
                maxNetClearAbove = 5.0
            else:
                maxNetClearAbove = None

        # # Select Spins and Apex
        # spinTopRpm = int(np.random.choice(self.spinTopList)) if self.spinTopList else 0
        # spinSideRpm = int(np.random.choice(self.spinSideList)) if self.spinSideList else 0
        # apexHeight = float(np.random.choice(self.apexValues)) if self.apexValues else 0.0
        
        # -----------------------------------------------------------
        # Call dynamic or canonical generator (bundle-safe)
        # -----------------------------------------------------------
        try:
            result = self.gen4D.generate_by_apex_ladder(
                interceptPoint=(x0, y0, z0),
                bouncePoint=(xb, yb),
                apexHeight=apexHeight,
                apexValues=self.apexValues,
                spinTopRpm=spinTopRpm,
                spinSideRpm=spinSideRpm,
                downhillSpeedPreferred=downhillSpeedPreferred,
                shotType="RALLY",
                maxInitialVelocityMps=44.704,
                maxNetClearAbove=maxNetClearAbove,
                landing_tol=0.10,
                net_eps=5e-3,
                maxItersPerApex=600,
            )
        except RuntimeError as e:
            reason = str(e)
            failureDetails = None
            if "::" in reason:
                tag, payload = reason.split("::", 1)
                if tag == "SOLVER_NO_CONVERGENCE":
                    try:
                        failureDetails = json.loads(payload)
                        reason = tag
                    except Exception:
                        failureDetails = {"raw": str(e)}
                        reason = tag

            return self._FailWithIntended(
                interceptPoint=interceptPoint,
                bouncePoint=(xb, yb),
                spinTopRpm=spinTopRpm,
                spinSideRpm=spinSideRpm,
                apexHeight=apexHeight,
                # defensivePosX=defensivePosX,
                # defensivePosY=defensivePosY,
                reason=reason,
                failureDetails=None,
            )

        # -----------------------------------------------------------
        # Backward-compatible unpacking
        # -----------------------------------------------------------
        def _UnpackGeneratedResult(generatedResult):
            if isinstance(generatedResult, dict) and "traj" in generatedResult:
                return (
                    generatedResult["traj"],
                    generatedResult.get("canonicalMeta", {}),
                    generatedResult.get("fencesMeta", {}),
                    generatedResult.get("simMeta", {}),
                )
            return generatedResult, {}, {}, {}

        traj, canonicalMeta, fencesMeta, simMeta = _UnpackGeneratedResult(result)

        downhillSpeed = float(canonicalMeta.get("downhillSpeed", 0.0))
        netClearance = fencesMeta.get("netClearance")

        snappedInterceptZ = self.SnapZToAllowedZValues(
            zValue=float(z0),
            allowedZValues=self.interceptZValues,
        )
        snappedApexHeight = min(
            self.apexValues,
            key=lambda candidateApex: abs(float(candidateApex) - float(traj["apexHeight"])),
        )
        interceptEqualsApex = float(snappedInterceptZ) == float(snappedApexHeight)
        isDownhillShot = float(downhillSpeed) > 0.0

        if isDownhillShot and interceptEqualsApex and (
            netClearance is None or float(netClearance) <= 0.0
        ):
            higherApexValues = sorted(
                [
                    float(candidateApex)
                    for candidateApex in self.apexValues
                    if float(candidateApex) > float(traj["apexHeight"]) + 1e-6
                ]
            )

            for retryApex in higherApexValues:
                try:
                    retryResult = self.gen4D.generate_by_apex_ladder(
                        interceptPoint=(x0, y0, z0),
                        bouncePoint=(xb, yb),
                        apexHeight=retryApex,
                        apexValues=self.apexValues,
                        spinTopRpm=spinTopRpm,
                        spinSideRpm=spinSideRpm,
                        downhillSpeedPreferred=downhillSpeedPreferred,
                        shotType="RALLY",
                        maxInitialVelocityMps=44.704,
                        maxNetClearAbove=maxNetClearAbove,
                        landing_tol=0.10,
                        net_eps=5e-3,
                        maxItersPerApex=600,
                    )
                except RuntimeError:
                    continue

                retryTraj, retryCanonicalMeta, retryFencesMeta, retrySimMeta = _UnpackGeneratedResult(retryResult)
                retryNetClearance = retryFencesMeta.get("netClearance")

                if retryNetClearance is None or float(retryNetClearance) <= 0.0:
                    continue

                traj = retryTraj
                canonicalMeta = retryCanonicalMeta
                fencesMeta = retryFencesMeta
                simMeta = retrySimMeta
                downhillSpeed = float(canonicalMeta.get("downhillSpeed", 0.0))
                netClearance = retryNetClearance
                break

        if self.debugLevel1:
            selectedApexHeight = float(traj.get("apexHeight", apexHeight))
            requestedApexHeight = float(apexHeight)
            print(
                "Final Trajectory: "
                + str(interceptCol) + ", " + str(interceptRow) + ", " + str(interceptZ)
                + ", " + str(opponentCol) + ", " + str(opponentRow)
                + ", " + str(col) + ", " + str(row)
                + ", requestedApex=" + str(requestedApexHeight)
                + ", selectedApex=" + str(selectedApexHeight)
                + ", " + str(spinTopRpm) + ", " + str(spinSideRpm)
            )

        # -----------------------------------------------------------
        # Accept trajectory
        # -----------------------------------------------------------
        b = int(traj["bounceIndex"])
        bx_sim = float(traj["fencesX"][b])
        by_sim = float(traj["fencesY"][b])
        # trueApex = float(np.max(traj["fencesZ"][: b + 1]))
        # usedApex = float(traj.get("usedApexHeight", trueApex))
        downhillSpeed = float(canonicalMeta.get("downhillSpeed", 0.0))
        distanceToBounce = float(np.hypot(xb - x0, yb - y0))

        intended = {
            "type": "ON_DEMAND_4D",
            "bounceX": float(xb),
            "bounceY": float(yb),
            "targetCell": {"col": int(col), "row": int(row)},
            "spinTopRpm": spinTopRpm,
            "spinSideRpm": spinSideRpm,
            "apexHeight": traj["apexHeight"], 
            "initialVelocity":traj["initialVelocity"],
            "airTravelDistance":traj["airTravelDistance"],
            "netClearance":fencesMeta["netClearance"],
            "downhillSpeed": downhillSpeed,
            "distanceToBounce": distanceToBounce,
        }

        snappedInterceptZ = self.SnapZToAllowedZValues(
            zValue=float(z0),
            allowedZValues=self.interceptZValues,
        )
        snappedApexHeight = min(
            self.apexValues,
            key=lambda candidateApex: abs(float(candidateApex) - float(traj["apexHeight"])),
        )
        interceptEqualsApex = float(snappedInterceptZ) == float(snappedApexHeight)
        isDownhillShot = float(downhillSpeed) > 0.0

        if isDownhillShot and interceptEqualsApex and (
            intended["netClearance"] is None or float(intended["netClearance"]) <= 0.0
        ):
            return self._FailWithIntended(
                interceptPoint=interceptPoint,
                bouncePoint=(xb, yb),
                spinTopRpm=spinTopRpm,
                spinSideRpm=spinSideRpm,
                apexHeight=apexHeight,
                reason="NO_NET_CLEARANCE",
                failureDetails={"netClearance": intended["netClearance"]},
            )

        # -----------------------------------------------------------
        # Populate simMeta fields
        # -----------------------------------------------------------
        simMeta["shotType"] = "RALLY"
        simMeta["hitter"] = canonHitter

        # -----------------------------------------------------------
        # Final shot dictionary
        # -----------------------------------------------------------
        shot = {
            "type": "RALLY",
            "shotType": "RALLY",
            "hitter": canonHitter,
            "intendedEntry": intended,
            "entryId": id(intended),
            "diag": {
                "label": hitter,
                "resolved": canonHitter,
                "panAngleDeg": float(traj.get("panAngleDeg", 0.0)),
                "solverIters": int(traj.get("solverIters", 0)),
                "landingErrorXY": float(traj.get("landingErrorXY", 0.0)),
                "netClearDeficit": float(traj.get("netClearDeficit", 0.0)),
                "netClearExcess": float(traj.get("netClearExcess", 0.0))
                if "netClearExcess" in traj
                else 0.0,
            },
            "interceptPoint": (x0, y0, z0),
            "panAngleDeg": float(traj.get("panAngleDeg", 0.0)),
            "bounceX": bx_sim,
            "bounceY": by_sim,
            "landingX": float(traj["landingX"]),
            "landingY": float(traj["landingY"]),
            "distanceToBounce": distanceToBounce,
            "apexHeight": traj["apexHeight"], 
            "initialVelocity":traj["initialVelocity"],
            "airTravelDistance":traj["airTravelDistance"],
            "netClearance":fencesMeta["netClearance"],
            "downhillSpeed": downhillSpeed,
            "spinTopRpm": spinTopRpm,
            "spinSideRpm": spinSideRpm,
            "transformed": traj,
            "opponentPosition": opponentPosition,
            "opponentSpeed": None,
            "defensivePosX": defensivePosX,
            "defensivePosY": defensivePosY,
            "clearsNet": True,
            "outcome": "IN",
            "winner": False,
            "canonicalMeta": canonicalMeta,
            "fencesMeta": fencesMeta,
            "simMeta": simMeta,
        }

        return shot

    # -----------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------
    def _ResolveHitter(self, hitter: str, y0: float) -> str:
        h = (hitter or "").upper()
        if h in {"PLAYER_BLUE", "BLUE"}:
            return "PLAYER_BLUE"
        if h in {"PLAYER_RED", "RED"}:
            return "PLAYER_RED"
        if h in {"OPP", "OPPONENT"}:
            return "PLAYER_RED"
        if h in {"PLAYER", "SELF", "ME"}:
            pass
        return "PLAYER_BLUE" if y0 <= self.court.netY else "PLAYER_RED"

    def _FailWithIntended(
        self,
        interceptPoint,     
        bouncePoint,
        spinTopRpm,
        spinSideRpm,
        apexHeight,
        reason=None,
        failureDetails=None,
    ):
        interceptX, interceptY, interceptZ = (
            interceptPoint[:3]
            if interceptPoint is not None and len(interceptPoint) >= 3
            else (0.0, 0.0, 0.0)
        )
        bounceX, bounceY = (bouncePoint if bouncePoint is not None else (interceptX, interceptY))
        spinTopRpm  = (spinTopRpm if spinTopRpm is not None else 0)
        spinSideRpm = (spinSideRpm if spinSideRpm is not None else 0)
        distanceToBounce = float(np.hypot(bounceX - interceptX, bounceY - interceptY))
        pan_deg = float(np.degrees(np.arctan2((bounceX - interceptX), (bounceY - interceptY))))

        intended = {
            "type": "ON_DEMAND_4D",
            "bounceX": float(bounceX),
            "bounceY": float(bounceY),
            "spinTopRpm": int(spinTopRpm),
            "spinSideRpm": int(spinSideRpm),
            "apexHeight": None if apexHeight is None else float(apexHeight),
            "distanceToBounce": distanceToBounce,
        }

        shot = {
            "type": "RALLY",
            "shotType": "RALLY",
            "hitter": None,
            "intendedEntry": intended,
            "entryId": id(intended),
            "interceptPoint": interceptPoint,
            "panAngleDeg": pan_deg,
            "bounceX": float(bounceX),
            "bounceY": float(bounceY),
            "landingX": float(bounceX),
            "landingY": float(bounceY),
            "distanceToBounce": distanceToBounce,
            "apexHeight": intended["apexHeight"],
            "initialVelocity": 0.0,
            "airTravelDistance": 0.0,
            "netClearance": 0.0,
            "downhillSpeed": 0.0,
            "spinTopRpm": int(spinTopRpm),
            "spinSideRpm": int(spinSideRpm),
            "transformed": None,
            "opponentPosition": None,
            "opponentSpeed": None,
            "defensivePosX": None,
            "defensivePosY": None,
            "clearsNet": False,
            "outcome": "NET",
            "winner": False,
            "reason": reason,
            "failureDetails": failureDetails,
            "canonicalMeta": {},
            "fencesMeta": {},
            "simMeta": {},
        }

        return shot
    
    def SnapZToAllowedZValues(self, zValue: float, allowedZValues):
        """Snap intercept height to nearest allowed canonical Z height."""
        return min(allowedZValues, key=lambda h: abs(h - zValue))