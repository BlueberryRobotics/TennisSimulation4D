# Trajectory4D/ServeShotRunner.py

import json
import numpy as np
from typing import List

from Trajectory4D.Trajectory4DDynamicGenerator import Trajectory4DDynamicGenerator
from Trajectory4D.FenceGridIndexer import (
    CellCenter,
    ServiceBoxCells,
    XyToCell,
    OpponentRowsSortedNearNet,
    Granularity,
)

# print("[VER] ServeShotRunner4D: grid-targeted serve (quarter), "
#       "supports dynamic or canonical generators (bundle-safe)")


class ServeShotRunner:
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

    # -----------------------------------------------------------
    # Main serve execution
    # -----------------------------------------------------------
    def HitServe(self, serveSide, opponentPosition):
        serveSideNormalized = (serveSide or "").upper()
        if self.debug:
            print(serveSide)

        #print("Apex Heights " + str(self.apexValues))
        maxNetClearAbove = None
        quarterKey = None

        # Strike / Launch point
        serveInterceptPoint = self._ComputeServeInterceptionPoint(serveSide)
        x0, y0, z0 = serveInterceptPoint
        allowedServeBounceCells = self._GetAllowedServeBounceCells(serveSideNormalized, y0)
        bounceFilter = lambda bounceKey: bounceKey in allowedServeBounceCells

        # -----------------------------------------
        # Try trajectic-based selection first
        # -----------------------------------------
        trajectic = None
        interceptCol, interceptRow = XyToCell(x0, y0, self.court)
        interceptZ = self.SnapZToAllowedZValues(zValue=z0, allowedZValues=self.interceptZValues)
        opponentCol, opponentRow = XyToCell(opponentPosition[0], opponentPosition[1], self.court)

        if self.debug:
            print(serveInterceptPoint)

        # trajectic selects two by intercept and opponent pos with 50%+ win percentage
        # does random sampling between the two for all other attributes
        if self.trajecticSelector is not None:
            trajectic = self.trajecticSelector.SampleTrajectic(
                interceptCol=interceptCol,
                interceptRow=interceptRow,
                interceptZ=interceptZ,
                opponentCol=opponentCol,
                opponentRow=opponentRow,
                apexValues=self.apexValues,
                bounceFilter=bounceFilter
            )
        
        defensivePosX = None
        defensivePosY = None
        downhillSpeedPreferred = None

        if trajectic is not None:
            if self.debug:
                print("Trajectic")
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
            if self.debugLevel1:
                print("Serve Trajectic is Random")

            cells = sorted(allowedServeBounceCells)

            if not cells:
                return self._FailWithIntended(
                    interceptPoint=serveInterceptPoint,
                    bouncePoint=(xb, yb),
                    spinTopRpm=spinTopRpm,
                    spinSideRpm=spinSideRpm,
                    apexHeight=apexHeight,
                    serveSide=serveSideNormalized,
                    reason=reason,
                    failureDetails=None,
                )

            # Select a target cell center
            col, row = cells[np.random.randint(len(cells))]
            xb, yb = CellCenter(col, row, self.court)

            # Near-net cap
            dy_from_net = abs(float(yb) - float(self.court.netY))
            if dy_from_net <= 1.0 * Granularity:
                maxNetClearAbove = 2.0
            elif dy_from_net <= 2.0 * Granularity:
                maxNetClearAbove = 5.0
            else:
                maxNetClearAbove = None

            # Spins and Apex
            spinTopRpm = int(np.random.choice(self.spinTopList)) if self.spinTopList else 0
            spinSideRpm = int(np.random.choice(self.spinSideList)) if self.spinSideList else 0
            serveApexValues = self.apexValues
            serveApexValues = [candidate for candidate in serveApexValues if candidate > 2.5]

            # apexHeight = float(np.random.choice(serveApexValues)) if serveApexValues else 0.0
            # apexProbabilities = [0.133, 0.124, 0.114, 0.104, 0.095, 0.085, 0.076, 0.066, 0.057, 0.048, 0.038, 0.029, 0.019, 0.012 ]
            apexProbabilities = [0.1930, 0.1665, 0.1419, 0.1192, 0.0985, 0.0798, 0.0631, 0.0483, 0.0355, 0.0246, 0.0158, 0.0089, 0.0039, 0.001]
            apexHeight = float(np.random.choice(self.apexValues, p=apexProbabilities)) if self.apexValues else 1.0

        # print("Server Intercept Z: " + str(z0))

        # print("SERVE APEX HEIGHT: " + str(apexHeight))
        # -----------------------------------------------------------
        # Call dynamic or canonical generator (bundle-safe)
        # -----------------------------------------------------------
        try:
            result = self.gen4D.generate_by_apex_ladder(
                interceptPoint=serveInterceptPoint,
                bouncePoint=(xb, yb),
                apexHeight=apexHeight,
                apexValues=self.apexValues,
                spinTopRpm=spinTopRpm,
                spinSideRpm=spinSideRpm,
                downhillSpeedPreferred=downhillSpeedPreferred,
                shotType="SERVE",
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
                interceptPoint=serveInterceptPoint,
                bouncePoint=(xb, yb),
                spinTopRpm=spinTopRpm,
                spinSideRpm=spinSideRpm,
                apexHeight=apexHeight,
                serveSide=serveSideNormalized,
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
                        interceptPoint=serveInterceptPoint,
                        bouncePoint=(xb, yb),
                        apexHeight=retryApex,
                        apexValues=self.apexValues,
                        spinTopRpm=spinTopRpm,
                        spinSideRpm=spinSideRpm,
                        downhillSpeedPreferred=downhillSpeedPreferred,
                        shotType="SERVE",
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
        # Accept the trajectory
        # -----------------------------------------------------------
        b = int(traj["bounceIndex"])
        bx_sim = float(traj["fencesX"][b])
        by_sim = float(traj["fencesY"][b])
        initialVelocity = traj["initialVelocity"]
        airTravelDistance = traj["airTravelDistance"]
        netClearance = fencesMeta["netClearance"]
        downhillSpeed = float(canonicalMeta.get("downhillSpeed", 0.0))
        distanceToBounce = float(np.hypot(xb - x0, yb - y0))

        # Only enforce strict net-impact rejection for the downhill edge case
        # that previously slipped through: interceptZ ~= apexHeight.
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
            return self._FailWithIntended(
                interceptPoint=serveInterceptPoint,
                bouncePoint=(xb, yb),
                spinTopRpm=spinTopRpm,
                spinSideRpm=spinSideRpm,
                apexHeight=apexHeight,
                serveSide=serveSideNormalized,
                reason="NO_NET_CLEARANCE",
                failureDetails={"netClearance": netClearance},
            )

        intended = {
            "type": "ON_DEMAND_4D",
            "bounceX": float(xb),
            "bounceY": float(yb),
            "serviceCell": {"col": int(col), "row": int(row)},
            "spinTopRpm": spinTopRpm,
            "spinSideRpm": spinSideRpm,
            "apexHeight": traj["apexHeight"], 
            "distanceToBounce": distanceToBounce,
            "downhillSpeed": downhillSpeed,
        }


        # -----------------------------------------------------------
        # Populate simMeta fields
        # -----------------------------------------------------------
        simMeta["shotType"] = "SERVE"
        simMeta["hitter"] = "PLAYER_BLUE"

        # -----------------------------------------------------------
        # Construct final shot dictionary
        # -----------------------------------------------------------
        shot = {
            "type": "SERVE",
            "shotType": "SERVE",
            "serveSide": serveSideNormalized,
            "hitter": "PLAYER_BLUE",
            "intendedEntry": intended,
            "interceptPoint": (x0, y0, z0),
            "panAngleDeg": float(traj.get("panAngleDeg", 0.0)),
            "solverIters": int(traj.get("solverIters", 0)),
            "landingErrorXY": float(traj.get("landingErrorXY", 0.0)),
            "netClearDeficit": float(traj.get("netClearDeficit", 0.0)),
            "netClearExcess": float(traj.get("netClearExcess", 0.0)
            )
            if "netClearExcess" in traj
            else 0.0,
            "bounceX": bx_sim,
            "bounceY": by_sim,
            "landingX": float(traj["landingX"]),
            "landingY": float(traj["landingY"]),
            "distanceToBounce": distanceToBounce,
            "apexHeight": traj["apexHeight"], 
            "initialVelocity": initialVelocity,
            "airTravelDistance": airTravelDistance,
            "netClearance": netClearance,
            "downhillSpeed": downhillSpeed,
            "spinTopRpm": spinTopRpm,
            "spinSideRpm": spinSideRpm,
            "transformed": traj,
            "opponentPosition": None,
            "defensivePosX": self.court.centerLineX,
            "defensivePosY": self.court.serverBaselineY,
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
    def _ComputeServeInterceptionPoint(self, serveSide: str):
        x, y = self.court.serveInterceptionPoseForServe(serveSide)

        # service height is random between 2.7, 3.0 and 3.3
        return (x, y, np.random.choice(self.interceptZValues[-3:]))

    def _GetAllowedServeBounceCells(self, serveSide: str, interceptY: float):
        allCells = ServiceBoxCells(self.court, serveSide, forHitter="PLAYER_BLUE")
        distanceBehindBaseline = float(self.court.serverBaselineY - interceptY)
        frontRowsToCull = 2 if (distanceBehindBaseline >= 4.572) else (1 if distanceBehindBaseline > 0.0 else 0)

        if frontRowsToCull <= 0:
            return set(allCells)

        opponentRowsSorted = OpponentRowsSortedNearNet(
            self.court, forHitter="PLAYER_BLUE"
        )
        blockedRows = set(opponentRowsSorted[:frontRowsToCull])
        return {
            (colValue, rowValue)
            for (colValue, rowValue) in allCells
            if rowValue not in blockedRows
        }

    def _FailWithIntended(
        self,
        interceptPoint,     
        bouncePoint,
        spinTopRpm,
        spinSideRpm,
        apexHeight,
        serveSide=None,
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

        if self.debug or self.debugLevel1:
            print("FAILED BOUNCE X AND Y: " + str(bounceX) + " " + str(bounceY))
            print("FAILED BOUNCE DISTANCE: " + str(distanceToBounce))
            print("FAILED PAN_DEG: " + str(pan_deg))
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
            "type": "SERVE",
            "shotType": "SERVE",
            "serveSide": (serveSide or "").upper(),
            "hitter": "PLAYER_BLUE",
            "intendedEntry": intended,
            "interceptPoint": (interceptX, interceptY, interceptZ),
            "panAngleDeg": pan_deg,
            "bounceX": float(bounceX),
            "bounceY": float(bounceY),
            "landingX": float(bounceX),
            "landingY": float(bounceY),
            "distanceToBounce": distanceToBounce,
            "apexHeight": 0.0 if apexHeight is None else float(apexHeight),
            "initialVelocity": 0.0,
            "airTravelDistance": 0.0,
            "netClearance": 0.0,
            "spinTopRpm": int(spinTopRpm),
            "spinSideRpm": int(spinSideRpm),
            "transformed": None,
            "opponentPosition": None,
            "defensivePosX": self.court.centerLineX,
            "defensivePosY": self.court.serverBaselineY,
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