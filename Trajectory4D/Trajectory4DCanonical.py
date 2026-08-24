import math
import numpy as np
from typing import Optional


MPH_TO_MPS = 0.44704


class Trajectory4DCanonical:

    def __init__(
        self,
        trajectoryLibrary,
        transformLayer,
        interceptZValues,
        apexHeight,
        apexValues,
        spinTopValues,
        spinSideValues,
        bounceDistanceTolerance,
        court,
        debug=False
    ):
        self.library = trajectoryLibrary
        self.transformLayer = transformLayer
        self.interceptZValues = np.array(interceptZValues, dtype=float)
        self.apexValues = np.array(apexValues, dtype=float)
        self.spinTopValues = np.array(spinTopValues, dtype=int)
        self.spinSideValues = np.array(spinSideValues, dtype=int)
        self.bounceDistanceTolerance = float(bounceDistanceTolerance)
        self.court = court
        self.debug = debug

        # build index tables
        self.indexByInterceptZ = {}
        self.indexByApex = {}
        self.indexBySpinTop = {}
        self.indexBySpinSide = {}

        for idx, entry in enumerate(self.library):
            iz = float(entry["interceptPoint"][2])
            ah = float(entry["apex_height"])
            st = int(entry["spin_top_rpm"])
            ss = int(entry["spin_side_rpm"])

            self.indexByInterceptZ.setdefault(iz, []).append(idx)
            self.indexByApex.setdefault(ah, []).append(idx)
            self.indexBySpinTop.setdefault(st, []).append(idx)
            self.indexBySpinSide.setdefault(ss, []).append(idx)

    # ------------------------------------------------------------------
    def generate_by_apex_ladder(
        self,
        interceptPoint,
        bouncePoint,
        apexHeight,
        apexValues,
        spinTopRpm,
        spinSideRpm,
        downhillSpeedPreferred=None,
        shotType: Optional[str] = None,
        maxInitialVelocityMps: Optional[float] = None,
        distanceTolerance=None,
        netHeightEps=1e-3, 
        maxNetClearAbove=None,
        landing_tol=None,
        net_eps=None,
        maxItersPerApex=None
    ):
        """
        Canonical selector using forward-distance matching.
        """

        interceptX, interceptY, interceptZ = interceptPoint
        bounceX, bounceY = bouncePoint
        #print("BY APEX LADDER BOUNCEPOINT: " +str(bounceX) + " " + str(bounceY))

        lengthY = abs(bounceY - interceptY)
        lengthX = abs(bounceX - interceptX)

        #fences forward distance
        #fencesForward = float(bounceY - interceptY)
        fencesForward = math.hypot(lengthY, lengthX)
        tol = distanceTolerance if distanceTolerance is not None else self.bounceDistanceTolerance

        # ---------------------------------------
        # 1. interceptZ bin
        # ---------------------------------------
        zIdx = int(np.argmin(np.abs(self.interceptZValues - interceptZ)))
        canonicalInterceptZ = float(self.interceptZValues[zIdx])
        candidates = set(self.indexByInterceptZ.get(canonicalInterceptZ, []))

        if not candidates:
            raise RuntimeError("NO_CANONICAL_INTERCEPT_Z_MATCH")

        # ---------------------------------------
        # 2. spin filter
        # ---------------------------------------
        if spinTopRpm not in self.indexBySpinTop:
            raise RuntimeError("NO_CANONICAL_SPIN_TOP_MATCH")
        if spinSideRpm not in self.indexBySpinSide:
            raise RuntimeError("NO_CANONICAL_SPIN_SIDE_MATCH")

        candidates &= set(self.indexBySpinTop[spinTopRpm])
        candidates &= set(self.indexBySpinSide[spinSideRpm])

        if not candidates:
            raise RuntimeError("NO_CANONICAL_SPIN_COMBO_MATCH")

        # ---------------------------------------
        # 3. apex ladder (FIRST PASSING APEX ONLY)
        # ---------------------------------------
        selectedSurvivors = None
        selectedUsedDownhillSpeed = False

        apexValues = [candidate for candidate in apexValues if candidate >= apexHeight]

        for apexTarget in sorted(apexValues):
            if apexTarget not in self.indexByApex:
                continue

            apexSet = set(self.indexByApex[apexTarget])
            group = candidates & apexSet
            if not group:
                continue

            # ---------------------------------------
            # 4. forward-distance match
            # ---------------------------------------
            bounceDistanceOK = []
            for idx in group:
                canonicalForward = float(self.library[idx]["distance"])
                if abs(canonicalForward - fencesForward) <= tol:
                    bounceDistanceOK.append(idx)

            if not bounceDistanceOK:
                continue

            # ---------------------------------------
            # 5. net-clear
            # ---------------------------------------
            survivors = []
            for idx in bounceDistanceOK:
                entry = self.library[idx]
                if self._canonical_clears_net(
                    entry,
                    interceptPoint,
                    bouncePoint,
                    netHeightEps
                ):
                    survivors.append((idx, entry))

            if survivors:
                # If request is apex==intercept, prefer downhill-band trajectories at
                # this exact apex when feasible; otherwise continue normal apex climb.
                if abs(apexHeight - canonicalInterceptZ) <= 1e-6 and abs(apexTarget - canonicalInterceptZ) <= 1e-6:
                    downhillSurvivors = [
                        survivor
                        for survivor in survivors
                        if self._entry_downhill_speed_mps(survivor[1]) > 0.0
                    ]
                    if downhillSurvivors:
                        downhillByBand = {}
                        for survivor in downhillSurvivors:
                            bandMph = self._entry_downhill_speed_band_mph(survivor[1])
                            if bandMph is None:
                                continue
                            downhillByBand.setdefault(int(bandMph), []).append(survivor)

                        if downhillByBand:
                            selectedBand = None
                            if downhillSpeedPreferred is not None:
                                preferredMph = int(round(float(downhillSpeedPreferred) / MPH_TO_MPS))
                                if preferredMph in downhillByBand:
                                    selectedBand = preferredMph

                            if selectedBand is None:
                                availableBands = sorted(downhillByBand.keys())
                                selectedBand = int(np.random.choice(availableBands))

                            downhillSurvivors = downhillByBand[selectedBand]

                        if downhillSpeedPreferred is not None:
                            targetSpeed = round(float(downhillSpeedPreferred), 4)
                            matchedBySpeed = [
                                survivor
                                for survivor in downhillSurvivors
                                if abs(round(self._entry_downhill_speed_mps(survivor[1]), 4) - targetSpeed) <= 1e-6
                            ]
                            if matchedBySpeed:
                                downhillSurvivors = matchedBySpeed
                        selectedSurvivors = downhillSurvivors
                        selectedUsedDownhillSpeed = True
                        break
                    # No valid downhill trajectories at this apex: keep climbing.
                    continue

                # Standard first-passing-apex behavior.
                selectedSurvivors = survivors
                break

        # ---------------------------------------
        # 6. choose trajectory from surviving set
        #    - downhill-specific branch: keep random band behavior
        #    - standard branch: pick fastest
        # ---------------------------------------
        if not selectedSurvivors:
            raise RuntimeError("NO_CANONICAL_APEX_DISTANCE_NETCLEAR_MATCH")

        if maxInitialVelocityMps is not None:
            selectedSurvivors = [
                (candidateIdx, candidateEntry)
                for candidateIdx, candidateEntry in selectedSurvivors
                if self._entry_initial_velocity_mps(candidateEntry) <= float(maxInitialVelocityMps)
            ]
            if not selectedSurvivors:
                raise RuntimeError("NO_CANONICAL_APEX_DISTANCE_NETCLEAR_MATCH")

        if selectedUsedDownhillSpeed:
            idx, entry = selectedSurvivors[np.random.randint(len(selectedSurvivors))]
        else:
            survivorSpeeds = [self._entry_initial_velocity_mps(candidateEntry) for _, candidateEntry in selectedSurvivors]
            fastestSpeed = max(survivorSpeeds)
            fastestCandidates = [
                selectedSurvivors[candidateIndex]
                for candidateIndex, candidateSpeed in enumerate(survivorSpeeds)
                if abs(candidateSpeed - fastestSpeed) <= 1e-9
            ]
            restCandidates = [
                selectedSurvivors[candidateIndex]
                for candidateIndex, candidateSpeed in enumerate(survivorSpeeds)
                if abs(candidateSpeed - fastestSpeed) > 1e-9
            ]

            # Weighted tier selection for non-downhill shots:
            # 60% from fastest tier, 40% from the remaining survivors.
            if restCandidates and float(np.random.random()) >= 0.60:
                idx, entry = restCandidates[np.random.randint(len(restCandidates))]
            else:
                idx, entry = fastestCandidates[np.random.randint(len(fastestCandidates))]

        # pan angle in radians calculation
        fx = float(bounceX - interceptX)
        fy = float(bounceY - interceptY)

        norm = math.hypot(fx, fy)
        fx /= norm
        fy /= norm

        # Rotation that maps canonical forward (0,+1) to fences forward (fx,fy)
        panAngle = math.atan2(-fx, fy)
        panAngleDeg = math.degrees(panAngle)

        # print("CANONICAL PAN ANGLE: " + str(panAngleDeg))

        fenceTraj = self.transformLayer.applyTransform(
            entry=entry,
            interceptPoint=interceptPoint,
            panAngleDeg=panAngleDeg,
            bouncePoint=bouncePoint,
            bounceCellHalfW=tol
        )

        # world-coord arrays after transform
        Xw = fenceTraj["fencesX"]
        Yw = fenceTraj["fencesY"]
        Zw = fenceTraj["fencesZ"]

        netY = self.court.netY
        netHeight = self.court.netHeight  

        # find crossing index
        crossingIdx = np.where( (Yw[:-1] - netY) * (Yw[1:] - netY) <= 0 )[0]
        if len(crossingIdx) > 0:
            k = crossingIdx[0]
            y0, y1 = Yw[k], Yw[k+1]
            z0, z1 = Zw[k], Zw[k+1]

            # interpolation fraction
            t = (netY - y0) / (y1 - y0 + 1e-12)

            z_at_net = z0 + t * (z1 - z0)
            clearance = z_at_net - netHeight

            netClearance = round(float(clearance),2)
        else:
            netClearance = None  # never crossed net

        # print("NET CLEARANCE " + str(netClearance))

        return {
            "traj": fenceTraj,
            "canonicalMeta": {
                "canonicalIndex": idx,
                "canonicalInterceptZ": canonicalInterceptZ,
                "canonicalApexHeight": entry["apex_height"],
                "canonicalSpinTopRpm": entry["spin_top_rpm"],
                "canonicalSpinSideRpm": entry["spin_side_rpm"],
                "canonicalForwardDistance": entry["distance"],
                "downhillSpeed": self._entry_downhill_speed_mps(entry),
                "usedDownhillSpeedSelection": bool(selectedUsedDownhillSpeed),
            },
            "fencesMeta": {
                "interceptPoint": interceptPoint,
                "bouncePoint": bouncePoint,
                "targetForward": fencesForward,
                "panAngleDeg": panAngleDeg,
                "netClearance": netClearance,
            },
            "simMeta": {
                "pointIndex": None,
                "shotIndexInPoint": None,
                "hitter": None,
                "shotType": None
            }
        }

    def _entry_downhill_speed_mps(self, entry) -> float:
        speedMph = entry.get("speed_mph")
        if speedMph is not None:
            try:
                speedMphValue = float(speedMph)
                if speedMphValue > 0.0:
                    return speedMphValue * MPH_TO_MPS
            except Exception:
                pass

        solveMode = str(entry.get("solve_mode") or entry.get("solveMode") or "").strip().lower()
        if solveMode in {"flat_speed_fan", "downhill_speed"}:
            try:
                speedMps = float(entry.get("initialVelocity", 0.0))
                return speedMps if speedMps > 0.0 else 0.0
            except Exception:
                return 0.0

        return 0.0

    def _entry_downhill_speed_band_mph(self, entry):
        speedMph = entry.get("speed_mph")
        if speedMph is not None:
            try:
                speedMphValue = int(round(float(speedMph)))
                if speedMphValue > 0:
                    return speedMphValue
            except Exception:
                pass

        try:
            initialVelocityMps = float(entry.get("initialVelocity", 0.0))
            if initialVelocityMps <= 0.0:
                return None
            speedMphValue = int(round(initialVelocityMps / MPH_TO_MPS))
            return speedMphValue if speedMphValue > 0 else None
        except Exception:
            return None

    def _entry_initial_velocity_mps(self, entry) -> float:
        try:
            initialVelocity = float(entry.get("initialVelocity", 0.0))
            if initialVelocity > 0.0:
                return initialVelocity
        except Exception:
            pass

        speedMph = entry.get("speed_mph")
        if speedMph is not None:
            try:
                speedMphValue = float(speedMph)
                return speedMphValue * MPH_TO_MPS if speedMphValue > 0.0 else 0.0
            except Exception:
                return 0.0

        return 0.0


    def _canonical_clears_net(self, entry, interceptPoint, bouncePoint, eps):
        """
        Fast approximate net-clearance check in canonical space.
        Uses linearized forward progression adjusted by panAngle.
        Good enough to pre-filter 100s of trajectories before TransformLayer.
        """

        interceptX, interceptY, interceptZ = interceptPoint
        bx, by = bouncePoint

        # 1. Compute pan angle for this shot (canonical-to-fences forward alignment)
        panAngle = math.atan2(bx - interceptX, by - interceptY)
        cosA = math.cos(panAngle)
        sinA = math.sin(panAngle)

        # 2. Canonical arrays
        y = np.asarray(entry["canonY"], float)
        z = np.asarray(entry["canonZ"], float)

        # 3. Canonical forward displacement from launch
        dy = y - y[0]

        # 4. Effective forward displacement *after rotation* (approx)
        #    dy_eff ≈ dy * cosA     (ignores dx*sinA term, acceptable for pre-filtering)
        dy_eff = dy * cosA          # fast approximation

        # 5. Solve for the dy_eff needed to reach the net plane
        netY = self.court.netY
        needed_dy_eff = netY - interceptY

        # 6. Convert needed forward distance into canonical dy index
        #    dy ≈ needed_dy_eff / cosA
        if abs(cosA) < 1e-9:
            return False  # extreme horizontal angle, bail

        dy_needed = needed_dy_eff / cosA

        # 7. Interpolate canonical z at that dy value
        #    dy increases monotonically, so we can use dy directly as the lookup
        if dy_needed <= dy[0]:
            z_at_net = z[0]
        elif dy_needed >= dy[-1]:
            z_at_net = z[-1]
        else:
            z_at_net = float(np.interp(dy_needed, dy, z))

        # 8. Final check
        return z_at_net >= (self.court.netHeight - eps)