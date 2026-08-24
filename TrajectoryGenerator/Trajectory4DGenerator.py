import math
import numpy as np
from Trajectory4DPhysics import Trajectory4DPhysics

MPH_TO_MPS = 0.44704


class Trajectory4DGenerator:

    def __init__(self, dt: float = 0.0025, velocityArrayMph=None):
        self.physics = Trajectory4DPhysics(dt=dt)
        self.dt = dt
        self.g = self.physics.g

        # velocities explicitly allowed for downward shots
        # current trajectory library does not include 115 and 130
        self.velocityArrayMph = velocityArrayMph or [70, 85, 100, 115, 130]

    # -------------------------------------------------------
    # spin → canonical omega
    # -------------------------------------------------------
    @staticmethod
    def _rpm_to_radsec(rpm: float) -> float:
        return (2.0 * math.pi * rpm) / 60.0

    def _omega(self, spinTop: float, spinSide: float) -> np.ndarray:
        return np.array([
            self._rpm_to_radsec(spinTop),
            0.0,
            self._rpm_to_radsec(spinSide)
        ], float)
    
        # -------------------------------------------------------
    def _snapApex(self, apex: float, bins) -> float:
        bins = np.asarray(bins, float)
        return float(bins[np.argmin(np.abs(bins - apex))])

 # -------------------------------------------------------
    # ERROR FUNCTION with APEX WEIGHTING
    # -------------------------------------------------------
    def _errors(self, vy0, vz0, z0, forwardTarget, apexTarget,
                spinTop, spinSide):

        canonX, canonY, canonZ, time, bIdx, initialVelocity, airTravelDistance = self.physics.simulate(
            interceptPoint=(0.0, 0.0, z0),
            v0=(0.0, vy0, vz0),
            spinTop=spinTop,
            spinSide=spinSide
        )

        # failed run
        if bIdx is None or bIdx <= 0:
            return (1e6, 1e6, None)

        # actual apex (before bounce)
        actualApex = float(np.max(canonZ[:bIdx+1]))

        # APEX WEIGHT: forces solver to care about apex as much as bounce
        APEX_WEIGHT = 3.5
        eApex = APEX_WEIGHT * (apexTarget - actualApex)

        # bounce distance error
        actualBounceY = float(canonY[bIdx])
        eBounce = forwardTarget - actualBounceY

        return (eApex, eBounce, (canonX, canonY, canonZ, time, bIdx, actualApex, initialVelocity, airTravelDistance))

    # -------------------------------------------------------
    # BROYDEN SOLVER with APEX ENFORCEMENT + STABILITY CLAMPS
    # -------------------------------------------------------
    def _solve_broyden(self, z0, forwardTarget, apexTarget, spinTop, spinSide,
                       maxIter=40, apexTol=0.04, bounceTol=0.34):

        # initial vz guess (clamped)
        raw_vz = 2 * self.g * (apexTarget - z0)
        raw_vz = max(raw_vz, 0.0)
        vz = math.sqrt(raw_vz)

        # initial vy guess
        vy = forwardTarget / max(0.1, (2 * vz / self.g)) if vz > 0 else 3.0

        J = np.eye(2)
        eA, eD, traj = self._errors(vy, vz, z0, forwardTarget,
                                    apexTarget, spinTop, spinSide)
        if traj is None:
            return None

        e_prev = np.array([eA, eD], float)

        for _ in range(maxIter):

            # BOTH constraints must be satisfied
            if abs(e_prev[1]) < bounceTol and abs(e_prev[0]) < apexTol:
                _, _, _, _, _, actualApex, _, _, = traj
 
                # APEX ENFORCEMENT — do NOT accept high-apex drift
                if abs(actualApex - apexTarget) <= apexTol * 1.5:
                    return (vy, vz, traj)
                else:
                    return None

            # Broyden update
            try:
                dp = -np.linalg.solve(J, e_prev)
            except:
                dp = -0.05 * e_prev

            dp = np.clip(dp, -3.0, 3.0)
            vy_new = vy + dp[0]
            vz_new = vz + dp[1]

            # ⭐ stability: never allow vz to collapse
            vz_new = max(vz_new, 0.1)

            # evaluate new state
            eA2, eD2, traj2 = self._errors(
                vy_new, vz_new, z0, forwardTarget, apexTarget,
                spinTop, spinSide
            )

            # backtrack if needed
            if traj2 is None:
                vy_new = vy + 0.3 * dp[0]
                vz_new = max(vz + 0.3 * dp[1], 0.1)
                eA2, eD2, traj2 = self._errors(
                    vy_new, vz_new, z0, forwardTarget, apexTarget,
                    spinTop, spinSide
                )
                if traj2 is None:
                    return None

            e_new = np.array([eA2, eD2])

            # update Jacobian
            dv = np.array([vy_new - vy, vz_new - vz])
            de = e_new - e_prev
            denom = dv @ dv + 1e-12
            J += np.outer((de - J @ dv), dv) / denom

            # advance
            vy, vz = vy_new, vz_new
            e_prev = e_new
            traj = traj2

        return None
    
    # -------------------------------------------------------
    # PUBLIC: generateCanonicalEntry
    # -------------------------------------------------------
    def generateCanonicalEntry(
        self,
        interceptPoint,
        bouncePoint,
        apexHeight,
        spinTopRpm,
        spinSideRpm,
        apexValues
    ):
        """
        Generate one or more canonical trajectories for the given geometry.

        - Normal case (apexHeight > interceptZ):
            Use Broyden solver exactly as before.

        - Degenerate downhill case (apexHeight == interceptZ):
            Use geometric downhill angle and vary speed only.
        """

        entries = []

        _, _, z0 = interceptPoint
        _, bounceY = bouncePoint

        forwardTarget = float(bounceY)
        apexTarget = float(apexHeight)

        # ------------------------------------------------------------------
        # CASE 1: DEGENERATE DOWNHILL (apex == interceptZ)
        # ------------------------------------------------------------------
        if abs(apexTarget - z0) < 1e-6:

            # Use configured downhill speed menu and solve launch angle per speed.
            # This includes tiny upward and downward angles around near-level launch,
            # while still allowing steeper downhill angles when geometry needs it.
            speed_mph_list = sorted({int(v) for v in self.velocityArrayMph})

            for speedMph in speed_mph_list:
                speedMps = float(speedMph) * MPH_TO_MPS
                sol = self._solve_downhill_fan(
                    z0=z0,
                    forwardTarget=forwardTarget,
                    speedMps=speedMps,
                    spinTop=spinTopRpm,
                    spinSide=spinSideRpm,
                    bounceTol=0.75,
                    upwardAngleCapDeg=1.5,
                    downwardSlackDeg=10.0,
                    stepDeg=0.05,
                )

                if sol is None:
                    continue

                entry = self._build_entry(
                    sol=sol,
                    interceptPoint=interceptPoint,
                    bouncePoint=bouncePoint,
                    apexTarget=apexTarget,
                    spinTopRpm=spinTopRpm,
                    spinSideRpm=spinSideRpm,
                    apexValues=apexValues,
                    solveMode="flat_speed_fan",
                    speedMph=float(speedMph),
                )
                entries.append(entry)

            return entries

        # ------------------------------------------------------------------
        # CASE 2: NORMAL (apex > interceptZ) — ORIGINAL BEHAVIOR
        # ------------------------------------------------------------------
        if apexTarget > z0:

            bounceTol = 0.15 if apexTarget < 3.0 else 0.34

            sol = self._solve_broyden(
                z0,
                forwardTarget,
                apexTarget,
                spinTopRpm,
                spinSideRpm,
                bounceTol=bounceTol
            )

            if abs(apexTarget - z0) < 1e-6:
                raise RuntimeError("Broyden should never be called for apex == interceptZ")

            if sol is None:
                return []

            vy0, vz0, (
                X, Y, Z, time, bIdx,
                actualApex, initialVelocity, airTravelDistance
            ) = sol

            # Final apex enforcement
            if abs(actualApex - apexTarget) > 0.15:
                return []

            Z = np.clip(Z, 0.0, None)
            landingIndex = len(X) - 1

            entry = {
                "interceptPoint": interceptPoint,
                "bouncePoint": bouncePoint,
                "distance": forwardTarget,
                "apex_height": self._snapApex(apexTarget, apexValues),
                "spin_top_rpm": spinTopRpm,
                "spin_side_rpm": spinSideRpm,
                "canonX": X,
                "canonY": Y,
                "canonZ": Z,
                "time": time,
                "trajectory3D": np.column_stack([X, Y, Z]),
                "bounceIndex": bIdx,
                "initialVelocity": initialVelocity,
                "airTravelDistance": airTravelDistance,
                "landingY": float(Y[landingIndex]),
                "solveMode": "broyden"
            }

            return [entry]

        # ------------------------------------------------------------------
        # OTHERWISE: invalid geometry
        # ------------------------------------------------------------------
        return []

    # -------------------------------------------------------
    # NEW: Solve with speed specified (downward shots only)
    # -------------------------------------------------------
    def _solve_downhill_fan(
        self,
        z0,
        forwardTarget,
        speedMps,
        spinTop,
        spinSide,
        bounceTol=0.75,
        upwardAngleCapDeg=1.5,
        downwardSlackDeg=10.0,
        stepDeg=0.05,
    ):
        # Angle from horizontal, positive = downward.
        thetaGeomDeg = math.degrees(math.atan2(z0, max(forwardTarget, 1e-9)))

        # Search window includes slightly upward shots near horizontal and
        # steeper downhill shots when needed for shorter targets.
        thetaMinDeg = min(-abs(float(upwardAngleCapDeg)), thetaGeomDeg - float(downwardSlackDeg))
        thetaMaxDeg = max(abs(float(upwardAngleCapDeg)), thetaGeomDeg + float(downwardSlackDeg))
        thetaMinDeg = max(thetaMinDeg, -12.0)
        thetaMaxDeg = min(thetaMaxDeg, 70.0)

        stepDeg = max(0.01, float(stepDeg))
        thetaDegValues = np.arange(thetaMinDeg, thetaMaxDeg + 0.5 * stepDeg, stepDeg)

        best = None
        bestErr = float("inf")
        bestAbsTheta = float("inf")

        for thetaDeg in thetaDegValues:
            theta = math.radians(thetaDeg)

            vy = speedMps * math.cos(theta)
            vz = -speedMps * math.sin(theta)   # theta<0 yields slight upward launch

            canonX, canonY, canonZ, time, bIdx, initVel, airDist = \
                self.physics.simulate(
                    interceptPoint=(0.0, 0.0, z0),
                    v0=(0.0, vy, vz),
                    spinTop=spinTop,
                    spinSide=spinSide
                )

            if bIdx is None or bIdx <= 0:
                continue

            err = abs(float(canonY[bIdx]) - float(forwardTarget))
            if err > bounceTol:
                continue

            absTheta = abs(thetaDeg)
            # Prefer smaller bounce error; tie-break toward near-level angles.
            if (err < bestErr) or (abs(err - bestErr) <= 1e-9 and absTheta < bestAbsTheta):
                bestErr = err
                bestAbsTheta = absTheta
                best = (
                    vy,
                    vz,
                    (canonX, canonY, canonZ, time, bIdx, initVel, airDist),
                )

        return best

    # -------------------------------------------------------
    # Helper: build trajectory dictionary
    # -------------------------------------------------------
    def _build_entry(
        self,
        sol,
        interceptPoint,
        bouncePoint,
        apexTarget,
        spinTopRpm,
        spinSideRpm,
        apexValues,
        solveMode,
        speedMph=None
    ):
        vy0, vz0, (X, Y, Z, time, bi, initVel, airDist) = sol

        Z = np.clip(Z, 0.0, None)
        landingIndex = len(X) - 1

        return {
            "interceptPoint": interceptPoint,
            "bouncePoint": bouncePoint,
            "distance": float(bouncePoint[1]),
            "apex_height": float(apexTarget),
            "spin_top_rpm": spinTopRpm,
            "spin_side_rpm": spinSideRpm,
            "canonX": X,
            "canonY": Y,
            "canonZ": Z,
            "time": time,
            "trajectory3D": np.column_stack([X, Y, Z]),
            "bounceIndex": bi,
            "initialVelocity": initVel,
            "airTravelDistance": airDist,
            "landingY": float(Y[landingIndex]),
            "solve_mode": solveMode,
            "speed_mph": speedMph,
        }