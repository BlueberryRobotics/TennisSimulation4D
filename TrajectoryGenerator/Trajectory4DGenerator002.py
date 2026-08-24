import math
import numpy as np
from Trajectory4DPhysics import Trajectory4DPhysics


class Trajectory4DGenerator:

    def __init__(self, dt: float = 0.0025):
        self.physics = Trajectory4DPhysics(dt=dt)
        self.dt = dt
        self.g = self.physics.g

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

        _, _, z0 = interceptPoint
        _, bounceY = bouncePoint

        forwardTarget = float(bounceY)
        apexTarget = float(apexHeight)

        # tighter bounce tolerance for shallow apex
        bounceTol = 0.15 if apexTarget < 3.0 else 0.34

        sol = self._solve_broyden(
            z0, forwardTarget, apexTarget,
            spinTopRpm, spinSideRpm,
            bounceTol=bounceTol
        )

        if sol is None:
            return None

        vy0, vz0, (X, Y, Z, time, bi, actualApex, initialVelocity, airTravelDistance) = sol

        # ⭐ Final apex enforcement
        if abs(actualApex - apexTarget) > 0.15:
            return None

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
            "bounceIndex": bi,
            "initialVelocity":initialVelocity,
            "airTravelDistance":airTravelDistance,
            "landingX": float(X[landingIndex]),
            "landingY": float(Y[landingIndex]),
        }

        return entry