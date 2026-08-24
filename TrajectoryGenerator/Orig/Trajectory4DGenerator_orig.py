import numpy as np
import math
from Trajectory4DPhysics import Trajectory4DPhysics


class Trajectory4DGenerator:

    def __init__(self, dt=0.0025):
        self.physics = Trajectory4DPhysics(dt=dt)

    def _snapApexToBins(self, apexValue, apexValues):
        """
        Snap physics-produced apex to nearest canonical apex bin.
        """
        bins = np.asarray(apexValues, dtype=float)
        idx = np.argmin(np.abs(bins - float(apexValue)))
        return float(bins[idx])

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
        Canonical generator: 
        * canonical intercept ALWAYS at (0,0,interceptZ)
        * canonical bounce ALWAYS at (0, forwardDistance)
        """

        # canonical intercept geometry
        x0, y0, z0 = interceptPoint    # should always be (0,0,z)
        xb, yb = bouncePoint           # should be (0, forwardDistance)

        # enforce apex >= intercept height
        targetApex = apexHeight
        if targetApex <= z0:
            targetApex = z0 + 0.01

        g = self.physics.g

        # initial upward velocity guess
        vz_initial = math.sqrt(2.0 * g * (targetApex - z0))

        # forward distance
        dy = float(yb - y0)
        if abs(dy) < 1e-6:
            dy = 1e-6

        time_to_apex = vz_initial / g
        if time_to_apex < 0.15:
            time_to_apex = 0.15

        vy_initial = dy / (2.0 * time_to_apex)
        theta = math.atan2(vz_initial, vy_initial)

        # small apex-correction loop
        for _ in range(5):
            speed = math.sqrt(vy_initial * vy_initial + vz_initial * vz_initial)
            vx0 = 0.0
            vy0 = speed * math.cos(theta)
            vz0 = speed * math.sin(theta)

            Xc, Yc, Zc, bounceIndex = self.physics.simulate(
                interceptPoint=(x0, y0, z0),
                v0=(vx0, vy0, vz0),
                spinTop=spinTopRpm,
                spinSide=spinSideRpm
            )

            if bounceIndex is None or bounceIndex <= 1:
                break

            actualApex = float(np.max(Zc[:bounceIndex+1]))
            diff = targetApex - actualApex

            if abs(diff) < 0.02:
                break

            theta += 0.45 * diff

        # final physics solve
        Xc, Yc, Zc, bounceIndex = self.physics.simulate(
            interceptPoint=(x0, y0, z0),
            v0=(0.0, vy_initial, vz_initial),
            spinTop=spinTopRpm,
            spinSide=spinSideRpm
        )

        if bounceIndex is None or bounceIndex <= 1:
            return None

        t = np.arange(len(Xc))