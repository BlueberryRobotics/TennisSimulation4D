# Trajectory4DPhysics.py
# Canonical physics upgraded to match dynamic generator physics exactly.
# Pre-bounce → bounce → must-rise → post-bounce → second bounce → stop.

import numpy as np
import math

class Trajectory4DPhysics:
    def __init__(self, dt=0.0025):
        self.dt_far = 0.02
        self.dt_near = 0.006
        self.z_switch = 1.0
        self.maxTime = 5.5

        # Physical constants (match dynamic generator)
        self.g = 9.81
        self.ballMass = 0.057
        self.ballRadius = 0.0335
        self.airDensity = 1.225
        self.dragCoefficient = 0.55
        self.crossSectionArea = math.pi * self.ballRadius**2
        self.dragFactor = 0.5 * self.airDensity * self.dragCoefficient * self.crossSectionArea / self.ballMass

        # Magnus
        self.magnusK = 1.25e-4

        # Bounce model
        self.restitution = 0.70
        self.mu_t = 0.06
        self.spinToTan = 5.0e-4
        self.hystRiseZ = 0.05
        self.minVzRebound = 0.20

    # ---------------------------------------------------------
    # Acceleration = drag + Magnus + gravity
    # ---------------------------------------------------------
    def _accel(self, v, omega):
        s = np.linalg.norm(v) + 1e-12
        a_drag = -self.dragFactor * s * v
        a_magnus = self.magnusK * np.cross(omega, v)
        return a_drag + a_magnus + np.array([0.0, 0.0, -self.g])

    def _step(self, p, v, omega, dt):
        a = self._accel(v, omega)
        v2 = v + a * dt
        p2 = p + v2 * dt
        return p2, v2

    # ---------------------------------------------------------
    # Bounce resolution (matches dynamic)
    # ---------------------------------------------------------
    def _apply_bounce(self, p_imp, v_imp, omega):
        x, y, _ = p_imp
        vx, vy, vz = v_imp

        # if rising or skimming
        if vz >= 0.0:
            return np.array([x, y, 0.0]), np.array([vx, vy, 0.0]), False

        vz_out = -self.restitution * vz
        Jn = (1.0 + self.restitution) * (-vz)

        def t_adj(vt):
            if abs(vt) < 1e-9:
                return 0.0
            return vt - self.mu_t * Jn * np.sign(vt)

        vx_out = t_adj(vx)
        vy_out = t_adj(vy) + self.spinToTan * omega[2]

        # If rebound is too weak, stop
        if vz_out < self.minVzRebound:
            return np.array([x, y, 0.0]), np.array([vx_out, vy_out, 0.0]), False

        return np.array([x, y, 0.0]), np.array([vx_out, vy_out, vz_out]), True

    # ---------------------------------------------------------
    # Full simulation: pre-bounce → bounce → rise → post-bounce → second bounce
    # ---------------------------------------------------------
    def simulate(self, interceptPoint, v0, spinTop, spinSide):
        p = np.array(interceptPoint, float)
        v = np.array(v0, float)
        initialVelocity = np.linalg.norm(v0)

        # canonical spin vectors (Option A2)
        def rpm_to_rad(rpm): return (2*math.pi*rpm)/60.0
        omega = np.array([rpm_to_rad(spinTop), 0.0, rpm_to_rad(spinSide)], float)

        Xs, Ys, Zs = [], [], []
        bounceIndex = None
        bounceCount = 0
        mustRise = False
        Ts = []

        t = 0.0
        steps = 0
        maxSteps = int(self.maxTime / self.dt_near) + 4
        prevX = 0
        prevY = 0
        prevZ = 0
        airTravelDistance = 0.0

        while steps < maxSteps and t <= self.maxTime:

            dt = self.dt_near if p[2] <= self.z_switch else self.dt_far
            p2, v2 = self._step(p, v, omega, dt)
            t2 = t + dt

            # accumulating actual air travel distance until first bounce
            if bounceIndex is None:
                # accumulate distance ONLY until bounce occurs
                dx = p2[0] - prevX
                dy = p2[1] - prevY
                dz = p2[2] - prevZ
                airTravelDistance += math.sqrt(dx*dx + dy*dy + dz*dz)

                prevX, prevY, prevZ = p2[0], p2[1], p2[2]

            # Bounce detection
            if p2[2] <= 0.05 and not mustRise:
                # impact interpolation
                z0, z1 = p[2], p2[2]
                tau = dt * z0 / (z0 - z1 + 1e-12)
                v_imp = v + (v2 - v) * (tau/dt)
                p_imp = p + v_imp * tau
                p_imp[2] = 0.0

                p_out, v_out, rebounded = self._apply_bounce(p_imp, v_imp, omega)

                # record exact impact point
                Xs.append(p_imp[0])
                Ys.append(p_imp[1])
                Zs.append(0.0)
                Ts.append(t)

                if bounceIndex is None:
                    bounceIndex = len(Xs) - 1
                bounceCount += 1

                if bounceCount >= 2 or not rebounded:
                    break

                p, v = p_out, v_out
                mustRise = True
                t = t2
                steps += 1
                continue

            # Normal integration
            p, v, t = p2, v2, t2

            Xs.append(p[0])
            Ys.append(p[1])
            Zs.append(max(0.0, p[2]))
            Ts.append(t)

            # must-rise hysteresis
            if mustRise and p[2] > self.hystRiseZ:
                mustRise = False

            # post-bounce low-speed termination
            # if bounceIndex is not None and p[2] < 0.05 and np.linalg.norm(v) < 1.0:
            #     break

            steps += 1

        # Return canonical-format tuple
        return np.asarray(Xs), np.asarray(Ys), np.asarray(Zs), np.asarray(Ts), bounceIndex, round(initialVelocity,2), round(airTravelDistance,2)