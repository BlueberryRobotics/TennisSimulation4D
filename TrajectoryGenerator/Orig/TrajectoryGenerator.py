# Trajectory3DGenerator.py
#
# Modernized HY physics generator for canonical 3D trajectories.
# Lightweight, fast, multiprocessing-safe, homotopy optional.
#
# Produces:
#   { "fencesX", "fencesY", "fencesZ", "t",
#     "bounceIndex", "landingX","landingY",
#     "apexHeight", "panAngleDeg" }

import numpy as np
import math

class Trajectory3DGenerator:
    def __init__(self, court, dt=0.0025):
        self.court = court
        self.dt = dt
        self.g = 9.81

    
    def _snapApex(self, apex):
        bins = np.array([1.0,1.3,1.6,2.0,2.6,2.7,2.8,3.0,3.6,4.5,6.0,8.0,10.0])
        return float(bins[np.argmin(np.abs(bins - apex))])


    # -------------------------------------------------------
    # Simple explicit RK integration (faster than HY's full solver)
    # -------------------------------------------------------
    def _integrate(self, s0, theta, phi, spinTop, spinSide):
        dt = self.dt
        g = self.g
        # Magnus constants (you can tune)
        k_drag = 0.002
        k_mag  = 0.0004

        # Initial velocity components
        vx = s0 * math.cos(theta) * math.cos(phi)
        vy = s0 * math.sin(theta)
        vz = s0 * math.cos(theta) * math.sin(phi)

        # state arrays
        X,Y,Z,T = [],[],[],[]
        x=y=z=0.0
        t=0.0

        for _ in range(50000):  # 50000*dt=125s max airtime
            # record
            X.append(x); Y.append(y); Z.append(z); T.append(t)

            # net collision stop
            if y > self.court.lengthGlobal:
                break
            if z < 0:
                break

            # speed
            v = math.sqrt(vx*vx + vy*vy + vz*vz)

            # drag
            drag = k_drag * v
            dvx_drag = -drag * vx
            dvy_drag = -drag * vy
            dvz_drag = -drag * vz

            # simple Magnus model
            dvx_mag =  k_mag * spinSide * vz
            dvy_mag = -k_mag * spinTop  * vx
            dvz_mag = -k_mag * spinSide * vx

            # update velocities
            vx += dt * (dvx_drag + dvx_mag)
            vy += dt * (dvy_drag + dvy_mag)
            vz += dt * (dvz_drag + dvz_mag - g)

            # update positions
            x += dt * vx
            y += dt * vy
            z += dt * vz
            t += dt

        return np.array(X), np.array(Y), np.array(Z), np.array(T)

    # -------------------------------------------------------
    def generate(self, launchPoint, bounceXY, apexHeight,
                 spinTopRpm, spinSideRpm,
                 seed=None):
        x0, y0, z0 = launchPoint
        xb, yb = bounceXY

        # convert spin
        spinTop  = spinTopRpm  / 60.0
        spinSide = spinSideRpm / 60.0

        # rough vacuum estimate for speed
        dx = xb - x0
        dy = yb - y0
        dist = math.hypot(dx, dy)
        s_guess = math.sqrt(self.g * apexHeight * 2)

        if seed:
            theta0 = seed.get("theta", 0.1)
            s0     = seed.get("speed", s_guess)
            phi0   = seed.get("phi", 0.0)
        else:
            theta0 = math.atan2(dy, dist*0.8)
            phi0   = 0.0
            s0     = s_guess

        # iterate a few refinements (not full homotopy)
        s = s0
        theta = theta0
        phi = phi0

        for _ in range(6):
            Xc,Yc,Zc,Tc = self._integrate(s, theta, phi, spinTop, spinSide)

            # find bounce approx
            if np.any(Zc<0):
                idx = np.argmax(Zc<0)
            else:
                idx = len(Zc)-1

            # check forward reach
            yf = Yc[idx]
            if abs(yf - (yb - y0)) < 0.1:  # reached target band
                break

            # simple correction
            theta += ( (yb - y0) - yf ) * 0.0005
            s += (dist - (Yc[idx]**2 + Xc[idx]**2)**0.5)*0.01

        # final fences coords
        fencesX = Xc + x0
        fencesY = Yc + y0
        fencesZ = Zc + z0

        bounceIndex = np.argmax(fencesZ<0) if np.any(fencesZ<0) else len(fencesZ)-1

        return {
            "fencesX": fencesX,
            "fencesY": fencesY,
            "fencesZ": fencesZ,
            "t": Tc,
            "bounceIndex": bounceIndex,
            "landingX": fencesX[bounceIndex],
            "landingY": fencesY[bounceIndex],
            "apexHeight": float(np.max(fencesZ[:bounceIndex+1])),
            "spin_top_rpm": spinTopRpm,
            "spin_side_rpm": spinSideRpm,
            "panAngleDeg": math.degrees(math.atan2(dy, dx))
        }