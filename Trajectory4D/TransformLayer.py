# Trajectory4D/TransformLayer.py
import numpy as np
import math


class TransformLayer:
    """
    Transform canonical trajectories into fences coordinates for 4D simulation.

    NEW canonical model supported:
        * Canonical intercept = (0,0,z)
        * Canonical forward axis = +Y
        * BouncePoint in canonical = (0, forwardDistance)
        * Must rotate canonical-forward → fences-forward
        * Must return trajectory3D for Visualizer
    """

    def __init__(self, debug=False):
        self.debug = debug

    # ----------------------------------------------------------------------
    # MAIN APPLY-TRANSFORM METHOD
    # ----------------------------------------------------------------------
    def applyTransform(
        self,
        entry,
        interceptPoint,
        panAngleDeg,
        opponentPosition=None,
        strikeHeightCanonical=0.0,
        opponentSpeed=None,
        bouncePoint=None,        # fences bounce center (for debug)
        bounceCellHalfW=0.50,
    ):
        """
        entry: canonical trajectory entry produced by Trajectory4DGenerator
               (canonical forward = +Y axis)

        interceptPoint: (ix, iy, iz) fences-space intercept
        panAngleDeg: rotation needed to map canonical forward onto fences forward
                     computed in Trajectory4DCanonical: atan2(dx, dy)

        bouncePoint: (bx, by) fences target bounce center (for debugging only)

        Returns dictionary containing:
            - fencesX, fencesY, fencesZ
            - trajectory3D (Nx3)
            - bounceIndex
            - landingX / landingY
            - panAngleDeg
        """

        # landingX = entry["interceptX"]
        # landingY = entry["interceptY"]
        time = entry["time"]

        # ----------------------------------------------------------
        # 1. Extract canonical trajectory arrays
        # ----------------------------------------------------------
        canonX = np.asarray(entry["canonX"], dtype=float)
        canonY = np.asarray(entry["canonY"], dtype=float)
        canonZ = np.asarray(entry["canonZ"], dtype=float)

        # First point in canonical coordinates (should be 0,0,z0)
        canonXOrigin, canonYOrigin, canonZOrigin = canonX[0], canonY[0], canonZ[0]

        # ----------------------------------------------------------
        # 2. Translate canonical so first point is origin
        # ----------------------------------------------------------
        dx = canonX - canonXOrigin
        dy = canonY - canonYOrigin
        dz = canonZ - canonZOrigin

        # ----------------------------------------------------------
        # 3. Rotate canonical (X,Y) to fences frame
        #    panAngleDeg is the rotation needed so canonical +Y → fences-forward
        # ----------------------------------------------------------
        theta = math.radians(panAngleDeg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Local canonical coordinate rotation
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        rz = dz

        # ----------------------------------------------------------
        # 4. Translate into fence coordinates by adding interceptPoint
        # ----------------------------------------------------------
        ix, iy, iz = interceptPoint

        fencesX = ix + rx
        fencesY = iy + ry
        fencesZ = iz + rz

        # ----------------------------------------------------------
        # 5. Bounce-cell debug report
        # ----------------------------------------------------------
        #if self.debug and bouncePoint is not None:
        bounceIndex = entry["bounceIndex"]
        wb_x = fencesX[bounceIndex]
        wb_y = fencesY[bounceIndex]
        bx, by = bouncePoint

        dx_err = wb_x - bx
        dy_err = wb_y - by
        dist_err = math.hypot(dx_err, dy_err)
        inside = (abs(dx_err) <= bounceCellHalfW) and (abs(dy_err) <= bounceCellHalfW)

        # print("\n[DEBUG][TRANSFORM] Bounce‑Cell Check:")
        # print("   Trajectory ID      " + str(entry["id"]))
        # print("   Pan Angle:         " + str(panAngleDeg))
        # print("   Target Apex:       " + str(entry["apex_height"]))
        # print(f"  Target bounce:      ({bx:.3f}, {by:.3f})")
        # print(f"  Transformed bounce: ({wb_x:.3f}, {wb_y:.3f})")
        # print(f"  Error dx, dy:       ({dx_err:.3f}, {dy_err:.3f})")
        # print(f"  Distance error:     {dist_err:.4f} m")
        # print(f"  Cell half-width:    {bounceCellHalfW:.3f} m")
        # print(f"  Inside bounce cell: {inside}")
        # if not inside:
        #     print("  ⚠ WARNING: bounce OUTSIDE target cell!")

        # ----------------------------------------------------------
        # 6. Pack (N,3) matrix for Visualizer
        # ----------------------------------------------------------
        traj3D = np.column_stack((fencesX, fencesY, fencesZ))

        if self.debug:
            bounceIndex = entry["bounceIndex"]
            # print(f"[DEBUG][TRANSFORM] Launch aligned to intercept: {interceptPoint}")
            # print(f"[DEBUG][TRANSFORM] Bounce (fences): ({fencesX[bounceIndex]:.3f}, {fencesY[bounceIndex]:.3f})")
        # print("TRANS TRAJ3D LEN:", len(traj3D), "bounceIndex:", entry["bounceIndex"])

        # ----------------------------------------------------------
        # 7. Return fence defined space trajectory bundle
        # ----------------------------------------------------------
        return {
            "fencesX": fencesX,
            "fencesY": fencesY,
            "fencesZ": fencesZ,
            "time": time,
            "trajectory3D": traj3D,
            "bounceIndex": entry["bounceIndex"],
            "landingX": float(fencesX[-1]),
            "landingY": float(fencesY[-1]),
            "panAngleDeg": panAngleDeg,
            # "usedApexHeight": float(entry["apex_height"]),
            "apexHeight": float(entry["apex_height"]),
            "initialVelocity":entry["initialVelocity"],
            "airTravelDistance":entry["airTravelDistance"],
            "solverIters": 0,
            "landingErrorXY": 0.0,
            "netClearDeficit": 0.0,
            "netClearExcess": 0.0,
        }