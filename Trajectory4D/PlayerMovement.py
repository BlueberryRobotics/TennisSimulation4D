# PlayerMovement.py
import math
import random
import numpy as np
from typing import List, Tuple, Optional


class PlayerMovement:
    """
    Defensive relocation model for the HITTER (after striking the ball).

    World convention assumed everywhere:
      - 0° = +Y (toward PlayerRed), +° = +X (right), -° = -X (left)
      - dx = sin(theta_deg), dy = cos(theta_deg)

    Workflow in chooseDefensivePosition():
      1) Estimate opponent intercept point from the current shot's fences trajectory.
      2) From that intercept, compute the opponent's WORLD return-cone angles (side-aware).
      3) Build a defensive corridor on the hitter's half, aligned to the cone bisector.
      4) Sample discrete corridor points and keep only those reachable by time-to-intercept.
      5) Return a random reachable point (or stay if none).
    """
    def __init__(self, court, lateralWidth=1.5,
                 reachZMin: float = 0.3,           # 10 centimeters
                 reachZMax: float = 3.3,            # 3 meters
                 minCorridorStep: float = 1.0,
                 debug: bool = False):
        self.court = court
        self.playerSpeed = float(court.playerSpeed)
        self.lateralWidth = float(lateralWidth)
        self.reactionTime = float(court.playerReactionTime)
        self.reachZMin = float(reachZMin)     # NEW
        self.reachZMax = float(reachZMax)     # NEW
        self.minCorridorStep = float(minCorridorStep)
        self.debug = bool(debug)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def chooseDefensivePosition(
        self,
        nearestInterceptPoint: Tuple[float,float,float],
        # hitPoint: Tuple[float, float, float],
        currentPosition: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Select a defensive spot for the HITTER (after the shot), within a
        corridor inferred from the opponent's likely return cone.

        Parameters
        ----------
        hitPoint : (x, y, z)
            Hitter's strike point (fences).
        panAngleDeg : float
            Outgoing shot fences pan angle (right-positive).
        trajectory : dict
            'transformed' shot with keys: fencesX, fencesY, fencesZ, time, bounceIndex, etc.
        currentPosition : (x, y)
            Hitter's current fences-space position at strike time (i.e., post-hit start).

        Returns
        -------
        (x, y) : The chosen defensive position (fences).
        """
        (xi, yi, ti) = nearestInterceptPoint

        if yi > 13:
            side_hitter = "North"
        else:
            side_hitter = "South"

        print("DEFENDER SIDE: " + side_hitter + " NearestInterceptPoint: " + str(nearestInterceptPoint))
        
        # If we cannot estimate, fallback to simple midcourt bias corridor.
        if nearestInterceptPoint is None:
            if self.debug:
                print("[DEBUG] DefMove: no opponent intercept found; using current fallback.")
            return random.choice(reachable) if reachable else currentPosition

        # 3) Compute opponent return cone [theta_min, theta_max] in WORLD
        theta_min, theta_max = self.computeOpponentReturnCone((xi, yi), side_hitter)
        print("Theta min/max " + str(theta_min) + ", " + str(theta_max))

        # 4) Build corridor aligned to the bisector of the return cone on the hitter's half
        corridor_points, t_available = self.buildDefensiveCorridorFromCone(
            currentPosition=currentPosition,
            side_hitter=side_hitter,
            theta_min=theta_min,
            theta_max=theta_max,
            t_to_opp=ti,
        )
        print("Corridor Points: " + str(corridor_points))

        # 5) Keep only reachable points by t_available (minus reaction)
        reachable = self.filterReachable(currentPosition, corridor_points, t_available)

        if self.debug:
            print(f"[DEBUG] DefMove: side={side_hitter}, t_avail={t_available:.3f}, "
                  f"corrPts={len(corridor_points)}, reachable={len(reachable)}")

        return random.choice(reachable) if reachable else currentPosition

    # Original reachability filter kept for compatibility,
    # now applying reaction time within the same signature.
    def filterReachableIntercepts(
        self,
        playerPos: Tuple[float, float],
        interceptCandidates: List[Tuple[float, float, float, float]],
    ) -> List[Tuple[float, float, float, float]]:
        """
        Given a list of (x, y, z, t) intercept candidates in fences space,
        return only those the player can physically reach.

        Parameters
        ----------
        playerPos : (px, py)
            Player's current fences-space position.
        interceptCandidates : list of (x, y, z, t)
            Candidate interception points from TransformLayer.
            x, y, z are fences coordinates; t is time since shot launch.

        Returns
        -------
        reachable : list of (x, y, z, t)
        """
        px, py = playerPos
        reachable = []

        for (x, y, z, t) in interceptCandidates:
            if t <= 0:
                continue
            # Effective movement time after reaction delay
            t_move = max(0.0, t - self.reactionTime)
            dist = math.hypot(x - px, y - py)
            maxReach = self.playerSpeed * t_move
            if dist <= maxReach:
                reachable.append((x, y, z, t))

        return reachable

    # ---------------------------------------------------------------------
    # Internals: corridor construction
    # ---------------------------------------------------------------------
    def buildDefensiveCorridorFromCone(
        self,
        currentPosition: Tuple[float, float],
        side_hitter: str,
        theta_min: float,
        theta_max: float,
        t_to_opp: float,
    ) -> Tuple[List[Tuple[float, float]], float]:
        """
        Build a set of candidate defensive points for the hitter from the
        opponent's WORLD return cone.
        """
        # Time available to move = up to opponent contact minus reaction
        t_available = max(0.0, t_to_opp - self.reactionTime)
        d_max = self.playerSpeed * t_available

        # WORLD bisector & axes (unit vectors)
        theta_mid = 0.5 * (theta_min + theta_max)
        ux, uy = self._dir_from_theta(theta_mid)     # axis along corridor
        vx, vy = -uy, ux                             # perpendicular

        # A simple, robust anchor near the hitter, nudged along the bisector
        step_forward = min(d_max * 0.5, 6.0 * self.court.granularity)  # cap nudge to ~6y
        anchor_x = currentPosition[0] + step_forward * ux
        anchor_y = currentPosition[1] + step_forward * uy

        # Axis extent (span we sample along)
        half_len = max(self.court.granularity * 2.0, d_max)  # at least 2y, else up to reach

        # Sample corridor points: along axis every 1m and lateral at {-w, 0, +w}
        pts: List[Tuple[float, float]] = []
        d = -half_len
        while d <= half_len + 1e-9:
            base_x = anchor_x + d * ux
            base_y = anchor_y + d * uy
            for lateral in (-self.lateralWidth, 0.0, self.lateralWidth):
                px = base_x + lateral * vx
                py = base_y + lateral * vy
                # Keep inside hitter's half & singles bounds
                if self._inside_hitter_half(px, py, side_hitter):
                    pts.append((px, py))
            d += self.minCorridorStep

        if self.debug:
            print(f"[DEBUG] DefCorr: side={side_hitter} "
                  f"theta=[{theta_min:.2f},{theta_max:.2f}] mid={theta_mid:.2f} "
                  f"d_max={d_max:.2f} anchor=({anchor_x:.2f},{anchor_y:.2f}) "
                  f"half_len={half_len:.2f} pts={len(pts)}")

        return pts, t_available

    def buildMidcourtCorridor(
        self,
        hitPoint: Tuple[float, float, float],
        panAngleDeg: float,
        trajectory: dict,
        currentPosition: Tuple[float, float],
    ) -> Tuple[List[Tuple[float, float]], float]:
        """
        Fallback when no intercept can be estimated:
        aim corridor toward the geometric mid‑court on hitter's half.
        """
        side_hitter = self._resolve_side(hitPoint[1])
        # Time to bounce as a crude bound (kept for parity with prior logic)
        t_avail = max(0.0, self._time_to_bounce(trajectory) - self.reactionTime)
        d_max = self.playerSpeed * t_avail

        # Midpoint on hitter's half
        cx = self.court.centerLineX
        if side_hitter == "PLAYER_BLUE":
            cy = 0.5 * (self.court.serverBaselineY + self.court.netY)
        else:
            cy = 0.5 * (self.court.netY + self.court.receiverBaselineY)

        # Axis from currentPosition to that mid‑half point
        dx, dy = (cx - currentPosition[0]), (cy - currentPosition[1])
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        vx, vy = -uy, ux

        half_len = max(self.court.granularity * 2.0, d_max)
        pts: List[Tuple[float, float]] = []
        d = -half_len
        while d <= half_len + 1e-9:
            base_x = currentPosition[0] + d * ux
            base_y = currentPosition[1] + d * uy
            for lateral in (-self.lateralWidth, 0.0, self.lateralWidth):
                px = base_x + lateral * vx
                py = base_y + lateral * vy
                if self._inside_hitter_half(px, py, side_hitter):
                    pts.append((px, py))
            d += self.minCorridorStep

        if self.debug:
            print(f"[DEBUG] DefCorr(FB): side={side_hitter} d_max={d_max:.2f} "
                  f"center=({cx:.2f},{cy:.2f}) pts={len(pts)}")

        return pts, t_avail

    def computeOpponentReturnCone(
        self,
        opp_intercept_xy: Tuple[float, float],
        side_hitter: str
    ) -> Tuple[float, float]:
        """
        WORLD return-cone angles [theta_min, theta_max] for the opponent,
        using side-aware targets (near‑net vs deep‑corner), symmetric to your rally logic.
        """
        x0, y0 = opp_intercept_xy
        g = self.court.granularity
        netY = self.court.netY

        # Opponent side is the opposite of the hitter
        side_opp = "PLAYER_RED" if side_hitter == "PLAYER_BLUE" else "PLAYER_BLUE"
        fwdW, rightW = self._player_basis(side_opp)

        # Targets lie on the HITTER's half (opponent aims into hitter's court)
        if side_opp == "PLAYER_RED":
            # Opponent at South hits toward North -> hitter's half y < net
            yNear = netY - g
            yDeep = self.court.serverBaselineY + g
        else:
            # Opponent at North hits toward South -> hitter's half y > net
            yNear = netY + g
            yDeep = self.court.receiverBaselineY - g

        xL_line = self.court.singlesLeftX
        xR_line = self.court.singlesRightX
        xNearLeft, xNearRight = (xL_line + g), (xR_line - g)
        xDeepLeft, xDeepRight = (xL_line + g), (xR_line - g)

        # Side-aware selection based on opponent's intercept x
        eps = 1e-9
        if x0 < xL_line - eps:
            # Outside left (from opponent POV): left target deep-left, right near-right
            tL = (xDeepLeft,  yDeep)
            tR = (xNearRight, yNear)
        elif x0 > xR_line + eps:
            # Outside right: right deep-right, left near-left
            tL = (xNearLeft,  yNear)
            tR = (xDeepRight, yDeep)
        else:
            # Inside: both near-net
            tL = (xNearLeft,  yNear)
            tR = (xNearRight, yNear)

        # LOCAL angles (opponent's frame), then map to WORLD
        aL = self._angle_local(x0, y0, tL[0], tL[1], fwdW, rightW)
        bL = self._angle_local(x0, y0, tR[0], tR[1], fwdW, rightW)
        thL = self._local_to_fences(aL, side_opp)
        thR = self._local_to_fences(bL, side_opp)

        theta_min, theta_max = (thL, thR) if thL <= thR else (thR, thL)

        if self.debug:
            print(f"[DEBUG] RetCone: oppSide={side_opp} oppInt=({x0:.3f},{y0:.3f}) "
                  f"targets L=({tL[0]:.3f},{tL[1]:.3f}) R=({tR[0]:.3f},{tR[1]:.3f}) "
                  f"theta=[{theta_min:.2f},{theta_max:.2f}]")

        return theta_min, theta_max

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def filterReachable(
        self,
        currentPosition: Tuple[float, float],
        candidates: List[Tuple[float, float]],
        t_available: float
    ) -> List[Tuple[float, float]]:
        """Keep positions within reach distance = speed * (t_available)."""
        d_max = self.playerSpeed * max(0.0, t_available)
        px, py = currentPosition
        out = [(x, y) for (x, y) in candidates if math.hypot(x - px, y - py) <= d_max]
        return out

    def _time_to_bounce(self, trajectory: dict) -> float:
        b = int(trajectory.get("bounceIndex", 0))
        t = trajectory["time"]
        return float(t[b]) if 0 <= b < len(t) else float(t[-1])

    def _resolve_side(self, y: float) -> str:
        return "PLAYER_BLUE" if y <= self.court.netY else "PLAYER_RED"

    def _inside_hitter_half(self, x: float, y: float, side_hitter: str) -> bool:
        within_singles = (self.court.singlesLeftX <= x <= self.court.singlesRightX)
        if side_hitter == "PLAYER_BLUE":
            return within_singles and (self.court.serverBaselineY <= y <= self.court.netY)
        else:
            return within_singles and (self.court.netY <= y <= self.court.receiverBaselineY)

    @staticmethod
    def _dir_from_theta(theta_deg: float) -> Tuple[float, float]:
        t = math.radians(theta_deg)
        return math.sin(t), math.cos(t)

    @staticmethod
    def _player_basis(side: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        # forward always points toward opponent half
        if side == "PLAYER_BLUE":
            return (0.0, +1.0), (+1.0, 0.0)  # forward=+Y, right=+X
        else:
            return (0.0, -1.0), (-1.0, 0.0)  # forward=-Y, right=-X

    @staticmethod
    def _angle_local(x0, y0, xt, yt, forwardW, rightW) -> float:
        dx = xt - x0
        dy = yt - y0
        dotF = dx * forwardW[0] + dy * forwardW[1]
        dotR = dx * rightW[0]   + dy * rightW[1]
        return math.degrees(math.atan2(dotR, dotF))

    @staticmethod
    def _local_to_fences(theta_local: float, side: str) -> float:
        # PLAYER_BLUE: θ_fences = θ_local; PLAYER_RED: θ_fences = θ_local + 180°
        th = theta_local if side == "PLAYER_BLUE" else (theta_local + 180.0)
        # normalize to (-180, 180]
        th = (th + 180.0) % 360.0 - 180.0
        return 180.0 if th == -180.0 else th