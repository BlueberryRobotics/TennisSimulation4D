# Trajectory4D/Trajectory4DGenerator.py
import json
import numpy as np
from typing import Tuple, Dict, Any, List, Optional

# print("[VER] Trajectory4DGenerator: apex error-driven + Broyden 2x2 + two-seed, "
#       "backtracking, near-net continuation; 1 sim/iter after seed; single source of truth")

class Trajectory4DDynamicGenerator:
    """
    3D ballistics with drag + Magnus + bounce.

    Apex-driven, fast error solver:
      • FIRST bounce within ±landing_tol of target (xb,yb),
      • z(net) ≥ netHeight (within net_eps),
      • optional upper cap: z(net) ≤ netHeight + maxNetClearAbove (near-net rule, with continuation).

    Public (name preserved to avoid runner changes):
      generate_by_apex_ladder(
          launchPoint=(x0,y0,z0),
          bounceXY=(xb,yb),
          apexHeights: List[float],          # discrete apex domain
          spinTopRpm, spinSideRpm,          # fixed spins (discrete)
          maxNetClearAbove: Optional[float],# None → no cap; else cap over tape (m)
          landing_tol=0.10,                 # ±10 cm landing tolerance
          net_eps=5e-3,                     # ~5 mm epsilon on net-plane tests
          maxItersPerApex=600               # iteration budget total (split across seeds)
      ) -> dict
      On failure: raises RuntimeError("SOLVER_NO_CONVERGENCE::<json>")
    """

    def __init__(self, court):
        self.court = court

        # Ball/air
        self.ballMass   = 0.057
        self.ballRadius = 0.0335
        self.airDensity = 1.225
        self.dragCoefficient = 0.55
        self.crossSectionArea = np.pi * self.ballRadius**2
        self.gravity = 9.81

        # Drag a = -k|v|v
        self.dragFactor = 0.5 * self.airDensity * self.dragCoefficient * self.crossSectionArea / self.ballMass

        # Magnus (3D): a_M = k (ω × v)
        self.magnusK = 1.25e-4

        # Integration (adaptive dt; near-ground finer)
        self.dt_far   = 0.02   # high in air
        self.dt_near  = 0.006  # near ground
        self.z_switch = 1.0    # switch height
        self.maxTime  = 3.5    # rallies/serves fit in this horizon

        # Bounce (tennis-like)
        self.restitution  = 0.70
        self.mu_t         = 0.06
        self.spinToTan    = 5.0e-4
        self.hystRiseZ    = 0.01
        self.minVzRebound = 0.20

    # ---------- spin / dynamics ----------
    @staticmethod
    def _rpm_to_rad_s(rpm: float) -> float:
        return (2.0 * np.pi * rpm) / 60.0

    def _compose_spin(self, spinTopRpm: float, spinSideRpm: float) -> np.ndarray:
        # top/backspin about X (right+), sidespin about Z (up)
        ex = np.array([1.0, 0.0, 0.0])
        ez = np.array([0.0, 0.0, 1.0])
        return self._rpm_to_rad_s(spinTopRpm) * ex + self._rpm_to_rad_s(spinSideRpm) * ez

    def _accel(self, v: np.ndarray, omega: np.ndarray) -> np.ndarray:
        s = np.linalg.norm(v) + 1e-12
        a_drag   = -self.dragFactor * s * v
        a_magnus = self.magnusK * np.cross(omega, v)
        a_grav   = np.array([0.0, 0.0, -self.gravity])
        return a_drag + a_magnus + a_grav

    def _step(self, p: np.ndarray, v: np.ndarray, omega: np.ndarray, dt: float):
        a  = self._accel(v, omega)
        v2 = v + a * dt
        p2 = p + v2 * dt
        return p2, v2

    # ---------- in-step impact ----------
    def _resolve_impact(self, p: np.ndarray, v: np.ndarray, p2: np.ndarray, v2: np.ndarray, omega: np.ndarray, dt: float):
        z0, z1 = p[2], p2[2]
        if z0 <= 0.0 and z1 <= 0.0:
            tau   = 0.0
            v_imp = v.copy()
            p_imp = p.copy(); p_imp[2] = 0.0
        else:
            tau = float(np.clip(dt * z0 / (z0 - z1 + 1e-12), 0.0, dt))
            v_imp = v + (v2 - v) * (tau / dt)
            p_imp = p + v_imp * tau
            p_imp[2] = 0.0

        p_out, v_out, rebounded = self._apply_bounce(p_imp, v_imp, omega)
        return p_imp, p_out, v_out, rebounded

    def _apply_bounce(self, p_imp: np.ndarray, v_imp: np.ndarray, omega: np.ndarray):
        x, y, _ = p_imp
        vx, vy, vz = v_imp
        if vz >= 0.0:
            return np.array([x, y, 0.0]), np.array([vx, vy, 0.0]), False

        vz_out = -self.restitution * vz
        Jn = (1.0 + self.restitution) * (-vz)

        def t_adj(vt):
            if abs(vt) < 1e-9: return 0.0
            return vt - self.mu_t * Jn * np.sign(vt)

        vx_out = t_adj(vx)
        vy_out = t_adj(vy) + self.spinToTan * omega[2]

        if vz_out < self.minVzRebound:
            return np.array([x, y, 0.0]), np.array([vx_out, vy_out, 0.0]), False

        return np.array([x, y, 0.0]), np.array([vx_out, vy_out, vz_out]), True

    # ---------- single simulation (adaptive dt; stops after first bounce/settling) ----------
    def _simulate_once(self, p0: np.ndarray, v0: np.ndarray, omega: np.ndarray) -> Dict[str, Any]:
        Xs, Ys, Zs, Ts = [], [], [], []
        p, v = p0.copy(), v0.copy()
        t = 0.0
        bounceIndex = None
        bounceCount = 0
        mustRise = False
        steps = 0
        maxSteps = int(self.maxTime / self.dt_near) + 4  # conservative

        while steps < maxSteps and t <= self.maxTime:
            dt = self.dt_near if p[2] <= self.z_switch else self.dt_far
            p2, v2 = self._step(p, v, omega, dt)
            t2 = t + dt

            if (p[2] > 0.0 and p2[2] <= 0.0) and (not mustRise):
                p_imp, p_out, v_out, rebounded = self._resolve_impact(p, v, p2, v2, omega, dt)
                Xs.append(p_imp[0]); Ys.append(p_imp[1]); Zs.append(0.0); Ts.append(t2)
                if bounceIndex is None:
                    bounceIndex = len(Xs) - 1
                bounceCount += 1
                if (bounceCount >= 2) or (not rebounded):
                    break
                p, v = p_out, v_out
                mustRise = True
                t = t2
                steps += 1
                continue

            p, v, t = p2, v2, t2
            Xs.append(p[0]); Ys.append(p[1]); Zs.append(max(0.0, p[2])); Ts.append(t)

            if mustRise and p[2] >= self.hystRiseZ:
                mustRise = False

            # Early stop if post-bounce speed tiny near ground
            if (bounceIndex is not None) and (p[2] < 0.05) and (np.linalg.norm(v) < 1.0):
                break

            steps += 1

        Xs = np.asarray(Xs); Ys = np.asarray(Ys); Zs = np.asarray(Zs); Ts = np.asarray(Ts)
        return {
            "fencesX": Xs, "fencesY": Ys, "fencesZ": Zs, "time": Ts,
            "trajectory3D": np.column_stack([Xs, Ys, Zs]) if len(Xs) else np.zeros((0,3)),
            "bounceIndex": bounceIndex,
            "landingX": float(Xs[-1]) if len(Xs) else float(p0[0]),
            "landingY": float(Ys[-1]) if len(Ys) else float(p0[1])
        }

    # ---------- apex error-driven + Broyden 2x2 with two-seed and backtracking ----------
    def generate_by_apex_ladder(self,
                                interceptPoint: Tuple[float, float, float],
                                bouncePoint: Tuple[float, float],
                                apexHeight: float,
                                apexValues: List[float],
                                spinTopRpm: float,
                                spinSideRpm: float,
                                maxNetClearAbove: Optional[float] = None,
                                landing_tol: float = 0.10,
                                net_eps: float = 5e-3,
                                maxItersPerApex: int = 600) -> Dict[str, Any]:
        """
        Error-driven solver with one discrete apex target:
          • two small seeds (θ,s) tried sequentially (split iteration budget)
          • (θ,s) updated by Broyden 2x2 (LM damping) on first-bounce (x,y)
          • backtracking line-search on (θ,s) if a trial step worsens landing error
          • pitch enforces net lower/cap; softly nudges toward target H
          • near-net cap applied via continuation (only strict when e_xy is small)
        """
        x0, y0, z0 = map(float, interceptPoint)
        xb, yb     = map(float, bouncePoint)
        p0 = np.array([x0, y0, z0], dtype=float)
        omega = self._compose_spin(spinTopRpm, spinSideRpm)

        # ----- choose ONE apex target from discrete list -----
        eps = 1e-6
        if z0 < self.court.netHeight - eps:
            candidates = [float(H) for H in apexValues if H > z0 + eps]
        else:
            candidates = [float(H) for H in apexValues if H >= z0 - eps]
        if not candidates:
            raise RuntimeError("SOLVER_NO_CONVERGENCE::" + json.dumps({
                "reason": "NO_APEX_CANDIDATES",
                "launchPoint": (x0, y0, z0),
                "bounceXY": (xb, yb),
                "spinTopRpm": float(spinTopRpm),
                "spinSideRpm": float(spinSideRpm)
            }))

        # heuristic apex guess grows mildly with horizontal distance
        dist_xy = max(0.5, np.hypot(xb - x0, yb - y0))
        h_guess = max(self.court.netHeight + 0.05, z0 + 0.30, z0 + 0.15 * dist_xy)
        H_target = min(candidates, key=lambda H: abs(H - h_guess))

        def compose_velocity(theta_deg: float, speed: float, pitch_deg: float) -> np.ndarray:
            th = np.radians(theta_deg)
            ph = np.radians(pitch_deg)
            sxy = speed * np.cos(ph)
            return np.array([sxy * np.sin(th), sxy * np.cos(th), speed * np.sin(ph)], dtype=float)

        def initial_pitch_from(H: float, z_launch: float, speed: float) -> float:
            dz = max(0.0, H - z_launch)
            vz0 = float(np.sqrt(2.0 * self.gravity * dz)) if dz > 0 else 0.0
            return float(np.degrees(np.arcsin(np.clip(vz0 / max(1.0, speed), -0.99, 0.99))))

        # Pitch gains and caps
        k_pitch_net  = 3.50
        k_pitch_apex = 2.75
        speed_min, speed_max = 8.0, 45.0
        pitch_min, pitch_max = -25.0, 35.0
        netY = float(self.court.netY)

        # Two tiny seeds for (theta, s0)
        theta0 = float(np.degrees(np.arctan2((xb - x0), (yb - y0))))
        s0_0   = 22.0 if dist_xy < 16.0 else 26.0
        pitch0 = initial_pitch_from(H_target, z0, s0_0)

        theta2 = theta0 + (3.0 if (xb - x0) >= 0 else -3.0)
        s0_2   = float(np.clip(s0_0 + (2.0 if (yb - y0) >= 0 else -2.0), speed_min, speed_max))
        pitch2 = initial_pitch_from(H_target, z0, s0_2)

        seeds = [(theta0, s0_0, pitch0), (theta2, s0_2, pitch2)]
        per_seed_budget = max(150, int(maxItersPerApex // len(seeds)))

        def _solve_once(theta: float, s0: float, pitch: float, maxIters: int) -> Dict[str, Any]:
            # --- one-time baseline + probes to seed J ---
            v0 = compose_velocity(theta, s0, pitch)
            sim0 = self._simulate_once(p0, v0, omega)
            b0 = sim0["bounceIndex"]
            tries = 0
            while (b0 is None or b0 <= 0) and tries < 6:
                s0 = min(speed_max, s0 * 1.03)
                pitch = min(pitch_max, pitch + 1.0)
                v0 = compose_velocity(theta, s0, pitch)
                sim0 = self._simulate_once(p0, v0, omega)
                b0 = sim0["bounceIndex"]
                tries += 1
            if b0 is None or b0 <= 0:
                raise RuntimeError("NO_BASELINE_BOUNCE")

            xB0 = float(sim0["fencesX"][b0]); yB0 = float(sim0["fencesY"][b0])
            e_prev = np.array([xb - xB0, yb - yB0], dtype=float)
            f_prev = np.array([xB0, yB0], dtype=float)  # landing point

            # Probes for J seed
            dtheta_probe_deg = 0.75
            ds_probe         = 0.50

            v_t = compose_velocity(theta + dtheta_probe_deg, s0, pitch)
            sim_t = self._simulate_once(p0, v_t, omega)
            bt = sim_t["bounceIndex"]
            if bt is None or bt <= 0:
                dx_dtheta = dy_dtheta = 0.0
            else:
                xBt = float(sim_t["fencesX"][bt]); yBt = float(sim_t["fencesY"][bt])
                dx_dtheta = (xBt - xB0) / dtheta_probe_deg
                dy_dtheta = (yBt - yB0) / dtheta_probe_deg

            v_s = compose_velocity(theta, s0 + ds_probe, pitch)
            sim_s = self._simulate_once(p0, v_s, omega)
            bs = sim_s["bounceIndex"]
            if bs is None or bs <= 0:
                dx_ds = dy_ds = 0.0
            else:
                xBs = float(sim_s["fencesX"][bs]); yBs = float(sim_s["fencesY"][bs])
                dx_ds = (xBs - xB0) / ds_probe
                dy_ds = (yBs - yB0) / ds_probe

            J = np.array([[dx_dtheta, dx_ds],
                          [dy_dtheta, dy_ds]], dtype=float) + 1e-6 * np.eye(2)

            # Damping & step caps
            lam = 0.10
            dtheta_cap_deg_base = 1.5
            ds_cap_base         = 1.0
            dpitch_cap_deg      = 1.0

            for it in range(maxIters):
                # LM step Δp = (J^T J + λI)^-1 J^T e
                JTJ = J.T @ J
                rhs = J.T @ e_prev
                try:
                    step = np.linalg.solve(JTJ + lam * np.eye(2), rhs)
                except np.linalg.LinAlgError:
                    base_dist = max(5.0, dist_xy)
                    step = np.array([2.5 * (e_prev[0] / base_dist), 0.18 * e_prev[1]], dtype=float)

                # Adaptive caps (bigger moves when far)
                dtheta_cap_deg = dtheta_cap_deg_base + min(1.5, abs(e_prev[0]) * 1.0)  # up to ~3.0°
                ds_cap         = ds_cap_base         + min(0.5,  abs(e_prev[1]) * 0.5) # up to ~1.5 m/s
                dtheta_cmd     = float(np.clip(step[0], -dtheta_cap_deg, dtheta_cap_deg))
                ds_cmd         = float(np.clip(step[1], -ds_cap,         ds_cap))

                # --- backtracking line-search on (θ,s) ---
                best = None
                for scale in (1.0, 0.5, 0.25):
                    th_try = theta + scale * dtheta_cmd
                    s_try  = float(np.clip(s0 + scale * ds_cmd, speed_min, speed_max))
                    v_try  = compose_velocity(th_try, s_try, pitch)
                    sim    = self._simulate_once(p0, v_try, omega)
                    b      = sim["bounceIndex"]
                    if b is None or b <= 0:
                        continue
                    xB = float(sim["fencesX"][b]); yB = float(sim["fencesY"][b])
                    e_xy_try = float(np.hypot(xb - xB, yb - yB))
                    if (best is None) or (e_xy_try < best[0]):
                        best = (e_xy_try, th_try, s_try, sim, xB, yB)
                if best is None:
                    lam = min(10.0, lam * 2.0)
                    continue

                e_xy, theta_new, s0_new, sim, xB_new, yB_new = best
                e = np.array([xb - xB_new, yb - yB_new], dtype=float)

                # Net-plane & apex for pitch control
                Ys = sim["fencesY"]; Zs = sim["fencesZ"]
                idxNet = int(np.argmin(np.abs(Ys - netY)))
                zNet = float(Zs[idxNet])

                # near-net cap continuation: only activate strongly once close in XY
                if maxNetClearAbove is None:
                    cap_ok = True
                    cap_active = False
                    cap_thresh = None
                else:
                    if e_xy > 0.25:  # far from target → ignore cap (just clear tape)
                        cap_ok = True
                        cap_active = False
                        cap_thresh = None
                    else:
                        alpha = float(np.clip((0.25 - e_xy) / 0.25, 0.0, 1.0))  # 0→1 as we get close
                        cap_thresh = self.court.netHeight + alpha * maxNetClearAbove
                        cap_ok = (zNet <= cap_thresh + net_eps)
                        cap_active = True

                clear_ok = (zNet >= self.court.netHeight - net_eps)

                # Apex (up to first bounce) — soft target when net constraints OK
                z_apex = float(np.max(Zs[: (sim["bounceIndex"] or 0) + 1]))
                e_apex = H_target - z_apex

                # Convergence
                if (e_xy <= landing_tol) and clear_ok and cap_ok:
                    out = sim.copy()
                    out["panAngleDeg"]     = float(theta_new)
                    out["solverIters"]     = it + 1
                    out["landingErrorXY"]  = e_xy
                    out["netClearDeficit"] = float(max(0.0, self.court.netHeight - zNet))
                    # if cap inactive, excess is 0
                    if maxNetClearAbove is None or not cap_active:
                        out["netClearExcess"] = 0.0
                    else:
                        out["netClearExcess"] = float(max(0.0, zNet - (cap_thresh or (self.court.netHeight + maxNetClearAbove))))
                    out["usedApexHeight"]  = float(H_target)
                    return out

                # Broyden update: J_{k+1} = J_k + ((Δf - J_k Δp) Δp^T) / (Δp^T Δp)
                f_prev = np.array([xB0, yB0], dtype=float)
                f_cur  = np.array([xB_new, yB_new], dtype=float)
                df     = f_cur - f_prev
                dp     = np.array([theta_new - theta, s0_new - s0], dtype=float)
                denom  = float(dp @ dp) + 1e-12
                J = J + np.outer((df - J @ dp), dp) / denom

                # Accept new (θ,s); keep pitch focused on net/cap, else apex
                theta, s0 = theta_new, s0_new
                xB0, yB0 = xB_new, yB_new
                e_prev   = e.copy()

                if not clear_ok:
                    d_pitch = k_pitch_net * (self.court.netHeight - zNet + net_eps)
                elif cap_active and (not cap_ok):
                    d_pitch = -k_pitch_net * (zNet - (cap_thresh or (self.court.netHeight + maxNetClearAbove)) + net_eps)
                else:
                    d_pitch = k_pitch_apex * np.clip(e_apex, -0.5, 0.5)
                pitch = float(np.clip(pitch + float(np.clip(d_pitch, -1.0, 1.0)), pitch_min, pitch_max))

                # Damping schedule
                lam = min(10.0, lam * 1.5) if e_xy > np.linalg.norm(e_prev) + 1e-4 else max(0.02, lam * 0.7)

            # Exhausted this seed
            raise RuntimeError("SEED_NO_CONVERGENCE")

        # Try both seeds (each with half the budget). Return first success.
        for (th, s, ph) in seeds:
            try:
                return _solve_once(th, s, ph, per_seed_budget)
            except RuntimeError:
                continue

        # Failure across seeds → raise payload
        failure_payload = {
            "reason": "APEX_SOLVE_NO_CONVERGENCE",
            "launchPoint": (x0, y0, z0),
            "bounceXY": (xb, yb),
            "targetApex": float(H_target),
            "spinTopRpm": float(spinTopRpm),
            "spinSideRpm": float(spinSideRpm),
            "maxNetClearAbove": float(maxNetClearAbove) if maxNetClearAbove is not None else None,
            "landing_tol": float(landing_tol),
            "net_eps": float(net_eps),
            "maxItersPerApex": int(maxItersPerApex)
        }
        raise RuntimeError("SOLVER_NO_CONVERGENCE::" + json.dumps(failure_payload))