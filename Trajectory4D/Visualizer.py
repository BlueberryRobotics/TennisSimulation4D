import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import numpy as np
import math
import os
import re
import base64
from io import BytesIO
from typing import Optional, Tuple, Dict, Any


# ---------------------------
# Utilities / helpers
# ---------------------------

# def TruncateAtGround(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
#     """
#     Return xs, ys, zs truncated at the first index where z < 0 (inclusive).
#     """
#     neg = np.where(zs < 0.0)[0]
#     if neg.size == 0:
#         return xs, ys, zs
#     cut = int(neg[0])
#     return xs[:cut + 1], ys[:cut + 1], zs[:cut + 1]


def DrawCourtLines(ax, court):
    """
    Plan view (X vs Y) singles lines:
      - Sidelines & baselines
      - Service lines (both halves)
      - Center service line
      - Net (dashed, no label)
    """
    colorLine = "#666"
    colorNet = "#000"
    lwMain = 1.6
    lwAux = 1.2

    courtLeftLine = court.singlesLeftX
    courtRightLine = court.singlesRightX
    yNorthBase = court.serverBaselineY
    ySouthBase = court.receiverBaselineY

    # Sidelines
    ax.plot([courtLeftLine, courtLeftLine], [yNorthBase, ySouthBase], color=colorLine, lw=lwMain)
    ax.plot([courtRightLine, courtRightLine], [yNorthBase, ySouthBase], color=colorLine, lw=lwMain)
    # Baselines
    ax.plot([courtLeftLine, courtRightLine], [yNorthBase, yNorthBase], color=colorLine, lw=lwMain)
    ax.plot([courtLeftLine, courtRightLine], [ySouthBase, ySouthBase], color=colorLine, lw=lwMain)

    # Service lines
    ax.plot([courtLeftLine, courtRightLine], [court.serviceLineY, court.serviceLineY], color=colorLine, lw=lwAux)
    ax.plot([courtLeftLine, courtRightLine], [court.opponentServiceLineY, court.opponentServiceLineY], color=colorLine, lw=lwAux)

    # Center service line (north + south halves)
    cx = court.centerLineX
    ax.plot([cx, cx], [court.serviceLineY, court.netY], color=colorLine, lw=lwAux)
    ax.plot([cx, cx], [court.netY, court.opponentServiceLineY], color=colorLine, lw=lwAux)

    # Net (dashed, no label)
    ax.axhline(court.netY, color=colorNet, linestyle="--", linewidth=1.1)


def DrawPlanGridOverlay(ax, court):
    g = float(court.granularity)
    xMin = float(court.xMin)
    xMax = float(court.xMax)
    yMin = float(court.yMin)
    yMax = float(court.yMax)
    cols = int(court.gridColumns)
    rows = int(court.gridRows)

    for col in range(cols + 1):
        x = xMin + (col * g)
        ax.plot([x, x], [yMin, yMax], color="#A9A9A9", linestyle="--", linewidth=0.7, alpha=0.55, zorder=0)

    centerBoundaryRow = rows // 2
    netYValue = float(getattr(court, "netY", yMin + (centerBoundaryRow * g)))
    for row in range(rows + 1):
        if hasattr(court, "RowBoundaryY"):
            y = float(court.RowBoundaryY(row))
        else:
            y = yMin + (row * g)

        if abs(y - netYValue) <= 1e-6:
            ax.plot([xMin, xMax], [y, y], color="#333", linestyle="--", linewidth=1.3, alpha=0.95, zorder=1)
        else:
            ax.plot([xMin, xMax], [y, y], color="#A9A9A9", linestyle="--", linewidth=0.7, alpha=0.55, zorder=0)

    for row in range(1, rows + 1):
        if hasattr(court, "GetRowCenterY"):
            yCenter = float(court.GetRowCenterY(row))
        else:
            yCenter = yMin + ((row - 0.5) * g)
        ax.text(
            xMin - (0.28 * g),
            yCenter,
            str(row),
            horizontalalignment="right",
            verticalalignment="center",
            fontsize=6,
            color="#444",
            clip_on=False,
            zorder=20,
        )

    for col in range(1, cols + 1):
        xCenter = xMin + ((col - 0.5) * g)
        ax.text(
            xCenter,
            yMin - (0.33 * g),
            str(col),
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=7,
            color="#444",
            clip_on=False,
            zorder=20,
        )

    centerX = xMin + ((xMax - xMin) / 2.0)
    ax.text(
        centerX,
        yMin - (0.8 * g),
        "Player",
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=9,
        color="#222",
        fontweight="bold",
        clip_on=False,
        zorder=21,
    )
    ax.text(
        centerX,
        yMax + (0.35 * g),
        "Opponent",
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=9,
        color="#222",
        fontweight="bold",
        clip_on=False,
        zorder=21,
    )


def _ContiguousSegments(indices: np.ndarray):
    """
    Given sorted indices, return list of (startIndex, endIndexInclusive) contiguous runs.
    """
    if indices.size == 0:
        return []
    runs = []
    start = indices[0]
    prev = indices[0]
    for k in indices[1:]:
        if k == prev + 1:
            prev = k
            continue
        runs.append((start, prev))
        start = k
        prev = k
    runs.append((start, prev))
    return runs


def _AddColoredSegments(ax, yVals: np.ndarray, zVals: np.ndarray,
                        indices: np.ndarray, color: str, linewidth: float = 2.2, alpha: float = 1.0):
    """
    On side view (Z vs Y), draw colored segments over the base trajectory for given indices.
    """
    runs = _ContiguousSegments(indices)
    segs = []
    for s, e in runs:
        if e <= s:
            continue
        ySeg = yVals[s:e+1]
        zSeg = zVals[s:e+1]
        pts = np.column_stack([ySeg, zSeg])
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        segs.append(segments)
    if not segs:
        return
    segs = np.concatenate(segs, axis=0)
    lc = LineCollection(segs, colors=color, linewidths=linewidth, alpha=alpha, zorder=3)
    ax.add_collection(lc)


def _ComputeInterceptRadius(transformed, court, defenderSide, defenderPos,
                            playerSpeed: float, reactionTime: float = 0.0, turnCost: float = 0.0,
                            maxIndexInclusive: Optional[int] = None):
    """
    ToF-based interception radius (per your spec):
      1) find nearest (XY) sample on defender's half,
      2) ToF_perp = time to that sample,
      3) R = max(0, ToF_perp - reactionTime - turnCost) * playerSpeed.

    Returns (R, tof_perp, idx_nearest) or (0.0, 0.0, None) if not computable.
    """
    try:
        X = transformed["fencesX"]; Y = transformed["fencesY"]; T = transformed["time"]
        if maxIndexInclusive is not None:
            end = int(maxIndexInclusive) + 1
            X = X[:end]
            Y = Y[:end]
            T = T[:end]

        if len(X) == 0:
            return 0.0, 0.0, None

        px, py = float(defenderPos[0]), float(defenderPos[1])
        netY = float(court.netY)

        # Opponent-half mask for the defender
        half_mask = (Y <= netY) if defenderSide == "PLAYER_BLUE" else (Y >= netY)
        idx_half = np.where(half_mask)[0]
        if idx_half.size == 0:
            return 0.0, 0.0, None

        d = np.hypot(X[idx_half] - px, Y[idx_half] - py)
        k_rel = int(np.argmin(d))
        k = int(idx_half[k_rel])
        tof_perp = float(T[k])

        slack = max(0.0, tof_perp - float(reactionTime) - float(turnCost))
        R = float(playerSpeed) * slack
        return R, tof_perp, k
    except Exception:
        return 0.0, 0.0, None


def _DrawInterceptCircle(ax, centerXY, radiusM, color="#8A2BE2", alpha=0.25, lw=2.0, label="Intercept radius"):
    """
    Draw the interception footprint circle around the defender (plan view).
    """
    if radiusM <= 0:
        return
    cx, cy = centerXY
    fill = Circle((cx, cy), radiusM, facecolor=color, edgecolor="none", alpha=alpha * 0.25, zorder=1)
    line = Circle((cx, cy), radiusM, facecolor="none", edgecolor=color, linewidth=lw, linestyle="--", zorder=2)
    ax.add_patch(fill)
    ax.add_patch(line)
    line.set_label(label)


def _FindInPlayEndIndex(zVals: np.ndarray, bounceIdx: int, epsilon: float = 1e-6) -> int:
    """
    Return the last index where the shot is considered in play.

    In-play end is the first post-bounce ground contact (z <= 0)
    after the post-bounce apex.
    """
    n = int(len(zVals))
    if n == 0:
        return -1

    b = max(0, min(int(bounceIdx), n - 1))
    post = zVals[b:]
    if len(post) <= 1:
        return n - 1

    apexRel = int(np.argmax(post))
    apexIdx = b + apexRel
    if apexIdx >= n - 1:
        return n - 1

    secondGround = np.where(zVals[apexIdx + 1:] <= float(epsilon))[0]
    if secondGround.size == 0:
        return n - 1

    return int(apexIdx + 1 + secondGround[0])


def _SafeFloat(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _Fmt(value, decimals: int = 6):
    """
    Format floats for panel display while avoiding '-0.000000' artifacts.
    """
    v = _SafeFloat(value)
    if v is None:
        return "n/a"
    epsilon = 0.5 * (10.0 ** (-int(decimals)))
    if abs(v) < epsilon:
        v = 0.0
    return f"{v:.{decimals}f}"


def _ComputeExecutionErrorFactors(shot, court):
    """
    Compute error terms shown in the visualization panel.
    Mirrors the same components used by ExecutionProbability.
    """
    if shot is None or court is None:
        return {"available": False, "reason": "missing shot or court"}

    trackedExecutionFactors = shot.get("shotExecutionFactors")
    if not isinstance(trackedExecutionFactors, dict):
        trackedExecutionFactors = shot.get("nextHitterExecutionFactors")
    trackedLuckCostIncrement = _SafeFloat(shot.get("currentHitterLuckCostIncrement"))
    if trackedLuckCostIncrement is None:
        trackedLuckCostIncrement = _SafeFloat(shot.get("nextHitterLuckCostIncrement"))
    trackedLuckCostCumulative = _SafeFloat(shot.get("currentHitterLuckCostCumulative"))
    if trackedLuckCostCumulative is None:
        trackedLuckCostCumulative = _SafeFloat(shot.get("nextHitterLuckCostCumulative"))
    trackedExecutionErrorIncrement = _SafeFloat(shot.get("nextHitterExecutionErrorIncrement"))

    if isinstance(trackedExecutionFactors, dict):
        incomingVelocityError = _SafeFloat(trackedExecutionFactors.get("incomingVelocityError"))
        outgoingVelocityError = _SafeFloat(trackedExecutionFactors.get("outgoingVelocityError"))
        proximityToNetError = _SafeFloat(trackedExecutionFactors.get("proximityToNetError"))
        proximityToBoundaryError = _SafeFloat(trackedExecutionFactors.get("proximityToBoundaryError"))
        proximityToNetLuckWeight = _SafeFloat(
            trackedExecutionFactors.get("proximityToNetLuckWeight")
        )
        proximityToBoundaryLuckWeight = _SafeFloat(
            trackedExecutionFactors.get("proximityToBoundaryLuckWeight")
        )
        if proximityToNetLuckWeight is None:
            proximityToNetLuckWeight = _SafeFloat(
                getattr(court, "proximityToNetLuckWeight", 1.0)
            )
        if proximityToBoundaryLuckWeight is None:
            proximityToBoundaryLuckWeight = _SafeFloat(
                getattr(court, "proximityToBoundaryLuckWeight", 1.0)
            )

        weightedNetClearanceError = _SafeFloat(
            trackedExecutionFactors.get("weightedNetClearanceError")
        )
        weightedInBoundsError = _SafeFloat(
            trackedExecutionFactors.get("weightedInBoundsError")
        )

        if weightedNetClearanceError is None and proximityToNetError is not None and proximityToNetLuckWeight is not None:
            weightedNetClearanceError = float(proximityToNetLuckWeight) * float(proximityToNetError)
        if weightedInBoundsError is None and proximityToBoundaryError is not None and proximityToBoundaryLuckWeight is not None:
            weightedInBoundsError = float(proximityToBoundaryLuckWeight) * float(proximityToBoundaryError)

        shotLuckCost = None
        if weightedNetClearanceError is not None and weightedInBoundsError is not None:
            shotLuckCost = (
                float(weightedNetClearanceError)
                + float(weightedInBoundsError)
            )

        if (
            incomingVelocityError is not None
            and outgoingVelocityError is not None
            and proximityToNetError is not None
            and proximityToNetLuckWeight is not None
            and proximityToBoundaryError is not None
            and proximityToBoundaryLuckWeight is not None
            and weightedNetClearanceError is not None
            and weightedInBoundsError is not None
            and shotLuckCost is not None
        ):
            return {
                "available": True,
                "incomingVelocityAttribution": float(incomingVelocityError),
                "outgoingVelocityAttribution": float(outgoingVelocityError),
                "incomingVelocityError": float(incomingVelocityError),
                "outgoingVelocityError": float(outgoingVelocityError),
                "proximityToNetError": float(proximityToNetError),
                "proximityToBoundaryError": float(proximityToBoundaryError),
                "proximityToNetLuckWeight": float(proximityToNetLuckWeight),
                "proximityToBoundaryLuckWeight": float(proximityToBoundaryLuckWeight),
                "weightedNetClearanceError": float(weightedNetClearanceError),
                "weightedInBoundsError": float(weightedInBoundsError),
                "shotLuckCost": float(shotLuckCost),
                "trackedLuckCostIncrement": trackedLuckCostIncrement,
                "accumulatedLuckCost": trackedLuckCostCumulative,
            }

    initialVelocity = _SafeFloat(shot.get("initialVelocity"))
    airTravelDistance = _SafeFloat(shot.get("airTravelDistance"))
    bounceX = _SafeFloat(shot.get("bounceX"))
    bounceY = _SafeFloat(shot.get("bounceY"))
    intendedNetClearance = _SafeFloat(shot.get("netClearance"))
    incomingInterceptVelocity = _SafeFloat(shot.get("incomingInterceptVelocity"))
    shotType = str(shot.get("type", "")).upper()
    serveSide = str(shot.get("serveSide", "")).upper()

    maxIncomingVelocity = _SafeFloat(getattr(court, "maxIncomingVelocity", 58.1152))
    courtLength = _SafeFloat(court.receiverBaselineY - court.serverBaselineY)

    outgoingVelocityErrorGain = _SafeFloat(
        getattr(court, "outgoingVelocityErrorGain", None)
    )
    incomingVelocityErrorGain = _SafeFloat(
        getattr(court, "incomingVelocityErrorGain", None)
    )

    if outgoingVelocityErrorGain is None or incomingVelocityErrorGain is None:
        return {
            "available": False,
            "reason": "missing outgoingVelocityErrorGain/incomingVelocityErrorGain in court settings",
        }

    if (
        initialVelocity is None
        or airTravelDistance is None
        or bounceX is None
        or bounceY is None
        or intendedNetClearance is None
    ):
        return {
            "available": False,
            "reason": "missing one or more required fields: initialVelocity, airTravelDistance, bounceX, bounceY, netClearance",
        }

    velocityScale = maxIncomingVelocity if maxIncomingVelocity is not None and maxIncomingVelocity > 0.0 else 1.0

    velocityDeviation = max(0.0, float(initialVelocity)) / float(velocityScale)

    incomingVelocityRatio = 0.0
    if incomingInterceptVelocity is not None and maxIncomingVelocity is not None and maxIncomingVelocity > 0.0:
        incomingVelocityRatio = max(0.0, incomingInterceptVelocity) / maxIncomingVelocity
        incomingVelocityRatio = min(1.0, incomingVelocityRatio)

    outgoingVelocityError = float(outgoingVelocityErrorGain) * velocityDeviation
    airTravelDistanceError = 0.0
    incomingVelocityError = float(incomingVelocityErrorGain) * incomingVelocityRatio
    totalError = outgoingVelocityError + incomingVelocityError

    # Match ExecutionProbability model terms exactly.
    optimalPrecisionDegrees = _SafeFloat(getattr(court, "optimalPrecisionDegrees", None))
    if optimalPrecisionDegrees is None:
        return {"available": False, "reason": "missing optimalPrecisionDegrees"}

    thetaBase = float(optimalPrecisionDegrees) * (np.pi / 180.0)
    thetaEffective = float(thetaBase) * (1.0 + float(totalError))
    radialError = float(airTravelDistance) * float(thetaEffective)

    sigmaNet = _SafeFloat(getattr(court, "sigmaNet", None))
    if sigmaNet is None or sigmaNet <= 0.0:
        sigmaNet = 0.23

    zScore = (0.0 - float(intendedNetClearance)) / float(sigmaNet)
    probabilityHitNet = 0.5 * (1.0 + math.erf(zScore / math.sqrt(2.0)))
    probabilityClearNet = 1.0 - probabilityHitNet

    cellSize = _SafeFloat(getattr(court, "granularity", 1.372))
    marginCells = _SafeFloat(getattr(court, "marginCells", 4))
    margin = float(cellSize) * float(marginCells)

    courtWidth = float(court.singlesRightX) - float(court.singlesLeftX)
    courtLength = float(court.receiverBaselineY) - float(court.serverBaselineY)
    courtXMin = margin
    courtXMax = margin + courtWidth
    courtYMin = float(court.serverBaselineY)
    courtYMax = float(court.receiverBaselineY)

    if shotType == "SERVE":
        if serveSide not in ("DEUCE", "AD"):
            serveSide = "DEUCE" if float(bounceX) <= float(court.centerLineX) else "AD"

        if serveSide == "DEUCE":
            serveXMin = float(court.singlesLeftX)
            serveXMax = float(court.centerLineX)
        else:
            serveXMin = float(court.centerLineX)
            serveXMax = float(court.singlesRightX)

        serveYMin = float(court.netY)
        serveYMax = float(court.opponentServiceLineY)

        left = max(serveXMin, float(bounceX) - radialError)
        right = min(serveXMax, float(bounceX) + radialError)
        bottom = max(serveYMin, float(bounceY) - radialError)
        top = min(serveYMax, float(bounceY) + radialError)
    else:
        left = max(courtXMin, float(bounceX) - radialError)
        right = min(courtXMax, float(bounceX) + radialError)
        bottom = max(courtYMin, float(bounceY) - radialError)
        top = min(courtYMax, float(bounceY) + radialError)

    if right <= left or top <= bottom or radialError <= 0.0:
        probabilityInBounds = 0.0
    else:
        inBoundsArea = (right - left) * (top - bottom)
        totalArea = (2.0 * radialError) ** 2
        probabilityInBounds = inBoundsArea / totalArea

    proximityToNetError = 1.0 - float(probabilityClearNet)
    proximityToBoundaryError = 1.0 - float(probabilityInBounds)

    proximityToNetLuckWeight = _SafeFloat(getattr(court, "proximityToNetLuckWeight", 1.0))
    proximityToBoundaryLuckWeight = _SafeFloat(getattr(court, "proximityToBoundaryLuckWeight", 1.0))

    weightedNetClearanceError = (
        float(proximityToNetLuckWeight) * float(proximityToNetError)
    )
    weightedInBoundsError = (
        float(proximityToBoundaryLuckWeight) * float(proximityToBoundaryError)
    )

    shotLuckCost = (
        float(weightedNetClearanceError)
        + float(weightedInBoundsError)
    )

    return {
        "available": True,
        "incomingVelocityAttribution": incomingVelocityError,
        "outgoingVelocityAttribution": outgoingVelocityError,
        "incomingVelocityError": incomingVelocityError,
        "outgoingVelocityError": outgoingVelocityError,
        "proximityToNetError": proximityToNetError,
        "proximityToBoundaryError": proximityToBoundaryError,
        "proximityToNetLuckWeight": proximityToNetLuckWeight,
        "proximityToBoundaryLuckWeight": proximityToBoundaryLuckWeight,
        "weightedNetClearanceError": weightedNetClearanceError,
        "weightedInBoundsError": weightedInBoundsError,
        "shotLuckCost": shotLuckCost,
        "trackedLuckCostIncrement": trackedLuckCostIncrement,
        "accumulatedLuckCost": trackedLuckCostCumulative,
    }


def _ResolveErrorPanelTitle(shot, title, filename):
    shotEntryNumber = None

    if isinstance(shot, dict):
        for key in ("shotNumber", "shotIndex", "entryNumber", "shotIdx"):
            value = shot.get(key)
            if value is None:
                continue
            try:
                shotEntryNumber = int(value)
                break
            except Exception:
                pass

    if shotEntryNumber is None:
        for text in (title, filename):
            if not isinstance(text, str):
                continue
            match = re.search(r"(?:shot|entry)\s*[_:\- ]\s*(\d+)", text, flags=re.IGNORECASE)
            if not match:
                continue
            try:
                shotEntryNumber = int(match.group(1))
                break
            except Exception:
                pass

    if shotEntryNumber is not None:
        return "Player Error Factors" if (shotEntryNumber % 2 == 1) else "Opp Error Factors"

    if isinstance(shot, dict):
        hitter = str(shot.get("hitter", "")).upper()
        if hitter in ("OPP", "PLAYER_RED"):
            return "Opp Error Factors"
        if hitter in ("PLAYER", "PLAYER_BLUE"):
            return "Player Error Factors"

    return "Player Error Factors"


# ---------------------------
# Main entry
# ---------------------------

def SaveTrajectoryPlot(
    transformed: Dict[str, Any] = None,
    court: Any = None,
    filename: str = "trajectory.png",
    title: str = "Trajectory",
    shot: Optional[Dict[str, Any]] = None,
    nextIntercept: Optional[Tuple[float, float, float, float]] = None,
    interceptPoint: Optional[Tuple[float, float, float]] = None,
    opponentPosition: Optional[Tuple[float, float]] = None,
    # Height band for eligibility (used in side view coloring)
    reachZMin: Optional[float] = None,
    reachZMax: Optional[float] = None,
    # NEW: interception circle controls (plan view)
    showInterceptCircle: bool = False,
    movementModel: Any = None,           # if provided, read speed/reaction/turn from it
    playerSpeedOverride: Optional[float] = None,
    reactionTimeOverride: Optional[float] = None,
    turnCostOverride: Optional[float] = None,
    returnBase64DataUrl: bool = False,
):
    """
    Visualize a single shot with:
      - Plan view (X vs Y): fixed court extents, court lines, truncated trajectory,
        Contact/Bounce/Intercept markers, OPP marker, optional movement arrow to next Intercept,
        and optional **Interception Circle** centered at OPP.
      - Side view (Z vs Y): equal aspect (meters == meters), ground line, thick net segment to 3 ft,
        baseline/service ground boxes, optional **eligibility coloring** (pre-desc, post-asc, post-desc),
        and Intercept marker.
    """
    # Resolve from shot when provided
    hitter = None
    if shot is not None:
        transformed = shot.get("transformed", transformed)
        if interceptPoint is None and shot.get("interceptPoint") is not None:
            ip = shot["interceptPoint"]; interceptPoint = (float(ip[0]), float(ip[1]), float(ip[2]))
        if opponentPosition is None and shot.get("opponentPosition") is not None:
            op = shot["opponentPosition"]; opponentPosition = (float(op[0]), float(op[1]))
        hitter = shot.get("hitter", None)

    if transformed is None or court is None:
        raise ValueError("Visualizer requires both `transformed` and `court`.")

    # Defaults for eligibility band
    if reachZMin is None: reachZMin = 0.10
    if reachZMax is None: reachZMax = 3.0

    defenderPosX = shot["defensivePosX"]
    defenderPosY = shot["defensivePosY"]
    defensiveMovePoint =   defenderPosX, defenderPosY 

    initialVelocity = round(shot["initialVelocity"],2)
    speedMph = initialVelocity / 0.44704

    # Trajectory arrays
    traj3d = transformed["trajectory3D"]  # (N, 3)
    xs = traj3d[:, 0].astype(float)
    ys = traj3d[:, 1].astype(float)
    zs = traj3d[:, 2].astype(float)

    # Bounce index (tagged by runners/transform)
    bounceIdx = int(transformed.get("bounceIndex", 0))
    bounceIdx = max(0, min(bounceIdx, len(ys) - 1))
    bx = float(xs[bounceIdx]); by = float(ys[bounceIdx]); bz = float(zs[bounceIdx])

    # Clip trajectory to in-play region: stop at first post-bounce ground contact.
    inPlayEndIdx = _FindInPlayEndIndex(zs, bounceIdx)
    xs2 = xs[:inPlayEndIdx + 1]
    ys2 = ys[:inPlayEndIdx + 1]
    zs2 = zs[:inPlayEndIdx + 1]

    errorFactors = _ComputeExecutionErrorFactors(shot, court)

    # --------------------------
    # Prepare figure
    # --------------------------
    plt.close('all')
    fig = plt.figure(figsize=(13.0, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.0, 2.0],
        height_ratios=[4.0, 1.25],
        wspace=0.25,
        hspace=0.12,
    )

    # ==========================
    # 1) Plan View (X vs Y)
    # ==========================
           
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_xlim(court.xMin, court.xMax)
    ax1.set_ylim(court.yMin, court.yMax)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xticks([])
    ax1.set_yticks([])

    DrawPlanGridOverlay(ax1, court)

    DrawCourtLines(ax1, court)

    # Trajectory (truncated)
    if len(xs2) > 1:
        ax1.plot(xs2, ys2, "-b", label="Trajectory", zorder=2)

    # Contact
    label = f"Velocity\u00A0{speedMph:.0f}\u00A0mph"
    if len(xs2) > 0:
        ax1.scatter([xs2[0]], [ys2[0]], color="green", s=28, zorder=4)
        ax1.annotate(label, (xs2[0], ys2[0]), textcoords="offset points", xytext=(8, 20), fontsize=8)

    # Bounce (first ground)
    ax1.scatter([bx], [by], color="red", s=36, zorder=5)
    ax1.annotate("Bounce", (bx, by), textcoords="offset points", xytext=(6, 6), fontsize=8)

    # Intercept marker (plan)
    if interceptPoint is not None:
        ix, iy, _ = interceptPoint
        ax1.scatter([ix], [iy], color="#8A2BE2", s=34, zorder=5)
        ax1.annotate(f"Intercept ({ix:.2f}, {iy:.2f})", (ix, iy),
                     textcoords="offset points", xytext=(8, 8), fontsize=8, color="#4B0082")
        
    # --------------------------------------------------------
    # Defensive move visualization (optional)
    # --------------------------------------------------------
    if defensiveMovePoint is not None and interceptPoint is not None:
        ix, iy, _ = interceptPoint
        dmX, dmY = defensiveMovePoint
        # Draw defensive move point
        ax1.plot(
            dmX, dmY,
            marker='o',
            markersize=7,
            color='orange',
            label='Defensive Move'
        )

        # Draw arrow from intercept → defensive move
        ax1.annotate(
            "",
            xy=(dmX, dmY),
            xytext=(ix, iy),
            arrowprops=dict(
                arrowstyle="->",
                lw=2.0,
                color='orange'
            )
        )

    # OPP marker (defender position)
    if opponentPosition is not None:
        ox, oy = opponentPosition
        ax1.scatter([ox], [oy], color="#444", s=42, marker="s", zorder=5)
        ax1.annotate(f"OPP ({ox:.2f}, {oy:.2f})", (ox, oy),
                     textcoords="offset points", xytext=(12, 0), fontsize=8, color="#222")

    # Optional arrow to next intercept (plan)
    if opponentPosition is not None and nextIntercept is not None:
        nx, ny, _, _ = nextIntercept
        ax1.annotate("", xy=(nx, ny), xytext=(opponentPosition[0], opponentPosition[1]),
                     arrowprops=dict(arrowstyle="->", lw=1.5, color="#444"))
        # ax1.annotate("to next intercept", (nx, ny),
        #              textcoords="offset points", xytext=(6, 6), fontsize=8, color="#222")

    # Interception circle (plan) — ToF-based travel radius
    if showInterceptCircle and opponentPosition is not None:
        # Determine defender side for this shot:
        #  - For serve rows: defender is PlayerRed
        #  - For rally rows: defender is opposite of hitter
        if shot is not None and shot.get("type") == "SERVE":
            opponentSide = "PLAYER_RED"
        else:
            if hitter == "PLAYER_BLUE":
                opponentSide = "PLAYER_RED"
            elif hitter == "PLAYER_RED":
                opponentSide = "PLAYER_BLUE"
            else:
                opponentSide = None

        if opponentSide is not None:
            # Movement parameters (from model or overrides)
            ps = float(playerSpeedOverride if playerSpeedOverride is not None
                       else getattr(movementModel, "playerSpeed", 3.6) if movementModel is not None else 3.6)
            rt = float(reactionTimeOverride if reactionTimeOverride is not None
                       else getattr(movementModel, "reactionTime", 0.0) if movementModel is not None else 0.0)
            tc = float(turnCostOverride if turnCostOverride is not None
                       else getattr(movementModel, "turnCost", 0.0) if movementModel is not None else 0.0)

            R, tof_perp, kNearest = _ComputeInterceptRadius(
                transformed=transformed,
                court=court,
                defenderSide=opponentSide,
                defenderPos=opponentPosition,
                playerSpeed=ps,
                reactionTime=rt,
                turnCost=tc,
                maxIndexInclusive=inPlayEndIdx,
            )
            _DrawInterceptCircle(ax1, opponentPosition, R, color="#8A2BE2", alpha=0.30, lw=2.0, label="Intercept radius")

            # Optional nearest-point annotation
            if kNearest is not None and 0 <= kNearest <= inPlayEndIdx:
                nx = float(transformed["fencesX"][kNearest]); ny = float(transformed["fencesY"][kNearest])
                ax1.scatter([nx], [ny], color="#8A2BE2", s=24, zorder=6)
                ax1.annotate(f"Nearest ToF={tof_perp:.2f}s R={R:.2f}m",
                             (nx, ny), textcoords="offset points", xytext=(8, 12),
                             fontsize=8, color="#4B0082")

    # ==========================
    # 2) Side View (Z vs Y)
    # ==========================
        
    ax2 = fig.add_subplot(gs[0, 1])

    # Base (light gray) for the truncated curve
    if len(ys2) > 1:
        ax2.plot(ys2, zs2, "-", color="#CCCCCC", linewidth=1.8, zorder=1)

    # Lock Y to fence‑to‑fence, Z to headroom; equal aspect (meters == meters)
    ax2.set_xlim(court.yMin, court.yMax)
    zMaxObserved = float(np.max(zs2)) if len(zs2) else 1.0
    # zTop = max(1.2, zMaxObserved * 1.10)
    ax2.set_ylim(0.0, 12)
    ax2.set_aspect("equal", adjustable="box")
    #ax2.set_box_aspect(1)

    # Ground line
    ax2.axhline(0.0, color="#666", linestyle="--", linewidth=1.0)

    # Thick net segment to 3 ft (1.3716 m default)
    netHeight = 0.9 #float(getattr(court, "netHeight", 1.3716))
    ax2.plot([court.netY, court.netY], [0.0, netHeight], color="#000",
             linewidth=3.0, solid_capstyle="butt")

    # Ground “boxes” at baselines & service lines
    boxHalfWidthY = 0.20
    boxHeightZ = 0.20
    for yLine in (float(court.serverBaselineY), float(court.serviceLineY),
                  float(court.opponentServiceLineY), float(court.receiverBaselineY)):
        rect = Rectangle((yLine - boxHalfWidthY, 0.0), 2 * boxHalfWidthY, boxHeightZ,
                         facecolor="#666", edgecolor="#444", linewidth=0.8, alpha=0.8)
        ax2.add_patch(rect)

    # Eligibility masks (optional coloring), limited to in-play segment.
    netY = float(court.netY)
    # Opponent half from hitter
    oppHalfMask = None
    if hitter == "PLAYER_BLUE":
        oppHalfMask = (ys >= netY)
    elif hitter == "PLAYER_RED":
        oppHalfMask = (ys <= netY)
    else:
        oppHalfMask = (ys >= netY)  # harmless default

    inPlayMask = np.arange(len(ys), dtype=int) <= int(inPlayEndIdx)
    heightMask = (zs >= float(reachZMin)) & (zs <= float(reachZMax))
    eligibleMask = oppHalfMask & heightMask & inPlayMask

    N = len(ys)
    idxAll = np.arange(N, dtype=int)
    preMask = eligibleMask & (idxAll < bounceIdx)
    postMask = eligibleMask & (idxAll >= bounceIdx)

    # Post-bounce: split ascent/descent at post-bounce apex
    if bounceIdx < N - 1:
        postZ = zs[bounceIdx:]
        peakRel = int(np.argmax(postZ))
        peakIdx = bounceIdx + peakRel
    else:
        peakIdx = bounceIdx

    postAscMask = postMask & (idxAll <= peakIdx)
    postDescMask = postMask & (idxAll > peakIdx)

    # Color the eligible segments
    _AddColoredSegments(ax2, ys, zs, np.where(preMask)[0],      color="#1E90FF", linewidth=2.8, alpha=1.0)  # blue
    _AddColoredSegments(ax2, ys, zs, np.where(postAscMask)[0],  color="#DAA520", linewidth=2.8, alpha=1.0)  # gold
    _AddColoredSegments(ax2, ys, zs, np.where(postDescMask)[0], color="#FF6347", linewidth=2.8, alpha=1.0)  # tomato

    # Bounce marker
    ax2.scatter([by], [bz], color="red", s=36, zorder=4)
    ax2.annotate("Bounce", (by, bz), textcoords="offset points", xytext=(6, 6), fontsize=8)

    # Intercept marker (side)
    if interceptPoint is not None:
        ix, iy, iz = interceptPoint
        ax2.scatter([iy], [iz], color="#8A2BE2", s=36, zorder=4)
        ax2.annotate(f"Intercept\nZ={iz:.2f} m", (iy, iz),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=8, color="#4B0082")

    ax2.set_title("Height Profile (Z vs Y) — Eligibility Highlighted")
    ax2.set_xlabel("Y (m)")
    ax2.set_ylabel("Z (m)")
    ax2.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    # ==========================
    # 3) Error Factors Panel
    # ==========================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_xlim(0.0, 1.0)
    ax3.set_ylim(0.0, 1.0)
    ax3.set_xticks([])
    ax3.set_yticks([])
    for spine in ax3.spines.values():
        spine.set_visible(False)

    panel = Rectangle((0.0, 0.0), 1.0, 1.0,
                      transform=ax3.transAxes,
                      facecolor="#F6F6F6",
                      edgecolor="#B8B8B8",
                      linewidth=0.9)
    ax3.add_patch(panel)

    panelTitle = _ResolveErrorPanelTitle(shot, title, filename)

    ax3.text(
        0.02, 0.96,
        panelTitle,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#222",
        transform=ax3.transAxes,
    )

    if errorFactors.get("available", False):
        shotLuckCostValue = errorFactors.get("shotLuckCost")

        accumulatedLuckCostValue = errorFactors.get("accumulatedLuckCost")
        if accumulatedLuckCostValue is None:
            accumulatedLuckCostValue = shotLuckCostValue

        factorText = "\n".join([
            f"IncomingVelocityAttr: {_Fmt(errorFactors['incomingVelocityAttribution'])}",
            f"OutgoingVelocityAttr: {_Fmt(errorFactors['outgoingVelocityAttribution'])}",
            "-------------------------------",
            f"ClearsNetLuckCost:   {_Fmt(errorFactors['weightedNetClearanceError'])}",
            f"InBoundsLuckCost:    {_Fmt(errorFactors['weightedInBoundsError'])}",
            f"ShotLuckCost:        {_Fmt(shotLuckCostValue)}",
            f"AccumulatedLuckCost: {_Fmt(accumulatedLuckCostValue)}",
        ])
    else:
        factorText = "\n".join([
            "IncomingVelocityAttr: n/a",
            "OutgoingVelocityAttr: n/a",
            "-------------------------------",
            "ClearsNetLuckCost:   n/a",
            "InBoundsLuckCost:    n/a",
            "ShotLuckCost:        n/a",
            "AccumulatedLuckCost: n/a",
        ])

    ax3.text(
        0.02, 0.82,
        factorText,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        color="#222",
        transform=ax3.transAxes,
    )

    # --------------------------
    # Save / encode
    # --------------------------
    fig.suptitle(title)

    if returnBase64DataUrl:
        imageBuffer = BytesIO()
        fig.savefig(imageBuffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        encodedImage = base64.b64encode(imageBuffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encodedImage}"

    outPath = filename if os.path.dirname(filename) else os.path.join("visualizations", filename)
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    fig.savefig(outPath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outPath