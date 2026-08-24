import math
"""
Design notes: the interception point and all movements are about the head 
of the racket and we include no information about the player per se.
Note also that we do not implement forced errors but only unforced errors.
Forced errors are an extension of not perfectly reaching or reacting to the ball in time
and so are included in player movement and reaction time failures. 
"""

class ExecutionProbability:
    def __init__(
        self,
        court,
        cellSize=1.372,          # 4.5 ft → meters
        marginCells=4,
        sigmaNet=0.23,           # ~0.75 ft → meters
        outgoingVelocityErrorGain=0.25,
        velocityErrorGain=None,
        incomingVelocityErrorGain=0.35,
        proximityToNetLuckWeight=1.0,
        proximityToBoundaryLuckWeight=1.0,
    ):
        self.court = court
        # -------------------------------------------------
        # Geometry (meters)
        # -------------------------------------------------
        self.cellSize = court.granularity
        self.marginCells = marginCells
        self.margin = cellSize * marginCells

        self.courtWidth = court.singlesRightX - court.singlesLeftX
        self.courtLength = court.receiverBaselineY - court.serverBaselineY
        self.netHeight = court.netHeight

        self.courtXMin = self.margin
        self.courtXMax = self.margin + self.courtWidth
        self.courtYMin = court.serverBaselineY
        self.courtYMax = court.receiverBaselineY
        self.netY = self.courtYMin + self.courtLength / 2.0

        # -------------------------------------------------
        # Precision / noise model
        # -------------------------------------------------
        self.sigmaNet = sigmaNet

        # player parameters
        self.maxIncomingVelocity = float(
            getattr(court, "maxIncomingVelocity", 58.1152)
        )
        self.optimalPrecisionDegrees = court.optimalPrecisionDegrees

        # Respect Court-level tuning first; fall back to constructor defaults.
        # Backward compatibility: if legacy velocityErrorGain is provided, treat it as outgoing.
        outgoingGainDefault = (
            float(velocityErrorGain)
            if velocityErrorGain is not None
            else float(outgoingVelocityErrorGain)
        )
        self.outgoingVelocityErrorGain = float(
            getattr(court, "outgoingVelocityErrorGain", outgoingGainDefault)
        )
        self.incomingVelocityErrorGain = float(
            getattr(court, "incomingVelocityErrorGain", incomingVelocityErrorGain)
        )

        self.proximityToNetLuckWeight = float(
            getattr(court, "proximityToNetLuckWeight", proximityToNetLuckWeight)
        )
        self.proximityToBoundaryLuckWeight = float(
            getattr(court, "proximityToBoundaryLuckWeight", proximityToBoundaryLuckWeight)
        )

        self.degreesToRadians = math.pi / 180.0

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def computeExecutionProbability(
        self,
        bounceX,
        bounceY,
        airTravelDistance,
        initialVelocity,
        intendedNetClearance,
        incomingInterceptVelocity=None,
        shotType=None,
        serveSide=None,
    ):
        return self.computeExecutionDetails(
            bounceX=bounceX,
            bounceY=bounceY,
            airTravelDistance=airTravelDistance,
            initialVelocity=initialVelocity,
            intendedNetClearance=intendedNetClearance,
            incomingInterceptVelocity=incomingInterceptVelocity,
            shotType=shotType,
            serveSide=serveSide,
        )["executionProbability"]

    def computeExecutionDetails(
        self,
        bounceX,
        bounceY,
        airTravelDistance,
        initialVelocity,
        intendedNetClearance,
        incomingInterceptVelocity=None,
        shotType=None,
        serveSide=None,
    ):
        if (
            bounceX is None
            or bounceY is None
            or airTravelDistance is None
            or initialVelocity is None
            or intendedNetClearance is None
        ):
            return {
                "executionProbability": 0.0,
                "probabilityClearNet": 0.0,
                "probabilityInBounds": 0.0,
                "incomingVelocityError": 0.0,
                "outgoingVelocityError": 0.0,
                "airTravelDistanceError": 0.0,
                "proximityToNetError": 1.0,
                "proximityToBoundaryError": 1.0,
                "weightedNetClearanceError": 0.0,
                "weightedInBoundsError": 0.0,
                "proximityToNetLuckWeight": float(self.proximityToNetLuckWeight),
                "proximityToBoundaryLuckWeight": float(self.proximityToBoundaryLuckWeight),
                "shotLuckCost": 0.0,
                "reason": "missing required input",
            }

        # Base angular precision
        thetaBase = self.optimalPrecisionDegrees * self.degreesToRadians

        # -------------------------------------------------
        # Linear monotonic deviation model:
        # higher velocity increases angular deviation.
        # Distance still scales radial error directly.
        # -------------------------------------------------
        velocityScale = self.maxIncomingVelocity if self.maxIncomingVelocity > 0.0 else 1.0

        velocityDeviation = max(0.0, float(initialVelocity)) / float(velocityScale)

        normalizedShotType = str(shotType or "").upper()

        incomingVelocityRatio = 0.0
        if incomingInterceptVelocity is not None and self.maxIncomingVelocity > 0.0:
            incomingVelocityRatio = max(0.0, float(incomingInterceptVelocity)) / self.maxIncomingVelocity
            incomingVelocityRatio = min(1.0, incomingVelocityRatio)

        outgoingVelocityError = self.outgoingVelocityErrorGain * velocityDeviation
        airTravelDistanceError = 0.0
        incomingVelocityError = self.incomingVelocityErrorGain * incomingVelocityRatio

        thetaEffective = thetaBase * (
            1.0
            + outgoingVelocityError
            + incomingVelocityError
        )

        # -------------------------------------------------
        # Radial landing error (meters)
        # -------------------------------------------------
        radialError = airTravelDistance * thetaEffective

        # -------------------------------------------------
        # Probability components
        # -------------------------------------------------
        probabilityClearNet = self._computeNetClearanceProbability(
            intendedNetClearance
        )

        if normalizedShotType == "SERVE":
            probabilityInBounds = self._computeServeInBoundsProbability(
                bounceX=bounceX,
                bounceY=bounceY,
                radialError=radialError,
                serveSide=serveSide,
            )
        else:
            probabilityInBounds = self._computeInBoundsProbability(
                bounceX,
                bounceY,
                radialError
            )

        executionProbability = probabilityClearNet * probabilityInBounds

        proximityToNetError = 1.0 - float(probabilityClearNet)
        proximityToBoundaryError = 1.0 - float(probabilityInBounds)

        weightedNetClearanceError = (
            self.proximityToNetLuckWeight * proximityToNetError
        )
        weightedInBoundsError = (
            self.proximityToBoundaryLuckWeight * proximityToBoundaryError
        )
        shotLuckCost = (
            weightedNetClearanceError
            + weightedInBoundsError
        )

        return {
            "executionProbability": float(executionProbability),
            "probabilityClearNet": float(probabilityClearNet),
            "probabilityInBounds": float(probabilityInBounds),
            "incomingVelocityError": float(incomingVelocityError),
            "outgoingVelocityError": float(outgoingVelocityError),
            "airTravelDistanceError": float(airTravelDistanceError),
            "proximityToNetError": float(proximityToNetError),
            "proximityToBoundaryError": float(proximityToBoundaryError),
            "weightedNetClearanceError": float(weightedNetClearanceError),
            "weightedInBoundsError": float(weightedInBoundsError),
            "proximityToNetLuckWeight": float(self.proximityToNetLuckWeight),
            "proximityToBoundaryLuckWeight": float(self.proximityToBoundaryLuckWeight),
            "shotLuckCost": float(shotLuckCost),
        }

    # -------------------------------------------------
    # Net clearance probability
    # -------------------------------------------------
    def _computeNetClearanceProbability(self, intendedNetClearance):
        if intendedNetClearance is None:
            return 0.0
        zScore = (0.0 - intendedNetClearance) / self.sigmaNet
        probabilityHitNet = self._normalCdf(zScore)
        return 1.0 - probabilityHitNet

    def _normalCdf(self, x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # -------------------------------------------------
    # In-bounds probability
    # -------------------------------------------------
    def _computeInBoundsProbability(self, bounceX, bounceY, radialError):
        left = max(self.courtXMin, bounceX - radialError)
        right = min(self.courtXMax, bounceX + radialError)
        bottom = max(self.courtYMin, bounceY - radialError)
        top = min(self.courtYMax, bounceY + radialError)

        if right <= left or top <= bottom:
            return 0.0

        inBoundsArea = (right - left) * (top - bottom)
        totalArea = (2.0 * radialError) ** 2

        return inBoundsArea / totalArea

    def _computeServeInBoundsProbability(self, bounceX, bounceY, radialError, serveSide):
        if radialError <= 0.0:
            return 0.0

        side = str(serveSide or "").upper()
        if side not in ("DEUCE", "AD"):
            # Infer side from bounce X when serve side is absent.
            side = "DEUCE" if float(bounceX) <= float(self.court.centerLineX) else "AD"

        serviceYMin = float(getattr(self.court, "netY", self.netY))
        serviceYMax = float(getattr(self.court, "opponentServiceLineY", self.netY))

        if side == "DEUCE":
            serviceXMin = float(self.court.singlesLeftX)
            serviceXMax = float(self.court.centerLineX)
        else:
            serviceXMin = float(self.court.centerLineX)
            serviceXMax = float(self.court.singlesRightX)

        left = max(serviceXMin, float(bounceX) - radialError)
        right = min(serviceXMax, float(bounceX) + radialError)
        bottom = max(serviceYMin, float(bounceY) - radialError)
        top = min(serviceYMax, float(bounceY) + radialError)

        if right <= left or top <= bottom:
            return 0.0

        inBoundsArea = (right - left) * (top - bottom)
        totalArea = (2.0 * radialError) ** 2
        return inBoundsArea / totalArea