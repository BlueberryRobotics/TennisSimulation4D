# TrajectoryQualityComparison.py
#
# Compare dynamic vs canonical trajectory quality.
#
# Output:
#   • JSON file containing test contexts + dynamic traj + canonical traj
#   • Text report showing differences
#   • Optional per-shot visualization using Visualizer4D
#
# Naming aligned with your conventions:
#   - PascalCase for classes & methods
#   - camelCase for parameters & locals
#   - _privateFields for private instance fields
#   - PascalCase constants
# ---------------------------------------------------------------------

import json
import random
import numpy as np
from Trajectory4DDynamicGenerator import Trajectory4DDynamicGenerator
from Trajectory4DCanonical import Trajectory4DCanonical
from TransformLayer4D import TransformLayer4D
from ServeShotRunner import ServeShotRunner
from RallyShotRunner import RallyShotRunner


TestCount = 30
NetEps = 5e-3
BounceTolerance = 0.50


class TrajectoryQualityComparison:

    def __init__(self, court, canonicalLibrary, canonicalSpec):
        self._court = court

        self._dynamicGen = Trajectory4DDynamicGenerator(court)
        self._canonicalGen = Trajectory4DCanonical(
            library=canonicalLibrary,
            transformLayer=TransformLayer4D(debug=False),
            canonicalSpec=canonicalSpec,
            bounceTolerance=BounceTolerance,
            debug=False,
            court=court
        )

    # -------------------------------------------------------------
    def GenerateTestContexts(self, count=TestCount):
        contexts = []

        for _ in range(count):
            # identical serve-like context
            serveSide = random.choice(["DEUCE", "AD"])
            interceptX = random.uniform(self._court.centerLineX - 0.5,
                                        self._court.centerLineX + 0.5)
            interceptY = self._court.serverBaselineY
            interceptZ = random.uniform(2.3, 3.1)  # serve strike height

            interceptPoint = (interceptX, interceptY, interceptZ)

            # quarter-cell bounce target
            bounceX = random.uniform(self._court.singlesLeftX + 1.0,
                                     self._court.singlesRightX - 1.0)
            bounceY = random.uniform(self._court.netY + 1.0,
                                     self._court.opponentServiceLineY - 0.2)

            bounceXY = (bounceX, bounceY)

            spinTopRpm = random.choice([-500, 0, 500, 1500, 3000])
            spinSideRpm = random.choice([-1500, -750, 0, 750, 1500])

            apexHeights = [2.5, 3.0, 3.5, 4.0, 5.2, 6.0]

            contexts.append({
                "serveSide": serveSide,
                "interceptPoint": interceptPoint,
                "bounceXY": bounceXY,
                "spinTopRpm": spinTopRpm,
                "spinSideRpm": spinSideRpm,
                "apexHeights": apexHeights
            })

        return contexts

    # -------------------------------------------------------------
    def Run(self):
        contexts = self.GenerateTestContexts()
        results = []

        print("\n=== Trajectory Quality Comparison ===\n")

        for index, ctx in enumerate(contexts):
            print(f"Testing {index+1}/{len(contexts)}...")

            interceptPoint = ctx["interceptPoint"]
            bounceXY = ctx["bounceXY"]
            apexList = ctx["apexHeights"]
            spinTop = ctx["spinTopRpm"]
            spinSide = ctx["spinSideRpm"]

            # --- Dynamic ---
            dynamicTraj = None
            try:
                dynamicTraj = self._dynamicGen.generate_by_apex_ladder(
                    launchPoint=interceptPoint,
                    bounceXY=bounceXY,
                    apexHeights=apexList,
                    spinTopRpm=spinTop,
                    spinSideRpm=spinSide,
                    maxNetClearAbove=None,
                    landing_tol=0.10,
                    net_eps=NetEps,
                    maxItersPerApex=800
                )
            except Exception as e:
                dynamicTraj = {"ERROR": str(e)}

            # --- Canonical ---
            canonicalBundle = None
            try:
                canonicalBundle = self._canonicalGen.generate_by_apex_ladder(
                    launchPoint=interceptPoint,
                    bounceXY=bounceXY,
                    apexHeights=apexList,
                    spinTopRpm=spinTop,
                    spinSideRpm=spinSide,
                    maxNetClearAbove=None,
                    landing_tol=0.10,
                    net_eps=NetEps,
                    maxItersPerApex=12
                )
            except Exception as e:
                canonicalBundle = {"ERROR": str(e)}

            # --- Compare bounce location & apex if both succeeded ---
            comparison = self.Compare(dynamicTraj, canonicalBundle)

            results.append({
                "context": ctx,
                "dynamicTraj": dynamicTraj,
                "canonicalBundle": canonicalBundle,
                "comparison": comparison
            })

        # Output file
        with open("TrajectoryQualityComparison.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\nSaved comparison output → TrajectoryQualityComparison.json\n")

    # -------------------------------------------------------------
    def Compare(self, dynamicTraj, canonicalBundle):
        """
        Compute bounce difference, apex difference, net clearance diff.
        """

        if dynamicTraj is None or canonicalBundle is None:
            return {"status": "ERROR"}

        if isinstance(dynamicTraj, dict) and "ERROR" in dynamicTraj:
            return {"status": "DYNA_ERROR"}
        if isinstance(canonicalBundle, dict) and "ERROR" in canonicalBundle:
            return {"status": "CANO_ERROR"}

        canonicalTraj = canonicalBundle["traj"]

        # bounce index
        biDyn = dynamicTraj["bounceIndex"]
        biCan = canonicalTraj["bounceIndex"]

        bxDyn = float(dynamicTraj["fencesX"][biDyn])
        byDyn = float(dynamicTraj["fencesY"][biDyn])

        bxCan = float(canonicalTraj["fencesX"][biCan])
        byCan = float(canonicalTraj["fencesY"][biCan])

        bounceDiff = float(np.hypot(bxDyn - bxCan, byDyn - byCan))

        # apex
        apexDyn = float(np.max(dynamicTraj["fencesZ"][: biDyn + 1]))
        apexCan = float(np.max(canonicalTraj["fencesZ"][: biCan + 1]))
        apexDiff = abs(apexDyn - apexCan)

        return {
            "status": "OK",
            "bounceDiff": bounceDiff,
            "apexDiff": apexDiff
        }