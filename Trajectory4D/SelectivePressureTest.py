from Trajectory4D.SelectivePressure import SelectivePressure

trajecticsSelector = SelectivePressure(
    "../SimulationShotData/ShotSelection/GenerationALL/ConsolidatedGenerationALL2.parquet",
    0.5,
    2)

apexValues = [
    1.00, 1.25, 1.50, 1.80, 2.10, 2.40, 2.70,
    3.00, 3.30, 3.60, 4.50, 6.00, 8.00, 10.00
]

trajectic = trajecticsSelector.SampleTrajectic(
    interceptCol=9,
    interceptRow=5,
    interceptZ=2.7,
    opponentCol=7,
    opponentRow=23,
    apexValues=apexValues
)

if trajectic is None:
    print("None")
    # fallback: random legal shot, or canonical generator
else:
    # pass tactic directly into the simulator
    # run_shot(
    #     interceptCol=,
    #     opponentCell=...,
    #     bouncePoint=(tactic["bounceCol"], tactic["bounceRow"]),
    #     apexHeight=tactic["apexHeight"],
    #     spinTop=tactic["spinTopRpm"],
    #     spinSide=tactic["spinSideRpm"],
    #     defensiveCell=(tactic["defensiveCol"], tactic["defensiveRow"])
    # )
    print("intercept: " + str(trajectic["interceptCol"]) + " " + str(trajectic["interceptRow"]) + " " + str(trajectic["interceptZ"]))
    print("bouncePoint: " + str(trajectic["bounceCol"]) + " " + str(trajectic["bounceRow"]))
    print("spinTopRpm=" + str(trajectic["spinTopRpm"]))
    print("spinSideRpm=" + str(trajectic["spinSideRpm"]))
    print("apexHeight=" + str(trajectic["apexHeight"]))
    print("defensive move " + str(trajectic["defensiveCol"]) + " " + str(trajectic["defensiveRow"]))

