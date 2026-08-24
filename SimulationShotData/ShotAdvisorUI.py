import streamlit as st
import duckdb
import pandas as pd
import pickle
import numpy as np
import os
from PIL import Image
from io import BytesIO

from Trajectory4D.Visualizer import SaveTrajectoryPlot
from Trajectory4D.TransformLayer import TransformLayer
from Trajectory4D.Trajectory4DCanonical import Trajectory4DCanonical
from CourtPlayerSettings import Court

# -------------------------------
# CONFIG
# -------------------------------
RESULTS_FILE = "FinalShotResults.parquet"
TRAJECTORY_LIBRARY_FILE = "trajectoryLibrary004.pkl"
COURT_IMAGE = "court_grid.png"  # your clickable grid image
TOP_N = 5

GRID_COLS = 14
GRID_ROWS = 26

# -------------------------------
# LOAD TRAJECTORY LIBRARY ONCE
# -------------------------------
@st.cache_resource
def LoadTrajectoryLibrary():
    with open(TRAJECTORY_LIBRARY_FILE, "rb") as f:
        lib = pickle.load(f)
    return lib

trajectoryLibrary = LoadTrajectoryLibrary()

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("🎾 Shot Decision Assistant")

st.write("Tap on the court to select **Player Position** and **Opponent Position**.")

# Load and show the court image
courtImg = Image.open(COURT_IMAGE)
imgWidth, imgHeight = courtImg.size

clickPoint = st.image(courtImg)

# Get click event coordinates
event = st.query_params  # Streamlit click events hack (use streamlit-image-coordinates if preferred)

# -------------------------------
# Mapping clicks → grid cells
# -------------------------------
def MapClickToGrid(x, y):
    col = int((x / imgWidth) * GRID_COLS) + 1
    row = int((y / imgHeight) * GRID_ROWS) + 1
    return col, row

if "playerCell" not in st.session_state:
    st.session_state.playerCell = None
if "opponentCell" not in st.session_state:
    st.session_state.opponentCell = None

# Mock click based on session parameters (or use st_click_detector)
if "x" in event and "y" in event:
    col, row = MapClickToGrid(float(event["x"]), float(event["y"]))

    if st.session_state.playerCell is None:
        st.session_state.playerCell = (col, row)
        st.success(f"Player position set to: {st.session_state.playerCell}")
    else:
        st.session_state.opponentCell = (col, row)
        st.success(f"Opponent position set to: {st.session_state.opponentCell}")

# -------------------------------
# When both positions are chosen → Query Parquet
# -------------------------------
if st.session_state.playerCell and st.session_state.opponentCell:

    playerCol, playerRow = st.session_state.playerCell
    opponentCol, opponentRow = st.session_state.opponentCell

    st.subheader("🔍 Looking up best shots…")

    # Query DuckDB
    conn = duckdb.connect()

    QUERY = f"""
        SELECT *
        FROM read_parquet('{RESULTS_FILE}')
        WHERE offensiveCol = {playerCol}
          AND offensiveRow = {playerRow}
          AND opponentCol  = {opponentCol}
          AND opponentRow  = {opponentRow}
        ORDER BY winPercentage DESC
        LIMIT {TOP_N}
    """

    df = conn.execute(QUERY).df()

    if df.empty:
        st.error("No matching shots found for this configuration.")
    else:
        st.write(f"Top {TOP_N} recommended shots:")
        st.dataframe(df)

        # -------------------------------
        # Select one shot
        # -------------------------------
        chosenIndex = st.selectbox("Select a shot", df.index)

        chosenShot = df.loc[chosenIndex]

        # -------------------------------
        # Load trajectory from canonical library
        # -------------------------------
        interceptCol = int(chosenShot["interceptCol"])
        interceptRow = int(chosenShot["interceptRow"])
        interceptZ   = float(chosenShot["interceptZ"])

        apexHeight   = float(chosenShot["apexHeight"])
        spinTopRpm   = int(chosenShot["spinTopRpm"])
        spinSideRpm  = int(chosenShot["spinSideRpm"])

        # Build canonical lookup key (depends on your library structure)
        # Example:
        key = (interceptCol, interceptRow, interceptZ,
               apexHeight, spinTopRpm, spinSideRpm)

        if key not in trajectoryLibrary:
            st.error("Trajectory not found in library.")
        else:
            trajectoryEntry = trajectoryLibrary[key]

            # -------------------------------
            # Visualize trajectory
            # -------------------------------
            st.subheader("📈 Shot Trajectory Visualization")

            tempFile = "temp_shot_plot.png"

            SaveTrajectoryPlot(
                shot=trajectoryEntry,
                transformed=trajectoryEntry["transformed"],
                court=Court(),
                filename=tempFile,
                title="Shot Visualization",
                nextIntercept=None,
                showInterceptCircle=False
            )

            st.image(tempFile, caption="Recommended Shot")

            st.success("Shot visualization complete!")
``