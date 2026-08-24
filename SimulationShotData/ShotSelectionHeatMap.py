import duckdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

db = duckdb.connect()

def q(sql):
    return db.execute(sql).fetchdf()

df = q("""
SELECT
    bounceRow,
    bounceCol,
    SUM(count) AS count,
    COUNT(*) AS uniqueTrajecticsCount,
    SUM(wins) * 1.0 / SUM(count) AS winPct
FROM read_parquet("ShotSelection\Generation2\ConsolidatedGeneration003_2_3C.parquet")
WHERE
    --wins * 1.0 / count >= .6
    interceptCol = 7
    AND interceptRow = 10
    AND interceptZ = 1.0
    AND opponentCol = 5
    AND opponentRow = 22
    AND bounceRow BETWEEN 14 AND 22
    AND bounceCol BETWEEN 5 AND 10
GROUP BY bounceRow, bounceCol
"""
)

heat_win = df.pivot(index="bounceRow", columns="bounceCol", values="winPct")
heat_cnt = df.pivot(index="bounceRow", columns="bounceCol", values="count")
heat_trajs = df.pivot(index="bounceRow", columns="bounceCol", values="uniqueTrajecticsCount")

data = heat_win.values
counts = heat_cnt.values
trajs = heat_trajs.values

# Debug: Print sample values to verify calculations
print("\nSample data (first valid cell):")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if not np.isnan(data[i, j]):
            print(f"Cell [{i},{j}]: winPct={data[i, j]:.4f}, count={int(counts[i, j])}, trajs={int(trajs[i, j])}")
            break
    else:
        continue
    break

fig, ax = plt.subplots(figsize=(8, 10))

colors = np.zeros(data.shape + (3,))

# Compute count tiers
valid_counts = counts[~np.isnan(counts)]
q1 = np.quantile(valid_counts, 1/3)
q2 = np.quantile(valid_counts, 2/3)
print(str(q1))
print(str(q2))

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        c = counts[i, j]

        if np.isnan(v) or c < 10:
            colors[i, j] = [0.9, 0.9, 0.9]  # grey (no data)

        elif v < 0.25:
            colors[i, j] = [0.9, 0.6, 0.6]  # red

        elif v < 0.5:
            colors[i, j] = [0.95, 0.9, 0.6]  # yellow

        else:
            # Green with confidence shading
            if c >= q2:
                colors[i, j] = [0.2, 0.7, 0.2]   # dark green
            elif c >= q1:
                colors[i, j] = [0.5, 0.8, 0.5]   # medium green
            else:
                colors[i, j] = [0.75, 0.9, 0.75] # light green

ax.imshow(colors, origin="lower")

# Axis labels
ax.set_xticks(range(len(heat_win.columns)))
ax.set_xticklabels(heat_win.columns)
ax.set_yticks(range(len(heat_win.index)))
ax.set_yticklabels(heat_win.index)

# Annotate winPct, count, and unique trajectics count (without labels)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if not np.isnan(data[i, j]):
            count_val = int(counts[i, j]) if not np.isnan(counts[i, j]) else 0
            trajs_val = int(trajs[i, j]) if not np.isnan(trajs[i, j]) else 0
            ax.text(j, i, f"{data[i, j]:.2f}\n{count_val}\n{trajs_val}",
                    ha="center", va="center", fontsize=8)

# Outline opponent position
opp_col = 5
opp_row = 22

if opp_col in heat_win.columns and opp_row in heat_win.index:
    x = list(heat_win.columns).index(opp_col)
    y = list(heat_win.index).index(opp_row)

    rect = Rectangle(
        (x - 0.5, y - 0.5),
        1, 1,
        linewidth=3,
        edgecolor="black",
        facecolor="none"
    )
    ax.add_patch(rect)

ax.set_xlabel("Bounce Column")
ax.set_ylabel("Bounce Row")
ax.set_title("Bounce Location WinPct Heat Map (Green = ≥50%)")

# Add legend at the bottom
legend_text = (
    "Legend:\n"
    "• Average win percentage\n"
    "• Total number of simulations per bounce cell (count)\n"
    "• Number of unique trajectics per bounce cell (trajs)"
)
fig.text(0.5, 0.02, legend_text, ha="center", fontsize=9, 
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.subplots_adjust(bottom=0.20)
plt.show()
