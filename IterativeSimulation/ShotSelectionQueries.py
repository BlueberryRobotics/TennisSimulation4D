import duckdb

conn = duckdb.connect()

PARQUET = "ShotSelection/Generation2/ConsolidatedGeneration003_2_3B.parquet"
OUTPUT = "TopTrajectics_WithExploration.csv"

def query(sql):
    return duckdb.execute(sql).fetchdf()

# Total Number of rows 
sql="""
SELECT COUNT(*) AS num_rows
FROM read_parquet("shots_20260708_043624.parquet")

"""
totalRows = query(sql)
print("Total Rows: "+ str(totalRows))

# Number of rows with wins = 0
sql="""
SELECT COUNT(*) AS num_rows
FROM read_parquet("shots_20260708_043624.parquet")
WHERE
    wins = 0 
"""
losingRows = query(sql)
print("Losing Rows: " + str(losingRows))

# Number of rows with count >= 10
sql="""
SELECT COUNT(*) AS num_rows
FROM read_parquet("ConsolidatedGen14.parquet")
WHERE
    count >= 10
"""
count10Rows = query(sql)
print("Rows with count >= 10: " + str(count10Rows))

# Number of rows with count >= 20
sql="""
SELECT COUNT(*) AS num_rows
FROM read_parquet("ConsolidatedGen14.parquet")
WHERE
    count >= 20
"""
count20Rows = query(sql)
print("Rows with count >= 20: " + str(count20Rows))

# Number of rows with count >= 30
sql="""
SELECT COUNT(*) AS num_rows
FROM read_parquet("ConsolidatedGen14.parquet")
WHERE
    count >= 30
"""
count30Rows = query(sql)
print("Rows with count >= 30: " + str(count30Rows))

# Number of rows with count >= 40
sql="""
SELECT COUNT(*) AS num_rows
FROM read_parquet("ConsolidatedGen14.parquet")
WHERE
    count >= 40
"""
count40Rows = query(sql)
print("Rows with count >= 40: " + str(count40Rows))

# Number of rows with count >= 50
sql="""
SELECT COUNT(*) AS num_rows
FROM read_parquet("ConsolidatedGen14.parquet")
WHERE
    count >= 50
"""
count50Rows = query(sql)
print("Rows with count >= 50: " + str(count50Rows))


# query = f"""
# WITH base AS (
#     SELECT *
#     FROM read_parquet('{PARQUET}')
#     WHERE wins > 0 AND count >= 2
# ),

# context_bounce_stats AS (
#     SELECT
#         interceptCol,
#         interceptRow,
#         interceptZ,
#         opponentCol,
#         opponentRow,
#         bounceCol,
#         bounceRow,
#         SUM(wins) AS totalWins,
#         SUM(count) AS totalCount,
#         SUM(wins) * 1.0 / SUM(count) AS winPct
#     FROM base
#     GROUP BY
#         interceptCol, interceptRow, interceptZ,
#         opponentCol, opponentRow,
#         bounceCol, bounceRow
# ),

# -- Winning bounce cells (random 3 per context)
# winning_cells AS (
#     SELECT *
#     FROM (
#         SELECT *,
#             ROW_NUMBER() OVER (
#                 PARTITION BY
#                     interceptCol, interceptRow, interceptZ,
#                     opponentCol, opponentRow
#                 ORDER BY RANDOM()
#             ) AS rand_rank
#         FROM context_bounce_stats
#         WHERE winPct > 0.5
#     )
#     WHERE rand_rank <= 4
# ),

# -- Losing bounce cells (random 1 per context)
# losing_cells AS (
#     SELECT *
#     FROM (
#         SELECT *,
#             ROW_NUMBER() OVER (
#                 PARTITION BY
#                     interceptCol, interceptRow, interceptZ,
#                     opponentCol, opponentRow
#                 ORDER BY RANDOM()
#             ) AS rand_rank
#         FROM context_bounce_stats
#         WHERE winPct <= 0.5
#     )
#     WHERE rand_rank <= 1
# ),

# -- Combine both
# selected_cells AS (
#     SELECT * FROM winning_cells
#     UNION ALL
#     SELECT * FROM losing_cells
# ),

# -- Join back to trajectics
# joined AS (
#     SELECT
#         b.*,
#         (b.wins * 1.0 / b.count) AS winPct
#     FROM base b
#     JOIN selected_cells c
#         ON b.interceptCol = c.interceptCol
#         AND b.interceptRow = c.interceptRow
#         AND b.interceptZ = c.interceptZ
#         AND b.opponentCol = c.opponentCol
#         AND b.opponentRow = c.opponentRow
#         AND b.bounceCol = c.bounceCol
#         AND b.bounceRow = c.bounceRow
# ),

# -- One trajectic per apexHeight
# apex_dedup AS (
#     SELECT *,
#         ROW_NUMBER() OVER (
#             PARTITION BY
#                 interceptCol, interceptRow, interceptZ,
#                 opponentCol, opponentRow,
#                 bounceCol, bounceRow,
#                 apexHeight
#             ORDER BY winPct DESC, count DESC
#         ) AS apex_rank
#     FROM joined
# ),

# apex_unique AS (
#     SELECT *
#     FROM apex_dedup
#     WHERE apex_rank = 1
# ),

# -- Limit per bounce cell
# ranked AS (
#     SELECT *,
#         ROW_NUMBER() OVER (
#             PARTITION BY
#                 interceptCol, interceptRow, interceptZ,
#                 opponentCol, opponentRow,
#                 bounceCol, bounceRow
#             ORDER BY winPct DESC, count DESC
#         ) AS trajectic_rank
#     FROM apex_unique
# )

# SELECT *
# FROM ranked
# WHERE trajectic_rank <= 3
# """
# count = conn.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]
# print(f"Row count: {count:,}")

# conn.execute(f"""
# COPY ({query})
# TO '{OUTPUT}'
# (HEADER, DELIMITER ',')
# """)

# print(f"✅ Export complete → {OUTPUT}")


# PARQUET = "ShotSelection/Generation2/ConsolidatedGeneration003_2_3B.parquet"

# output_csv = "Top50FromEachBounceCell001.csv"
# query = f"""
# COPY (
#         WITH context_bounce_stats AS (
#             SELECT
#                 interceptCol,
#                 interceptRow,
#                 interceptZ,
#                 opponentCol,
#                 opponentRow,
#                 bounceCol,
#                 bounceRow,
#                 SUM(wins) AS totalWins,
#                 SUM(count) AS totalCount,
#                 SUM(wins) * 1.0 / NULLIF(SUM(count), 0) AS winPct
#             FROM read_parquet('{PARQUET}')
#             WHERE
#                 wins > 0
#                 AND count >= 2
#             GROUP BY
#                 interceptCol, interceptRow, interceptZ,
#                 opponentCol, opponentRow,
#                 bounceCol, bounceRow
#         ),

#         -- ✅ Top 10 successful bounce cells per context
#         top_success_cells AS (
#             SELECT *,
#                 ROW_NUMBER() OVER (
#                     PARTITION BY
#                         interceptCol, interceptRow, interceptZ,
#                         opponentCol, opponentRow
#                     ORDER BY
#                         winPct DESC, totalWins DESC
#                 ) AS rnk
#             FROM context_bounce_stats
#             WHERE winPct > 0.5
#         ),
#         selected_success_cells AS (
#             SELECT * FROM top_success_cells WHERE rnk <= 5
#         ),

#         -- ✅ Bottom 5 least explored cells per context
#         least_explored_cells AS (
#             SELECT *,
#                 ROW_NUMBER() OVER (
#                     PARTITION BY
#                         interceptCol, interceptRow, interceptZ,
#                         opponentCol, opponentRow
#                     ORDER BY
#                         totalCount ASC
#                 ) AS rnk
#             FROM context_bounce_stats
#         ),
#         selected_explored_cells AS (
#             SELECT * FROM least_explored_cells WHERE rnk <= 1
#         ),

#         -- ✅ Combine selected bounce cells
#         selected_cells AS (
#             SELECT * FROM selected_success_cells
#             UNION ALL
#             SELECT * FROM selected_explored_cells
#         ),

#         -- ✅ Pull trajectics for those cells
#         ranked_trajectics AS (
#             SELECT
#                 t.*,

#                 (t.wins * 1.0 / NULLIF(t.count, 0)) AS winPct,

#                 ROW_NUMBER() OVER (
#                     PARTITION BY
#                         t.interceptCol, t.interceptRow, t.interceptZ,
#                         t.opponentCol, t.opponentRow,
#                         t.bounceCol, t.bounceRow
#                     ORDER BY
#                         (t.wins * 1.0 / NULLIF(t.count, 0)) DESC,
#                         t.count DESC
#                 ) AS trajectic_rank

#             FROM read_parquet('{PARQUET}') t
#             JOIN selected_cells c
#                 ON t.interceptCol = c.interceptCol
#                 AND t.interceptRow = c.interceptRow
#                 AND t.interceptZ = c.interceptZ
#                 AND t.opponentCol = c.opponentCol
#                 AND t.opponentRow = c.opponentRow
#                 AND t.bounceCol = c.bounceCol
#                 AND t.bounceRow = c.bounceRow
#         )

#         -- ✅ Final selection: top 5 trajectics per bounce cell
#         SELECT *
#         FROM ranked_trajectics
#         WHERE trajectic_rank <= 5
# )
# TO '{output_csv}' (HEADER, DELIMITER ',');
# """

# count = duckdb.connect().execute(query)

# print("Export complete →", output_csv)




# sql = f"""
#     SELECT
#         wins,
#         count,
#         wins * 1.0 / count AS winPct
#     FROM read_parquet('{PARQUET}')
#     WHERE winPct >= 0.2 AND count >= 2
# """

# rows =  query(sql)
# print("Rows Count: " + str(len(rows)))

# interceptZValues = [
#     0.30, 0.60, 1.00, 1.25, 1.50, 1.80,
#     2.10, 2.40, 2.70, 3.00, 3.30
# ]
# Court rows: 5,6,7 | 8,9,10
# Court columns: 5,6,7,8,9 | 10,11,12,13 | 14,15,16,17 | 18,19,20,21,22

# 9, 5, 2.7, 5, 23

# #--------------------------------

# parquet_path = "ShotSelection/GenerationALL/ConsolidatedGenerationALL3.parquet"

# query = f"""
# SELECT
#     COUNT(*) AS total,
#     MIN(initialVelocity) AS min_vel,
#     MAX(initialVelocity) AS max_vel,
#     AVG(initialVelocity) AS avg_vel,

#     AVG(apexHeight) AS avg_apex,
#     AVG(interceptZ) AS avg_intercept,

#     AVG(apexHeight - interceptZ) AS avg_deltaZ,
#     MIN(apexHeight - interceptZ) AS min_deltaZ,
#     MAX(apexHeight - interceptZ) AS max_deltaZ

# FROM read_parquet('{parquet_path}')
# WHERE initialVelocity >= 0.0
# """
# # Execute query
# result = duckdb.connect().execute(query).fetchone()

# # Unpack results
# (
#     total,
#     min_vel,
#     max_vel,
#     avg_vel,
#     avg_apex,
#     avg_intercept,
#     avg_deltaZ,
#     min_deltaZ,
#     max_deltaZ,
# ) = result


# # ----------------------------
# # PRINT RESULTS
# # ----------------------------

# print("\n=== HIGH SPEED TRAJECTORY STATS (>= 0.0 m/s) ===\n")

# print("Source: " + parquet_path)

# print(f"Total trajectories in context (TrajICs): {int(total):,}")

# print("\n--- Velocity (m/s) ---")
# print(f"Min:   {min_vel:.2f}")
# print(f"Max:   {max_vel:.2f}")
# print(f"Mean:  {avg_vel:.2f}")

# print("\n--- Velocity (mph) ---")
# print(f"Min:   {min_vel * 2.237:.2f}")
# print(f"Max:   {max_vel * 2.237:.2f}")
# print(f"Mean:  {avg_vel * 2.237:.2f}")

# print("\n--- Geometry ---")
# print(f"Average InterceptZ:  {avg_intercept:.2f} m")
# print(f"Average ApexHeight: {avg_apex:.2f} m")

# print("\n--- Apex Difference (ΔZ = apex - intercept) ---")
# print(f"Average ΔZ: {avg_deltaZ:.4f} m")
# print(f"Min ΔZ:     {min_deltaZ:.4f} m")
# print(f"Max ΔZ:     {max_deltaZ:.4f} m")

# print("\n==============================================\n")

# #---------------------------------

# output_csv = "HighSpeedTrajics_Top1000.csv"

# query = f"""
# COPY (
#     SELECT
#         interceptCol,
#         interceptRow,
#         interceptZ,
#         opponentCol,
#         opponentRow,
#         bounceCol,
#         bounceRow,
#         apexHeight,
#         spinTopRpm,
#         spinSideRpm,
#         initialVelocity,
#         wins,
#         count,
#         wins * 1.0 / count AS winPct
#     FROM read_parquet('{parquet_path}')
#     WHERE initialVelocity >= 18.0
#     ORDER BY initialVelocity DESC
#     LIMIT 1000
# )
# TO '{output_csv}' (HEADER, DELIMITER ',');
# """

# duckdb.connect().execute(query)

# print("Export complete →", output_csv)


# # get top return shots based on intercept and opponent positions
# sql = """
# COPY (
#     SELECT
#         *,
#         wins * 1.0 / count AS winPct
#     FROM read_parquet("ShotSelection/GenerationALL/ConsolidatedGenerationALL3.parquet")
#     WHERE
#         interceptCol = 9
#         AND interceptRow = 5
#         AND interceptZ = 2.7
#         AND opponentCol = 5
#         AND opponentRow = 23
#         --AND bounceCol BETWEEN 5 AND 10
#         --AND bounceRow BETWEEN 14 AND 18
#         AND winPct >= .5
#         AND count >= 3.0
#         --ORDER BY count DESC, winPct DESC, avgPointShotCount ASC
#         ORDER BY bounceCol ASC, bounceRow ASC
#         LIMIT 10000
# )

# TO 'Rallies_0905270523.csv'
# (HEADER, DELIMITER ',')
# """
# query(sql)

# # get top return shots based on intercept and opponent positions
# sql = """
# COPY (
#     SELECT
#         *,
#         wins * 1.0 / count AS winPct
#     FROM read_parquet("ShotSelection/Generation2/ConsolidatedGeneration2A.parquet")
#     WHERE
#         -- 7, 5, 2.7, 10, 23
#         interceptCol = 7
#         AND interceptRow = 5
#         AND interceptZ = 2.70
#         AND opponentCol = 10
#         AND opponentRow = 23
#         --AND winPct >= .5
#         --AND count >= 3
#         --ORDER BY count DESC, winPct DESC, avgPointShotCount ASC
#         ORDER BY bounceCol ASC, bounceRow ASC
#         --LIMIT 100
# )

# TO 'Shot_7_5_27_10_23.csv'
# (HEADER, DELIMITER ',')
# """
# query(sql)

# get top return shots based on intercept and opponent positions
# sql = """
# COPY (
#     SELECT
#         *,
#         wins * 1.0 / count AS winPct
#     FROM read_parquet("ConsolidatedResults010.parquet")
#     WHERE
#         interceptZ = 1.80
#         AND interceptRow = 1
#         AND interceptCol = 14
#         AND opponentRow = 21
#         AND opponentCol = 9
#     ORDER BY count DESC, winPct DESC, avgPointShotCount ASC
# )
# TO 'ShotSelection/Shot_180_01_14_21_09.csv'
# (HEADER, DELIMITER ',')
# """
# query(sql)

# sql = """
# COPY (
#     SELECT
#         bounceCol,
#         bounceRow,

#         COUNT(*) AS count,
#         SUM(wins) AS wins,
#         SUM(wins) * 1.0 / COUNT(*) AS winPct
#     FROM read_parquet("ConsolidatedResults010.parquet")
#     WHERE
#         interceptCol = 6
#         AND interceptRow = 10
#         AND interceptZ = 1.0
#         AND opponentCol = 7
#         AND opponentRow = 18
#     GROUP BY
#         bounceCol,
#         bounceRow
#     ORDER BY count DESC, winPct DESC
# )
# TO 'ShotSelection/Shot_100_06_10_07_18.csv'
# (HEADER, DELIMITER ',')
# """
# query(sql)

# sql = """
# COPY (
#     SELECT
#         bounceRow,
#         bounceCol,
#         COUNT(*) AS count,
#         SUM(wins) * 1.0 / COUNT(*) AS winPct
#     FROM read_parquet("ConsolidatedResults010.parquet")
#     WHERE
#         interceptCol = 6
#         AND interceptRow = 18
#         AND interceptZ = 1.0
#         AND opponentCol = 7
#         AND opponentRow = 5
#         AND bounceRow BETWEEN 5 AND 13
#         AND bounceCol BETWEEN 5 AND 10
#     GROUP BY bounceRow, bounceCol
# )
# TO 'ShotSelection/Shot_100_06_18_07_20.csv'
# (HEADER, DELIMITER ',')
# """
# query(sql)

# sql = """
#     SELECT
#         MIN(interceptRow),
#         MAX(interceptRow)
#     FROM read_parquet("ConsolidatedResults010.parquet");
#     """
# interceptMinMax = query(sql)
# print(str(interceptMinMax))

# sql = """
# SELECT
#     MIN(opponentRow),
#     MAX(opponentRow)
# FROM read_parquet("ConsolidatedResults010.parquet");
# """
# opponentMinMax = query(sql)
# print(str(opponentMinMax))


# # Number of rows
# # Generation0 = 179,604,092
# # Generation1 = 159,096,064
# sql="""
# SELECT COUNT(*) AS num_rows_winpct_ge_50
# FROM read_parquet("ConsolidatedGeneration1.parquet")
# """
# winningRows = query(sql)
# print(str(winningRows))

# # Number of rows with greater than 10 entries
# # Generation0 = 499,410
# # Generation1 = 482,513
# sql="""
# SELECT COUNT(*) AS num_rows_winpct_ge_50
# FROM read_parquet("ConsolidatedGeneration1.parquet")
# WHERE
#     count >= 10
# """
# winningRows = query(sql)
# print(str(winningRows))

# # Number with Win percentage > 50%
# # Generation0 = 98,702,245
# # Generation1 = 87,409,349
# # Number with Win percentage > 50% and count > 1
# # Generation0 = 350,984
# # Generation1 = 315,955
# # Number with count > 1
# # Generation1 = 862,048
# # Number with count > 10
# # Generation1 = 486,073
# sql="""
# SELECT COUNT(*) AS num_rows_winpct_ge_50
# FROM read_parquet("Generation0/ConsolidatedGeneration0.parquet")
# WHERE
#     COUNT > 50
#     AND wins * 2 >= count 
# """
# winningRows = query(sql)
# print(str(winningRows))

# # Top 50% with greater than 10 entries
# # Generation0 = 133,613
# # Generation1 = 131,401
# sql="""
# SELECT COUNT(*) AS num_rows_winpct_ge_50
# FROM read_parquet("ConsolidatedGeneration1.parquet")
# WHERE
#     count >= 10
#     AND wins * 2 >= count 
# """
# winningRows = query(sql)
# print(str(winningRows))

# # Top 66% with greater than 10 entries
# # Generation0 = 366,690
# # Generation1 = 355,660
# sql="""
# SELECT COUNT(*) AS num_rows_winpct_ge_50
# FROM read_parquet("ConsolidatedGeneration1.parquet")
# WHERE
#     count >= 10
#     AND wins * 3 >= count 
# """
# winningRows = query(sql)
# print(str(winningRows))



