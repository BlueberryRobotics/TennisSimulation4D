import duckdb

db = duckdb.connect()

PARQUET = "ConsolidatedResults010.parquet"

def query(sql):
    return db.execute(sql).fetchdf()


sql = """
SELECT COUNT(*) AS num_trajectories
FROM read_parquet("ConsolidatedGeneration03.parquet")
"""
totalTrajs = query(sql)
print(str(totalTrajs))

sql = """
SELECT COUNT(*) AS more_than_five
FROM read_parquet("ConsolidatedResults010.parquet")
WHERE count > 10
"""
numSingletons = query(sql)
print(str(numSingletons))

# Top 100 by winning percentage
# sql = """
# SELECT
#     *,
#     wins * 1.0 / count AS winPct
# FROM read_parquet("ConsolidatedResults010.parquet")
# WHERE count >= 10
# ORDER BY winPct DESC
# LIMIT 100
# """
# topWinningRows = query(sql)
# print(topWinningRows)

# Top 1000 by number of wins
# sql = """
#     COPY (
#         SELECT
#             *,
#             wins * 1.0 / count AS winPct
#         FROM read_parquet("ConsolidatedResults010.parquet")
#         WHERE interceptZ != 2.7
#         ORDER BY wins DESC
#         LIMIT 5000
#     )
#     TO 'top_notServe_5000_by_wins.csv'
#     (HEADER, DELIMITER ',')
# """

# query(sql)

# sql = """
#     COPY (
#         SELECT
#             *,
#             wins * 1.0 / count AS winPct
#         FROM read_parquet("ConsolidatedResults010.parquet")
#         WHERE winPct = 1
#         ORDER BY wins DESC
#     )
#     TO 'winners.csv'
#     (HEADER, DELIMITER ',')
# """

# query(sql)

# number of winners including serves
sql = """
SELECT COUNT(*) AS perfect_tactics_including_serves
FROM read_parquet("ConsolidatedResults010.parquet")
WHERE wins = count  
"""

perfectTactics = query(sql)
print(perfectTactics.map("{:,}".format))

# number of winners not including serves
sql = """
SELECT COUNT(*) AS perfect_tactics_not_serves
FROM read_parquet("ConsolidatedResults010.parquet")
WHERE wins = count AND interceptZ != 2.7
"""

numWinners = query(sql)
print(numWinners.map("{:,}".format))

# count of the number of serves
sql = """
SELECT COUNT(*) AS num_serves
FROM read_parquet("ConsolidatedResults010.parquet")
WHERE
    interceptZ = 2.7
    AND interceptRow = 5
    AND opponentRow = 23
    AND bounceRow BETWEEN 14 AND 18
    AND (
        (interceptCol = 6 AND opponentCol = 10 AND bounceCol BETWEEN 7 AND 10)
        OR
        (interceptCol = 9 AND opponentCol = 5 AND bounceCol BETWEEN 5 AND 8)
    );
"""
serves = query(sql)
print(serves.map("{:,}".format))

# top winners that are not serves
# sql= """
# COPY(
#     SELECT
#         *,
#         wins * 1.0 / count AS winPct
#     FROM read_parquet("ConsolidatedResults010.parquet")
#     WHERE
#         NOT (
#             interceptZ = 2.7
#             AND interceptRow = 5
#             AND opponentRow = 23
#             AND bounceRow BETWEEN 14 AND 18
#             AND (
#                 (interceptCol = 6 AND opponentCol = 10 AND bounceCol BETWEEN 7 AND 10)
#                 OR
#                 (interceptCol = 9 AND opponentCol = 5 AND bounceCol BETWEEN 5 AND 8)
#             )
#         )
#     ORDER BY wins DESC
#     LIMIT 1000
# )
# TO 'NonServiceWinners.csv'
# (HEADER, DELIMITER ',')
# """
# query(sql)

# top winners that are not serves
# sql= """
# COPY(
#     SELECT
#         *,
#         wins * 1.0 / count AS winPct
#     FROM read_parquet("ConsolidatedResults010.parquet")
#     WHERE
#         NOT (
#             interceptZ = 2.7
#             AND interceptRow = 5
#             AND opponentRow = 23
#             AND bounceRow BETWEEN 14 AND 18
#             AND (
#                 (interceptCol = 6 AND opponentCol = 10 AND bounceCol BETWEEN 7 AND 10)
#                 OR
#                 (interceptCol = 9 AND opponentCol = 5 AND bounceCol BETWEEN 5 AND 8)
#             )
#         )
#     ORDER BY count DESC
#     LIMIT 10000
# )
# TO 'NonServiceCounts.csv'
# (HEADER, DELIMITER ',')
# """
# query(sql)

sql="""
SELECT COUNT(*) AS num_nonserve_winpct_ge_50
FROM read_parquet("ConsolidatedGeneration1.parquet")
WHERE
    wins * 2 >= count
    AND NOT (
        interceptZ = 2.7
        AND interceptRow = 5
        AND opponentRow = 23
        AND bounceRow BETWEEN 14 AND 17
        AND (
            (interceptCol = 6 AND opponentCol = 10 AND bounceCol BETWEEN 8 AND 10)
            OR
            (interceptCol = 9 AND opponentCol = 5 AND bounceCol BETWEEN 5 AND 7)
        )
    );
"""
nonServeWin50Percent = query(sql)
print(nonServeWin50Percent.map("{:,}".format))

sql="""
SELECT COUNT(*) AS num_all_winpct_ge_50
FROM read_parquet("ConsolidatedGeneration1.parquet")
WHERE
    wins * 2 >= count
"""
allWin50Percent = query(sql)
print(allWin50Percent.map("{:,}".format))