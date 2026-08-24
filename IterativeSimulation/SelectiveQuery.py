import duckdb

conn = duckdb.connect()

PARQUET = "Gen14Reference.parquet"
OUTPUT = "Gen14ReferenceExploration.csv"

query = f"""
WITH base AS (
    SELECT *
    FROM read_parquet('{PARQUET}')
    WHERE wins > 0 AND count >= 10
),

context_bounce_stats AS (
    SELECT
        interceptCol,
        interceptRow,
        interceptZ,
        opponentCol,
        opponentRow,
        bounceCol,
        bounceRow,
        SUM(wins) AS totalWins,
        SUM(count) AS totalCount,
        SUM(wins) * 1.0 / SUM(count) AS winPct
    FROM base
    GROUP BY
        interceptCol, interceptRow, interceptZ,
        opponentCol, opponentRow,
        bounceCol, bounceRow
),

-- Winning bounce cells (random 3 per context)
winning_cells AS (
    SELECT *
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY
                    interceptCol, interceptRow, interceptZ,
                    opponentCol, opponentRow
                ORDER BY RANDOM()
            ) AS rand_rank
        FROM context_bounce_stats
        WHERE winPct > 0.5
    )
    WHERE rand_rank <= 5
),

-- Losing bounce cells (random 1 per context)
losing_cells AS (
    SELECT *
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY
                    interceptCol, interceptRow, interceptZ,
                    opponentCol, opponentRow
                ORDER BY RANDOM()
            ) AS rand_rank
        FROM context_bounce_stats
        WHERE winPct <= 0.5
    )
    WHERE rand_rank <= 1
),

-- Combine both
selected_cells AS (
    SELECT * FROM winning_cells
    UNION ALL
    SELECT * FROM losing_cells
),

-- Join back to trajectics
joined AS (
    SELECT
        b.*,
        (b.wins * 1.0 / b.count) AS winPct
    FROM base b
    JOIN selected_cells c
        ON b.interceptCol = c.interceptCol
        AND b.interceptRow = c.interceptRow
        AND b.interceptZ = c.interceptZ
        AND b.opponentCol = c.opponentCol
        AND b.opponentRow = c.opponentRow
        AND b.bounceCol = c.bounceCol
        AND b.bounceRow = c.bounceRow
),

-- One trajectic per apexHeight
apex_dedup AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY
                interceptCol, interceptRow, interceptZ,
                opponentCol, opponentRow,
                bounceCol, bounceRow,
                apexHeight
            ORDER BY count DESC, winPct DESC
        ) AS apex_rank
    FROM joined
),

apex_unique AS (
    SELECT *
    FROM apex_dedup
    WHERE apex_rank = 1
),

-- Limit per bounce cell
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY
                interceptCol, interceptRow, interceptZ,
                opponentCol, opponentRow,
                bounceCol, bounceRow
            ORDER BY count DESC, winPct DESC
        ) AS trajectic_rank
    FROM apex_unique
)

SELECT *
FROM ranked
WHERE trajectic_rank <= 3
"""
count = conn.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]
print(f"Row count: {count:,}")

conn.execute(f"""
COPY ({query})
TO '{OUTPUT}'
(HEADER, DELIMITER ',')
""")

print(f"Export complete → {OUTPUT}")