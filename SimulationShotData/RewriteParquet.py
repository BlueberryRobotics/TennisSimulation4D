import pandas as pd

df = pd.read_parquet("shots_20260331_142106.parquet", engine="pyarrow")
df.to_parquet("rewritten_file.parquet", engine="pyarrow")