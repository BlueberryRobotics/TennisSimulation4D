import duckdb
import argparse
import os

def ExportToCsv(parquetPath, csvPath="preview_output.csv", limit=10):
    # Connect
    conn = duckdb.connect()

    # Query first N rows
    query = f"""
        SELECT *
        FROM read_parquet('{parquetPath}')
        LIMIT {limit}
    """

    # Execute
    df = conn.execute(query).df()

    # Save to CSV
    df.to_csv(csvPath, index=False)

    print(f"\nExported first {limit} rows to CSV:")
    print(f"  {os.path.abspath(csvPath)}\n")

    # Show in terminal (optional)
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export first rows of Parquet to CSV.")
    parser.add_argument("parquetPath", help="Path to Parquet file")
    parser.add_argument("--limit", type=int, default=500, help="Number of rows to export")
    parser.add_argument("--csv", type=str, default="preview_output.csv",
                        help="Output CSV filename")

    args = parser.parse_args()

    ExportToCsv(args.parquetPath, args.csv, args.limit)