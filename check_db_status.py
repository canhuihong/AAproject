import sqlite3
import pandas as pd
from src.config import DB_PATH

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("Checking DB Status...")

# Check Prices count
cursor.execute("SELECT COUNT(*) FROM prices")
print(f"Total Prices Rows: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(DISTINCT ticker) FROM prices")
print(f"Unique Tickers with Prices: {cursor.fetchone()[0]}")

# Check Fundamentals count and non-zero shares
cursor.execute("SELECT COUNT(*) FROM fundamentals")
total_fund = cursor.fetchone()[0]
print(f"Total Fundamentals Rows: {total_fund}")

cursor.execute("SELECT COUNT(DISTINCT ticker) FROM fundamentals")
print(f"Unique Tickers with Fundamentals: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM fundamentals WHERE shares_count > 0")
valid_shares = cursor.fetchone()[0]
print(f"Rows with Valid Shares: {valid_shares}")

# Show a sample
print("\nSample Fundamentals:")
df = pd.read_sql("SELECT * FROM fundamentals ORDER BY date DESC LIMIT 5", conn)
print(df[['date', 'ticker', 'shares_count', 'net_income']].to_string())

conn.close()
