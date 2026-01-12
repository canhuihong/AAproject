import yfinance as yf
import pandas as pd
from src.config import PROXY_URL
import os

# Setup proxy
os.environ["HTTP_PROXY"] = PROXY_URL
os.environ["HTTPS_PROXY"] = PROXY_URL

def check_keys(ticker):
    print(f"\nChecking {ticker}...")
    try:
        t = yf.Ticker(ticker)
        bs = t.balance_sheet
        qbs = t.quarterly_balance_sheet
        
        if bs.empty:
            print("  Balance Sheet is Empty")
        else:
            print("  Balance Sheet Keys (first 10):", bs.index[:10].tolist())
            candidates = [k for k in bs.index if 'Share' in k or 'Ordinary' in k]
            print("  Share Candidates in Annual:", candidates)
            for c in candidates:
                print(f"    {c}: {bs.loc[c].values[:2]}") # Print first 2 values

        if qbs.empty:
            print("  Quarterly BS is Empty")
        else:
            candidates_q = [k for k in qbs.index if 'Share' in k or 'Ordinary' in k]
            print("  Share Candidates in Quarterly:", candidates_q)

    except Exception as e:
        print(f"  Error: {e}")

check_keys("AAPL")
check_keys("MSFT")
