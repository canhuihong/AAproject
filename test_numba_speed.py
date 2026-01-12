import time
import pandas as pd
import numpy as np
from src.factor_engine import FactorEngine
from src.numba_engine import run_parallel_ols

def test_speed():
    engine = FactorEngine()
    print("Initializing Engine and fetching data...")
    
    # Run for a known date, e.g., last Friday
    analysis_date = '2025-01-09' 
    
    # Calling get_scored_universe which now uses Numba internally
    t0 = time.time()
    df = engine.get_scored_universe(analysis_date=analysis_date)
    t1 = time.time()
    
    print(f"\nTime Taken: {t1-t0:.4f} seconds")
    print(f"Stocks Scored: {len(df)}")
    
    if not df.empty:
        print("\nTop 5 Results:")
        print(df[['alpha', 'beta_mkt', 'close']].head())
    else:
        print("No results returned.")

if __name__ == "__main__":
    test_speed()
