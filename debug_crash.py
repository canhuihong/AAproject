# debug_crash.py
import logging
import pandas as pd
from src.data_manager import DataManager
from src.factor_engine import FactorEngine
from src.optimizer import PortfolioOptimizer

logging.basicConfig(level=logging.INFO)

def check_system():
    print("\n🔍 --- STEP 1: Database Check ---")
    db = DataManager()
    tickers = db.get_all_tickers()
    print(f"✅ Total Tickers in DB: {len(tickers)}")
    if len(tickers) < 10:
        print("❌ ERROR: Not enough data. Run init_data.py again.")
        return

    print("\n🔍 --- STEP 2: Factor Engine Check ---")
    fe = FactorEngine()
    try:
        # 尝试取今天的排名
        ranks = fe.get_scored_universe()
        print(f"✅ Factor Engine ran successfully.")
        print(f"   Top stock: {ranks.index[0] if not ranks.empty else 'NONE'}")
        print(f"   Total scored: {len(ranks)}")
    except Exception as e:
        print(f"❌ CRASH in FactorEngine: {e}")
        import traceback
        traceback.print_exc()
        return

    if ranks.empty:
        print("⚠️ Warning: Factor Engine returned empty DataFrame.")
        return

    print("\n🔍 --- STEP 3: Optimizer Check ---")
    top_10 = ranks.head(10).index.tolist()
    print(f"   Optimizing for: {top_10}")
    
    try:
        # 尝试获取最近一天的数据
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        opt = PortfolioOptimizer(top_10, analysis_date=today)
        weights = opt.optimize_sharpe_ratio()
        print(f"✅ Optimizer result:\n{weights}")
    except Exception as e:
        print(f"❌ CRASH in Optimizer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_system()