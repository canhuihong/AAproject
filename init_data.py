import yfinance as yf
import pandas as pd
import datetime
import time
from tqdm import tqdm
import logging
import requests
import os
from io import StringIO

# 引入配置
# 确保你的 src/config.py 里已经有了 SP500_LIMIT, SP600_LIMIT 这些定义
from src.config import DATA_DIR, ETF_BLOCKLIST, PROXY_URL, DB_PATH, SP500_LIMIT, SP600_LIMIT, SP400_LIMIT, NASDAQ_LIMIT
from src.data_manager import DataManager

# 详细的日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InitData")

def get_tickers_from_wiki(url, name):
    """【爬虫】从维基百科获取代码 (稳健版)"""
    logger.info(f"🌐 Crawling {name} from Wikipedia...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/91.0.4472.124 Safari/537.36"
    }
    
    proxies = {
        "http": os.environ.get("HTTP_PROXY", PROXY_URL),
        "https": os.environ.get("HTTPS_PROXY", PROXY_URL)
    }
    
    try:
        response = requests.get(url, headers=headers, proxies=proxies, timeout=20)
        response.raise_for_status()
        
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        
        col_name = 'Symbol' if 'Symbol' in df.columns else 'Ticker symbol'
        if col_name not in df.columns:
            col_name = df.columns[0]
            
        raw_tickers = df[col_name].astype(str).tolist()
        
        cleaned_tickers = []
        garbage_list = [
            'CONSTITUENTS', 'EXCHANGES', 'SYMBOL', 'TICKER', 'SECURITY', 'COMPANY', 'GICS SECTOR', 
            'FOUNDATION', 'OPERATOR', 'TYPE', 'WEBSITE'
        ]
        
        for t in raw_tickers:
            # 1. Basic Cleaning
            t = t.replace('.', '-').replace('$', '').strip()
            
            # 2. Garbage Filter
            if t.upper() in garbage_list: continue
            if len(t) > 5 and not t.isalpha(): continue # Skip weird long strings
            if not t: continue
            
            cleaned_tickers.append(t)
            
        tickers = cleaned_tickers
        
        logger.info(f"✅ Successfully fetched {len(tickers)} tickers for {name}")
        return tickers
        
    except Exception as e:
        logger.error(f"❌ Failed to scrape {name}: {e}")
        return []

def process_single_stock(ticker, db, last_update_date=None, is_benchmark=False):
    """
    【下载核心】处理单个股票
    升级点：混合下载年度(Financials)和季度(Quarterly)财报，解决历史数据不足问题
    """
    try:
        # ==========================================
        # A. 智能跳过判断 (Smart Skip)
        # ==========================================
        # 为了修复数据缺失，建议第一次运行时先把这里改短，或者直接删掉库重跑
        # 这里保留 10y 的长度以确保覆盖 2021 年的回测需求
        download_period = "10y" 
        start_date = None
        
        if last_update_date:
            last_dt = datetime.datetime.strptime(last_update_date, '%Y-%m-%d')
            today_dt = datetime.datetime.now()
            days_diff = (today_dt - last_dt).days
            
            # 极速检查
            if days_diff < 1:
                return 0 
            
            # 周末豁免
            if today_dt.weekday() >= 5 and days_diff <= 2: 
                return 0

            # 增量更新
            next_day = last_dt + datetime.timedelta(days=1)
            # 如果下一天就是今天，且还没收盘(简单判断)，可能取不到数据，建议跳过
            if next_day.date() == today_dt.date():
                 # 简单策略：如果还没过下午5点(美股收盘)，就不强求更新今天的数据
                 if today_dt.hour < 17:
                     return 0

            start_date = next_day.strftime('%Y-%m-%d')
            download_period = None 

        # Santize ticker
        original_ticker = ticker
        ticker = ticker.replace('$', '').strip() 
        if original_ticker != ticker:
            logger.info(f"🔧 Sanitized ticker: {original_ticker} -> {ticker}")

        # ==========================================
        # B. 价格下载 (Price Data)
        # ==========================================
        # logger.debug(f"Processing: {ticker}")
        obj = yf.Ticker(ticker)

        # 【新增修复】 检查拆股 (Splits)
        # 如果上次更新后发生了拆股，必须全量重下，否则价格不连续
        if start_date:
            try:
                splits = obj.splits
                if not splits.empty:
                    # 找到最近一次拆股时间
                    last_split_date = splits.index.max().to_pydatetime()
                    last_db_date = datetime.datetime.strptime(last_update_date, '%Y-%m-%d')
                    
                    # 如果拆股发生在上次更新之后，或者就是同一天，强制重跑
                    if last_split_date >= last_db_date:
                        logger.info(f"🔄 Split detected for {ticker} on {last_split_date.date()}. Forcing full redownload.")
                        start_date = None
                        download_period = "10y"
            except Exception:
                pass # 获取拆股数据失败，安全起见按原计划跑 (或者也可以选择强制重跑，这里先保守)
        
        if start_date:
            hist = obj.history(start=start_date, auto_adjust=True)
        else:
            hist = obj.history(period=download_period, auto_adjust=True)
            
        if not hist.empty:
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            
            records = []
            for d, row in hist.iterrows():
                records.append((d.strftime('%Y-%m-%d'), ticker, row['Close'], row['Volume']))
            db.save_prices(records)
        
        # Benchmark 或 增量更新无数据时，直接返回
        if is_benchmark: return 1
        if start_date and hist.empty: return 1

        # ==========================================
        # C. 财报下载 (Fundamentals) - 混合增强版
        # ==========================================
        # 同时抓取 .quarterly_financials (灵敏，近1年) 和 .financials (年度，近4年)
        
        q_fin = obj.quarterly_financials
        q_bs = obj.quarterly_balance_sheet
        a_fin = obj.financials
        a_bs = obj.balance_sheet
        
        # 没有任何数据则退出
        if (q_fin.empty and a_fin.empty) or (q_bs.empty and a_bs.empty):
            return -1

        # 辅助函数：解析数据框并提取为 list of tuples
        def extract_data(fin_df, bs_df):
            if fin_df.empty or bs_df.empty: return []
            
            common_dates = fin_df.columns.intersection(bs_df.columns)
            shares = obj.info.get('sharesOutstanding')
            
            if not shares or len(common_dates) == 0: return []

            recs = []
            for date in common_dates:
                try:
                    # 提取 Net Income 和 Revenue，容错处理
                    ni = fin_df.loc['Net Income', date] if 'Net Income' in fin_df.index else 0
                    rev = fin_df.loc['Total Revenue', date] if 'Total Revenue' in fin_df.index else 0
                    
                    # 权益字段可能有变种
                    eq = 0
                    for k in ['Stockholders Equity', 'Total Stockholder Equity', 'Total Equity']:
                        if k in bs_df.index:
                            eq = bs_df.loc[k, date]
                            break

                    # [FF5新增] 总资产 (用于 CMA)
                    assets = bs_df.loc['Total Assets', date] if 'Total Assets' in bs_df.index else 0
                    
                    # [FF5新增] 营业利润 (用于 RMW)
                    op_inc = 0
                    for k in ['Operating Income', 'Operating Profit', 'EBIT']:
                        if k in fin_df.index:
                            op_inc = fin_df.loc[k, date]
                            break
                    
                    # 60天前视偏差防护 (Pit-in-Time Lag)
                    # 假设财报发布日 = 报告期 + 60天
                    eff_date = date + datetime.timedelta(days=60)
                    if eff_date > datetime.datetime.now(): continue
                    
                    recs.append((
                        eff_date.strftime('%Y-%m-%d'), # 数据可用日期 (用于回测)
                        ticker, 
                        float(ni), float(eq), float(rev), float(shares), 
                        date.strftime('%Y-%m-%d'),      # 原始报告期
                        float(assets), float(op_inc)    # [FF5新增]
                    ))
                except Exception:
                    continue
            return recs

        # 分别提取
        q_records = extract_data(q_fin, q_bs)
        a_records = extract_data(a_fin, a_bs)
        
        # 合并 (REPLACE INTO 会自动去重)
        all_records = q_records + a_records
        
        if all_records:
            db.save_fundamentals(all_records)
            return 1 # 更新成功

    except Exception:
        return -1

    return 1

def main():
    db = DataManager()
    
    print("\n" + "="*60)
    print("🚀 QML Reborn: Robust Update Mode (Hybrid Fundamentals)")
    print("📢 Version: With Ticker Sanitization Fix (No $)")
    print("="*60)

    # 1. 扫描现状
    print("📊 Scanning existing database...")
    existing_map = db.get_latest_dates_map()
    print(f"✅ Found {len(existing_map)} stocks already in DB.")

    # 2. 强制检查 Benchmark (SPY)
    print("\n-------- Checking Benchmark (SPY) --------")
    spy_status = process_single_stock('SPY', db, existing_map.get('SPY'), is_benchmark=True)
    if spy_status == 0:
        print("⏭️  SPY is up-to-date (Skipped).")
    elif spy_status == 1:
        print("✅ SPY Data Updated.")
    else:
        print("⚠️ SPY Update Failed (Check Network).")

    # 3. 抓取正股名单
    sp500 = get_tickers_from_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "S&P 500")
    if SP500_LIMIT is not None:
        print(f"🚧 Test Mode: Limiting S&P 500 to first {SP500_LIMIT} stocks.")
        sp500 = sp500[:SP500_LIMIT]

    sp600 = get_tickers_from_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "S&P 600")
    if SP600_LIMIT is not None:
        print(f"🚧 Test Mode: Limiting S&P 600 to first {SP600_LIMIT} stocks.")
        sp600 = sp600[:SP600_LIMIT]

    # [新增] S&P 400 MidCap
    sp400 = get_tickers_from_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "S&P 400")
    if SP400_LIMIT is not None:
        print(f"🚧 Test Mode: Limiting S&P 400 to first {SP400_LIMIT} stocks.")
        sp400 = sp400[:SP400_LIMIT]

    # [新增] Nasdaq 100
    nasdaq = get_tickers_from_wiki("https://en.wikipedia.org/wiki/Nasdaq-100", "Nasdaq 100")
    if NASDAQ_LIMIT is not None:
        print(f"🚧 Test Mode: Limiting Nasdaq 100 to first {NASDAQ_LIMIT} stocks.")
        nasdaq = nasdaq[:NASDAQ_LIMIT]
    
    full_list = sorted(list(set(sp500 + sp600 + sp400 + nasdaq)))
    final_list = [t for t in full_list if t not in ETF_BLOCKLIST]
    
    print(f"\n🎯 Total Targets: {len(final_list)} stocks")
    print("-" * 60)
    
    # 4. 批量执行
    counts = {'Skip':0, 'Upd':0, 'Fail':0}
    pbar = tqdm(final_list, unit="stock")
    
    for i, ticker in enumerate(pbar):
        last_date = existing_map.get(ticker)
        
        status = process_single_stock(ticker, db, last_update_date=last_date)
        
        if status == 0: counts['Skip'] += 1
        elif status == 1: counts['Upd'] += 1
        else: counts['Fail'] += 1
        
        pbar.set_postfix(counts)
        
        # 动态限流
        if status == 1:
            time.sleep(0.05) 
            if counts['Upd'] % 100 == 0:
                time.sleep(0.5)

    print("\n" + "="*60)
    print("✅ PROCESS COMPLETED!")
    print(f"   ⏭️  Skipped (Fresh):    {counts['Skip']}")
    print(f"   ⬇️  Downloaded (New):   {counts['Upd']}")
    print(f"   ⚠️  Failed/Error:       {counts['Fail']}")
    print("="*60)

if __name__ == "__main__":
    main()