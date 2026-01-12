import yfinance as yf
import pandas as pd
import datetime
import time
from tqdm import tqdm
import logging
import requests
import os
import random
from io import StringIO

# 引入配置
# 确保你的 src/config.py 里已经有了 SP500_LIMIT, SP600_LIMIT 这些定义
from src.config import DATA_DIR, ETF_BLOCKLIST, PROXY_URL, DB_PATH, SP500_LIMIT, SP600_LIMIT, SP400_LIMIT, NASDAQ_LIMIT, RFR_TICKER
from src.data_manager import DataManager

# 详细的日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InitData")

# Silencing yfinance noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

def get_tickers_from_wiki(url, name):
    """【爬虫】从维基百科获取代码 + 板块 (稳健版)"""
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
        
        df = None
        target_col = None
        sector_col = None
        
        # 1. Data Cleaning / Column Detection
        # 我们需要同时找到 Ticker 和 Sector
        ticker_candidates = ['Symbol', 'Ticker', 'Ticker symbol', 'Ticker Symbol']
        sector_candidates = ['GICS Sector', 'Sector', 'GICS Sector', 'Industry'] # Wikipedia 常用列名
        
        for table in tables:
            # Check Ticker
            found_ticker = None
            for cand in ticker_candidates:
                if cand in table.columns:
                    found_ticker = cand
                    break
            
            # Check Sector (Current logic: MUST find ticker, Sector is optional but preferred)
            found_sector = None
            for cand in sector_candidates:
                if cand in table.columns:
                    found_sector = cand
                    break
            
            if found_ticker:
                df = table
                target_col = found_ticker
                sector_col = found_sector # Might be None
                break
                
        if df is None:
            logger.warning(f"⚠️ Could not find explicit Ticker column for {name}, trying first table...")
            df = tables[0]
            target_col = df.columns[0]
            
        # 2. Extract Data
        # Returns list of dict: [{'ticker': 'AAPL', 'sector': 'Technology'}, ...]
        results = []
        
        garbage_list = [
            'CONSTITUENTS', 'EXCHANGES', 'SYMBOL', 'TICKER', 'SECURITY', 'COMPANY', 'GICS SECTOR', 
            'FOUNDATION', 'OPERATOR', 'TYPE', 'WEBSITE'
        ]
        
        for idx, row in df.iterrows():
            t = str(row[target_col])
            
            # Basic Cleaning
            t = t.replace('.', '-').replace('$', '').strip()
            
            # Garbage Filter
            if t.upper() in garbage_list: continue
            if len(t) > 5 and not t.isalpha(): continue 
            if not t: continue
            
            # Sector
            sec = "Unknown"
            if sector_col and sector_col in row:
                sec = str(row[sector_col]).strip()
            
            results.append({'ticker': t, 'sector': sec})
            
        logger.info(f"✅ Successfully fetched {len(results)} items for {name}")
        return results
        
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
            
            # 【CRITICAL FIX】防止请求当天的还没产生的数据
            # 如果 next_day >= 今天，说明昨天的已经有了，今天的还没收盘 -> 跳过
            if next_day.date() >= datetime.datetime.now().date():
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
        
        # 只有在确实需要下载时才联网
        hist = pd.DataFrame()
        
        # Retry Logic (3 Attempts)
        for attempt in range(3):
            try:
                if start_date:
                    hist = obj.history(start=start_date, auto_adjust=True)
                else:
                    hist = obj.history(period=download_period, auto_adjust=True)
                
                if not hist.empty:
                    break
                
                # If empty, maybe rate limited? Wait a bit
                time.sleep(2 * (attempt + 1))
            except Exception as e:
                logger.warning(f"⚠️ Retry {attempt+1}/3 failed for {ticker}: {e}")
                time.sleep(3 * (attempt + 1))
            
        if not hist.empty:
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            
            records = []
            for d, row in hist.iterrows():
                records.append((d.strftime('%Y-%m-%d'), ticker, row['Close'], row['Volume']))
            db.save_prices(records)
        
        # Benchmark 或 增量更新无数据时，直接返回
        if is_benchmark or ticker == RFR_TICKER: return 1
        if start_date and hist.empty: return 1

        # ==========================================
        # C. 财报下载 (Fundamentals) - MERGED MODE
        # ==========================================
        def extract_fundamentals(fin_df, bs_df):
            """Helper to extract common dates and metrics"""
            if fin_df.empty or bs_df.empty: return []
            
            common = fin_df.columns.intersection(bs_df.columns)
            recs = []
            
            # Fetch shares once
            shares = obj.info.get('sharesOutstanding')
            if not shares: return []

            for date in common:
                try:
                    ni = fin_df.loc['Net Income', date] if 'Net Income' in fin_df.index else 0
                    rev = fin_df.loc['Total Revenue', date] if 'Total Revenue' in fin_df.index else 0
                    
                    # [New for FF5] Operating Income (RMW)
                    # Try 'Operating Income' or 'EBIT'
                    op_inc = 0
                    if 'Operating Income' in fin_df.index:
                        op_inc = fin_df.loc['Operating Income', date]
                    elif 'EBIT' in fin_df.index:
                        op_inc = fin_df.loc['EBIT', date]
                        
                    eq = 0
                    for k in ['Stockholders Equity', 'Total Stockholder Equity', 'Total Equity']:
                        if k in bs_df.index:
                            eq = bs_df.loc[k, date]
                            break
                            
                    # [New for FF5] Total Assets (CMA)
                    assets = 0
                    if 'Total Assets' in bs_df.index:
                        assets = bs_df.loc['Total Assets', date]
                    
                    # 60天前视偏差防护
                    eff_date = date + datetime.timedelta(days=60)
                    if eff_date > datetime.datetime.now(): continue
                    
                    recs.append((
                        eff_date.strftime('%Y-%m-%d'), 
                        ticker, 
                        float(ni), float(eq), float(rev), float(shares), 
                        date.strftime('%Y-%m-%d'),
                        float(assets),       # New
                        float(op_inc)        # New
                    ))
                except Exception:
                    continue
            return recs

        # 1. Get Both Sets
        q_recs = extract_fundamentals(obj.quarterly_financials, obj.quarterly_balance_sheet)
        a_recs = extract_fundamentals(obj.financials, obj.balance_sheet)
        
        # 2. Merge & Deduplicate (Prefer Quarterly if date conflict? Actually dates usually differ)
        # Use a dict to dedup by report_date
        combined = {}
        for r in a_recs + q_recs:
             # r[-1] is report_date
             combined[r[-1]] = r
             
        fund_recs = list(combined.values())
            
        if fund_recs:
            db.save_fundamentals(fund_recs)
            return 1 # 更新成功

    except Exception:
        # 捕获所有网络异常，防止单个股票中断整个流程
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

    # 2.1 强制检查 Risk Free Rate (^IRX)
    print("\n-------- Checking Risk Free Rate (^IRX) --------")
    rfr_status = process_single_stock(RFR_TICKER, db, existing_map.get(RFR_TICKER), is_benchmark=True)
    if rfr_status == 1:
        print("✅ RFR Data Updated.")
    else:
        print("⚠️ RFR Update Failed (Check Network).")

    # 3. 抓取正股名单 + 板块信息
    sp500_raw = get_tickers_from_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "S&P 500")
    if SP500_LIMIT is not None:
        print(f"🚧 Test Mode: Limiting S&P 500 to first {SP500_LIMIT} stocks.")
        sp500_raw = sp500_raw[:SP500_LIMIT]

    sp600_raw = get_tickers_from_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "S&P 600")
    sp400_raw = get_tickers_from_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "S&P 400") # MidCap
    nasdaq_raw = get_tickers_from_wiki("https://en.wikipedia.org/wiki/Nasdaq-100", "NASDAQ 100")
    
    # Merge Phase
    # Use a dict to dedup by ticker, keeping the first non-Unknown sector found
    merged_map = {}
    
    for item in sp500_raw + sp600_raw + sp400_raw + nasdaq_raw:
        t = item['ticker']
        s = item['sector']
        
        if t in ETF_BLOCKLIST: continue
        
        # If new or updating Unknown sector
        if t not in merged_map:
            merged_map[t] = s
        elif merged_map[t] == 'Unknown' and s != 'Unknown':
            merged_map[t] = s
            
    final_tickers = sorted(list(merged_map.keys()))
    
    # 3.1 保存 Sector 信息到数据库
    print(f"💾 Saving Sector Info for {len(final_tickers)} stocks...")
    now_str = datetime.datetime.now().strftime('%Y-%m-%d')
    info_records = []
    for t in final_tickers:
        info_records.append((t, merged_map[t], None, now_str)) # (ticker, sector, industry, date)
    
    db.save_stock_info(info_records)
    
    print(f"\n🎯 Total Targets: {len(final_tickers)} stocks")
    print("-" * 60)
    
    # 4. 批量执行
    counts = {'Skip':0, 'Upd':0, 'Fail':0}
    pbar = tqdm(final_tickers, unit="stock")
    
    for i, ticker in enumerate(pbar):
        last_date = existing_map.get(ticker)
        
        status = process_single_stock(ticker, db, last_update_date=last_date)
        
        if status == 0: counts['Skip'] += 1
        elif status == 1: counts['Upd'] += 1
        else: counts['Fail'] += 1
        
        pbar.set_postfix(counts)
        
        # 动态限流
        if status == 1:
            time.sleep(random.uniform(0.3, 0.7)) 
            # 每 50 个请求多歇会
            if counts['Upd'] % 50 == 0:
                time.sleep(2.0)

    print("\n" + "="*60)
    print("✅ PROCESS COMPLETED!")
    print(f"   ⏭️  Skipped (Fresh):    {counts['Skip']}")
    print(f"   ⬇️  Downloaded (New):   {counts['Upd']}")
    print(f"   ⚠️  Failed/Error:       {counts['Fail']}")
    print("="*60)

if __name__ == "__main__":
    main()