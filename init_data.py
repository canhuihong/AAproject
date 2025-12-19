import yfinance as yf
import pandas as pd
import datetime
import time
import logging
import requests
import os
from io import StringIO

# 进度条兼容性处理
try:
    from tqdm import tqdm
except ImportError:
    print("建议安装 tqdm: pip install tqdm")
    def tqdm(iterable, desc=""): return iterable

from src.config import DATA_DIR, ETF_BLOCKLIST, PROXY_URL, DB_PATH
from src.data_manager import DataManager

# 详细的日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InitData")

def get_tickers_from_wiki(url, name):
    """【爬虫】从维基百科获取代码 (稳健版)"""
    logger.info(f"🌐 Crawling {name} from Wikipedia...")
    
    # 1. 设置完整的请求头 (伪装成浏览器)
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
        
        # 解析表格
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        
        # 兼容不同的列名写法
        col_name = 'Symbol' if 'Symbol' in df.columns else 'Ticker symbol'
        if col_name not in df.columns:
            col_name = df.columns[0] # 盲猜第一列
            
        # 清洗代码 (把 BRK.B 转为 BRK-B 以适配 Yahoo)
        tickers = df[col_name].astype(str).str.replace('.', '-', regex=False).tolist()
        
        logger.info(f"✅ Successfully fetched {len(tickers)} tickers for {name}")
        return tickers
        
    except Exception as e:
        logger.error(f"❌ Failed to scrape {name}: {e}")
        return []

def process_single_stock(ticker, db, last_update_date=None, is_benchmark=False):
    """
    【下载核心】处理单个股票 (含断点续传、周末跳过、财报清洗)
    返回状态码：0=跳过, 1=更新, -1=失败
    """
    try:
        # ==========================================
        # A. 智能跳过判断 (Smart Skip)
        # ==========================================
        download_period = "5y" # 默认下载长度
        start_date = None
        
        if last_update_date:
            last_dt = datetime.datetime.strptime(last_update_date, '%Y-%m-%d')
            today_dt = datetime.datetime.now()
            days_diff = (today_dt - last_dt).days
            
            # 1. 极速检查：24小时内更新过 -> 绝对跳过
            if days_diff < 1:
                return 0 
            
            # 2. 周末豁免：今天是周末且数据只滞后1-2天 -> 跳过
            # (周六=5, 周日=6)
            if today_dt.weekday() >= 5 and days_diff <= 2: 
                return 0

            # 否则，设置增量下载的起始日期
            next_day = last_dt + datetime.timedelta(days=1)
            start_date = next_day.strftime('%Y-%m-%d')
            download_period = None 

        # ==========================================
        # B. 价格下载 (Price Data)
        # ==========================================
        obj = yf.Ticker(ticker)
        
        # 只有在确实需要下载时才联网
        if start_date:
            hist = obj.history(start=start_date, auto_adjust=True)
        else:
            hist = obj.history(period=download_period, auto_adjust=True)
            
        if not hist.empty:
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            
            records = []
            for d, row in hist.iterrows():
                # 存入数据库
                records.append((d.strftime('%Y-%m-%d'), ticker, row['Close'], row['Volume']))
            db.save_prices(records)
        
        # 如果是Benchmark，不查财报，直接返回成功
        if is_benchmark: return 1

        # 如果增量更新时没下到价格(例如休市)，通常也无需查财报，节省时间
        if start_date and hist.empty: return 1

        # ==========================================
        # C. 财报下载 (Fundamentals)
        # ==========================================
        # 使用 quarterly_financials 获取更灵敏的季度数据
        fin = obj.quarterly_financials
        bs = obj.quarterly_balance_sheet
        
        # 兜底：如果季度没数据，试下年度
        if fin.empty: fin = obj.financials
        if bs.empty: bs = obj.balance_sheet
        
        # 还没数据？那就算了
        if fin.empty or bs.empty: return -1
        
        common_dates = fin.columns.intersection(bs.columns)
        shares = obj.info.get('sharesOutstanding')
        
        if not shares or len(common_dates) == 0: return -1

        fund_recs = []
        for date in common_dates:
            try:
                # 提取关键字段，使用 .get 避免 KeyError
                ni = fin.loc['Net Income', date] if 'Net Income' in fin.index else 0
                rev = fin.loc['Total Revenue', date] if 'Total Revenue' in fin.index else 0
                
                # 权益字段可能有变种
                eq = 0
                for k in ['Stockholders Equity', 'Total Stockholder Equity', 'Total Equity']:
                    if k in bs.index:
                        eq = bs.loc[k, date]
                        break
                
                # 60天前视偏差防护
                eff_date = date + datetime.timedelta(days=60)
                if eff_date > datetime.datetime.now(): continue
                
                fund_recs.append((
                    eff_date.strftime('%Y-%m-%d'), # 数据的可用日期
                    ticker, 
                    float(ni), float(eq), float(rev), float(shares), 
                    date.strftime('%Y-%m-%d')      # 原始报告期
                ))
            except Exception:
                continue
            
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
    print("🚀 QML Reborn: Robust Update Mode (Weekends Safe)")
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
    sp600 = get_tickers_from_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "S&P 600")
    
    full_list = sorted(list(set(sp500 + sp600)))
    final_list = [t for t in full_list if t not in ETF_BLOCKLIST]
    
    print(f"\n🎯 Total Targets: {len(final_list)} stocks")
    print("-" * 60)
    
    # 4. 批量执行 (带计数器)
    counts = {'Skip':0, 'Upd':0, 'Fail':0}
    pbar = tqdm(final_list, unit="stock")
    
    for i, ticker in enumerate(pbar):
        last_date = existing_map.get(ticker)
        
        status = process_single_stock(ticker, db, last_update_date=last_date)
        
        if status == 0: counts['Skip'] += 1
        elif status == 1: counts['Upd'] += 1
        else: counts['Fail'] += 1
        
        # 实时更新进度条后缀
        pbar.set_postfix(counts)
        
        # 【恢复】简单的限流逻辑，防止 Yahoo 封禁
        # 只有在发生真实网络请求(Upd)时才 sleep，Skip 时不 sleep
        if status == 1:
            time.sleep(0.05) 
            # 每 100 个请求歇口气
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