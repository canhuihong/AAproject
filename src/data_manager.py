import sqlite3
import pandas as pd
import logging
from src.config import DB_PATH

logger = logging.getLogger("QML.DataManager")

class DataManager:
    def __init__(self):
        self.db_path = DB_PATH
        self._initialize_db()

    def _get_conn(self):
        # 允许跨线程共享连接 (对应多线程下载场景)
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _initialize_db(self):
        """初始化数据库表结构 (Schema V4 - Added stock_info)"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 1. 价格表 (保持原样)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            date TEXT,
            ticker TEXT,
            close REAL,
            volume REAL,
            PRIMARY KEY (date, ticker)
        )
        ''')

        # 2. 基本面表 (重大升级：存储原始财报数据)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fundamentals (
            date TEXT,
            ticker TEXT,
            net_income REAL,      -- 净利润 (核心)
            total_equity REAL,    -- 股东权益 (核心)
            total_revenue REAL,   -- 营收
            shares_count REAL,    -- 流通股本
            report_period TEXT,   -- 原始财报日期
            total_assets REAL,    -- [FF5新增] 总资产 (用于 CMA)
            operating_income REAL,-- [FF5新增] 营业利润 (用于 RMW)
            operating_cash_flow REAL, -- [New] 经营性现金流 (用于 Accruals)
            long_term_debt REAL,  -- [Factor2.0] 长期债务 (用于 Leverage Delta)
            total_current_assets REAL, -- [Factor2.0] 流动资产 (用于 Current Ratio)
            total_current_liabilities REAL, -- [Factor2.0] 流动负债
            gross_profit REAL,    -- [Factor2.0] 毛利 (用于 Margin Delta)
            capital_expenditure REAL, -- [Factor2.0] 资本开支 (用于 FCF)
            PRIMARY KEY (date, ticker)
        )
        ''')
        
        # 3. 股票信息表 (Sector/Industry)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_info (
            ticker TEXT PRIMARY KEY,
            sector TEXT,
            industry TEXT,
            last_updated TEXT
        )
        ''')
        
        # 简单迁移逻辑：逐个尝试添加新列
        migrations = [
            "ALTER TABLE fundamentals ADD COLUMN total_assets REAL",
            "ALTER TABLE fundamentals ADD COLUMN operating_income REAL",
            "ALTER TABLE fundamentals ADD COLUMN operating_cash_flow REAL",
            "ALTER TABLE fundamentals ADD COLUMN long_term_debt REAL",
            "ALTER TABLE fundamentals ADD COLUMN total_current_assets REAL",
            "ALTER TABLE fundamentals ADD COLUMN total_current_liabilities REAL",
            "ALTER TABLE fundamentals ADD COLUMN gross_profit REAL",
            "ALTER TABLE fundamentals ADD COLUMN capital_expenditure REAL"
        ]
        for sql in migrations:
            try:
                cursor.execute(sql)
            except:
                pass # 列已存在则忽略

        conn.commit()
        conn.close()

    def save_prices(self, records):
        """批量保存价格"""
        if not records: return
        conn = self._get_conn()
        try:
            conn.executemany(
                'INSERT OR REPLACE INTO prices (date, ticker, close, volume) VALUES (?, ?, ?, ?)',
                records
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Save prices failed: {e}")
        finally:
            conn.close()

    def save_fundamentals(self, records):
        """批量保存基本面 (V3结构)"""
        if not records: return
        conn = self._get_conn()
        try:
            conn.executemany(
                '''INSERT OR REPLACE INTO fundamentals 
                   (date, ticker, net_income, total_equity, total_revenue, shares_count, report_period, 
                    total_assets, operating_income, operating_cash_flow,
                    long_term_debt, total_current_assets, total_current_liabilities, gross_profit, capital_expenditure) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                records
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Save fundamentals failed: {e}")
        finally:
            conn.close()
            
    def get_latest_dates_map(self):
        """【新增】全库扫描：获取每只股票的最新日期 (用于断点续传)"""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT ticker, MAX(date) FROM prices GROUP BY ticker")
            return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to scan DB dates: {e}")
            return {}
        finally:
            conn.close()

    def get_cross_section_data(self, date_str=None):
        """
        【核心】获取截面数据
        这里逻辑比旧版复杂：它需要做 GroupBy 找到截止日期的最新财报
        """
        conn = self._get_conn()
        
        # 1. 确定日期
        if not date_str:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(date) FROM prices")
            res = cursor.fetchone()
            date_str = res[0] if res else None
            
        if not date_str: 
            conn.close()
            return pd.DataFrame()

        # 2. 获取当天的价格
        price_query = f"SELECT ticker, close FROM prices WHERE date = '{date_str}'"
        df_price = pd.read_sql(price_query, conn)
        
        # 3. 获取截止该日期最新的财报 (Pit-in-Time Logic)
        fund_query = f'''
        SELECT 
            f.ticker, 
            f.net_income, 
            f.total_equity, 
            f.total_revenue,
            f.shares_count,
            f.total_assets,
            f.operating_income,
            f.operating_cash_flow,
            f.long_term_debt,
            f.total_current_assets,
            f.total_current_liabilities,
            f.gross_profit,
            f.capital_expenditure
        FROM fundamentals f
        INNER JOIN (
            SELECT ticker, MAX(date) as max_date
            FROM fundamentals
            WHERE date <= '{date_str}'
            GROUP BY ticker
        ) latest ON f.ticker = latest.ticker AND f.date = latest.max_date
        '''
        df_fund = pd.read_sql(fund_query, conn)
        
        # [Factor 2.0] Fetch Lagged Fundamentals (1 Year Ago) for Piotroski Delta
        # Logic: Find latest report <= date_str - 365 days
        import datetime
        dt_curr = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        dt_prev = dt_curr - datetime.timedelta(days=365)
        prev_date_str = dt_prev.strftime('%Y-%m-%d')
        
        prev_fund_query = f'''
        SELECT 
            f.ticker, 
            f.net_income as prev_net_income, 
            f.total_assets as prev_total_assets,
            f.long_term_debt as prev_long_term_debt,
            f.total_current_assets as prev_total_current_assets,
            f.total_current_liabilities as prev_total_current_liabilities,
            f.shares_count as prev_shares_count,
            f.gross_profit as prev_gross_profit,
            f.total_revenue as prev_total_revenue
        FROM fundamentals f
        INNER JOIN (
            SELECT ticker, MAX(date) as max_date
            FROM fundamentals
            WHERE date <= '{prev_date_str}'
            GROUP BY ticker
        ) latest ON f.ticker = latest.ticker AND f.date = latest.max_date
        '''
        try:
            df_prev = pd.read_sql(prev_fund_query, conn)
        except Exception:
            df_prev = pd.DataFrame()

        conn.close()
        
        # 4. Merge Current + Lagged
        df_merged = pd.merge(df_fund, df_prev, on='ticker', how='left')
        
        # 5. Merge with Price
        return pd.merge(df_price, df_merged, on='ticker', how='left')

    def save_stock_info(self, records):
        """批量保存股票板块信息 (records: list of tuples/dicts)"""
        # records format: [(ticker, sector, industry, updated_date), ...]
        if not records: return
        conn = self._get_conn()
        try:
            conn.executemany(
                'INSERT OR REPLACE INTO stock_info (ticker, sector, industry, last_updated) VALUES (?, ?, ?, ?)',
                records
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Save stock_info failed: {e}")
        finally:
            conn.close()

    def get_sector_map(self):
        """获取全市场板块映射 {ticker: sector}"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT ticker, sector FROM stock_info")
            return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to get sector map: {e}")
            return {}
        finally:
            conn.close()

    def get_risk_free_rate_series(self, start_date=None, end_date=None):
        """
        获取无风险利率序列 (Daily)
        Source: ^IRX (13-week T-bill yield index)
        Calculation: Yield / 100 / 252
        Fallback: 0.04 / 252 (4% annual)
        """
        from src.config import RFR_TICKER
        
        conn = self._get_conn()
        try:
            # 1. Fetch IRX data
            query = f"SELECT date, close FROM prices WHERE ticker = '{RFR_TICKER}'"
            if start_date: query += f" AND date >= '{start_date}'"
            if end_date: query += f" AND date <= '{end_date}'"
            
            df = pd.read_sql(query, conn)
            
            if df.empty:
                logger.warning(f"No data found for {RFR_TICKER}. Using default 4.0% RFR.")
                return pd.Series(dtype=float) # Return empty to trigger fallback downstream
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # 2. Convert Index Value to Daily Rate
            # ^IRX is quoted in annualized percentage (e.g., 4.25 means 4.25%)
            # Daily Rate = (Close / 100) / 252
            daily_rfr = (df['close'] / 100.0) / 252.0
            
            return daily_rfr
            
        except Exception as e:
            logger.error(f"Failed to get RFR: {e}")
            return pd.Series(dtype=float)
        finally:
            conn.close()

    def check_fundamental_health(self, ticker):
        """
        【新增】检查基本面数据是否完整 (Factor 2.0 迁移检查)
        检查最新的一条记录，看新字段 (long_term_debt) 是否为空
        Returns: True (Healthy), False (Needs Update)
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            # Fetch the latest report
            cursor.execute(f"SELECT long_term_debt FROM fundamentals WHERE ticker = '{ticker}' ORDER BY date DESC LIMIT 1")
            row = cursor.fetchone()
            if not row:
                return True # No data at all, nothing to 'fix' (or treat as healthy to avoid loop)
                # Actually if no data, we might need download. But price update handles that.
                # Here we care about EXISTING partial data.
            
            val = row[0]
            if val is None:
                return False # Missing new field!
            return True
        except Exception:
            return True # Fallback
        finally:
            conn.close()