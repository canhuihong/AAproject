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
        """初始化数据库表结构 (Schema V2)"""
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
        # 即使代码短了，是因为我们不再存储计算好的 PE/PB，而是存原始数据
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fundamentals (
            date TEXT,
            ticker TEXT,
            net_income REAL,      -- 净利润 (核心)
            total_equity REAL,    -- 股东权益 (核心)
            total_revenue REAL,   -- 营收
            shares_count REAL,    -- 流通股本
            report_period TEXT,   -- 原始财报日期
            PRIMARY KEY (date, ticker)
        )
        ''')
        
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
        """批量保存基本面 (V2结构)"""
        if not records: return
        conn = self._get_conn()
        try:
            conn.executemany(
                '''INSERT OR REPLACE INTO fundamentals 
                   (date, ticker, net_income, total_equity, total_revenue, shares_count, report_period) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
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
            f.shares_count
        FROM fundamentals f
        INNER JOIN (
            SELECT ticker, MAX(date) as max_date
            FROM fundamentals
            WHERE date <= '{date_str}'
            GROUP BY ticker
        ) latest ON f.ticker = latest.ticker AND f.date = latest.max_date
        '''
        df_fund = pd.read_sql(fund_query, conn)
        conn.close()
        
        # 4. 合并返回
        return pd.merge(df_price, df_fund, on='ticker', how='left')