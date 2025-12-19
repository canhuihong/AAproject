import pandas as pd
import numpy as np
import logging
from src.data_manager import DataManager
from src.config import FULL_BLOCKLIST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FactorEngine")

class FactorEngine:
    def __init__(self):
        self.db = DataManager()
        
    def get_price_history(self, end_date):
        """获取用于计算动量的历史价格"""
        conn = self.db._get_conn()
        try:
            # 优化：只取需要的列
            query = f"SELECT date, ticker, close FROM prices WHERE date <= '{end_date}'"
            df = pd.read_sql(query, conn)
        except Exception as e:
            logger.error(f"Error reading prices: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
        
        if df.empty: return df
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values(['ticker', 'date'])

    def calculate_momentum(self, analysis_date, lookback_days=252):
        """计算动量因子 (12个月)"""
        df = self.get_price_history(analysis_date)
        if df.empty: return pd.Series(dtype=float)

        def get_mom(x):
            # 至少要有半年数据才算动量
            if len(x) < 120: return np.nan
            
            # 取一年前的价格，如果不够一年就取最早的
            start_price = x.iloc[-lookback_days] if len(x) >= lookback_days else x.iloc[0]
            end_price = x.iloc[-1]
            
            if start_price <= 0: return np.nan
            return (end_price / start_price) - 1

        return df.groupby('ticker')['close'].apply(get_mom).rename("factor_mom_raw")

    def get_scored_universe(self, analysis_date=None):
        """
        核心打分逻辑
        """
        if not analysis_date:
            analysis_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        # 1. 获取截面原始数据 (新版：包含 net_income 等)
        df_raw = self.db.get_cross_section_data(analysis_date)
        if df_raw.empty: return pd.DataFrame()
        
        df_raw.set_index('ticker', inplace=True)
        
        # 数据类型清洗
        cols = ['close', 'net_income', 'total_equity', 'shares_count']
        for c in cols:
            df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')
        
        # 2. 现场计算因子 (On-the-fly Calculation)
        # 市值
        df_raw['market_cap'] = df_raw['close'] * df_raw['shares_count']
        
        # 估值 (PE倒数) = Net Income / Market Cap
        # 简单年化：季度利润 * 4
        df_raw['net_income_annual'] = df_raw['net_income'] * 4
        
        # 过滤亏损股和资不抵债股
        df_raw = df_raw[df_raw['net_income_annual'] > 0]
        df_raw = df_raw[df_raw['total_equity'] > 0]
        
        # 计算具体指标
        df_raw['pe_ratio'] = df_raw['market_cap'] / df_raw['net_income_annual']
        df_raw['factor_value'] = 1.0 / df_raw['pe_ratio']  # E/P
        df_raw['factor_quality'] = df_raw['net_income_annual'] / df_raw['total_equity'] # ROE
        
        # 3. 合并动量
        mom_series = self.calculate_momentum(analysis_date)
        df = df_raw.join(mom_series)
        
        # 4. 黑名单过滤与去极值
        df = df[~df.index.isin(FULL_BLOCKLIST)]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['factor_value', 'factor_quality', 'factor_mom_raw'], inplace=True)
        
        if df.empty: return pd.DataFrame()

        # 5. 标准化打分 (Z-Score)
        factors = ['factor_value', 'factor_quality', 'factor_mom_raw']
        for f in factors:
            mean = df[f].mean()
            std = df[f].std()
            # 避免除以0
            if std == 0: std = 1
            
            # 简单的 Z-Score
            df[f+'_z'] = (df[f] - mean) / std
            
        # 6. 综合得分
        df['final_score'] = (
            0.4 * df['factor_value_z'] +
            0.3 * df['factor_quality_z'] +
            0.3 * df['factor_mom_raw_z']
        )
        
        return df.sort_values('final_score', ascending=False)