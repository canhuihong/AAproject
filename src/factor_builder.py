import pandas as pd
import numpy as np
import logging
from src.data_manager import DataManager

logger = logging.getLogger("FactorBuilder")

class FactorBuilder:
    def __init__(self):
        self.db = DataManager()
        
    def get_full_universe_data(self, start_date=None):
        """
        获取全量数据：价格 + 财报
        并合并计算出: Return, MarketCap, Book-to-Market
        """
        # 1. 获取所有价格
        conn = self.db._get_conn()
        try:
            # 价格 (Price)
            query_price = "SELECT date, ticker, close FROM prices"
            if start_date:
                query_price += f" WHERE date >= '{start_date}'"
            df_price = pd.read_sql(query_price, conn)
            
            # 财报 (Fundamentals)
            # 我们需要 net_income (或者 total_equity?) 
            # Fama-French HML 使用 Book Equity / Market Equity
            # 所以我们需要 total_equity 和 shares_count
            query_fund = """
                SELECT date, ticker, total_equity, shares_count, total_assets, operating_income 
                FROM fundamentals
            """
            df_fund = pd.read_sql(query_fund, conn)
        except Exception as e:
            logger.error(f"DB Read Error: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
            
        if df_price.empty or df_fund.empty:
            return pd.DataFrame()
            
        df_price['date'] = pd.to_datetime(df_price['date'])
        df_fund['date'] = pd.to_datetime(df_fund['date'])
        
        # 2. 财报数据对齐 (Forward Fill)
        # 2.1 股价
        price_pivot = df_price.pivot(index='date', columns='ticker', values='close')
        all_dates = price_pivot.index
        
        # 2.2 权益 (BE)
        equity_pivot = df_fund.pivot(index='date', columns='ticker', values='total_equity').reindex(all_dates).ffill()
        
        # 2.3 股本 (Shares)
        shares_pivot = df_fund.pivot(index='date', columns='ticker', values='shares_count').reindex(all_dates).ffill()
        
        # 2.4 [FF5] 营业利润 (Op Inc)
        op_pivot = df_fund.pivot(index='date', columns='ticker', values='operating_income').reindex(all_dates).ffill()
        
        # 2.5 [FF5] 总资产 (Assets)
        asset_pivot = df_fund.pivot(index='date', columns='ticker', values='total_assets').reindex(all_dates).ffill()
        
        # 3. 变量计算
        ret_pivot = price_pivot.pct_change()
        mcap_pivot = price_pivot * shares_pivot
        bm_pivot = equity_pivot / mcap_pivot
        
        # [FF5] 盈利因子 (Operating Profitability = OpInc / BE)
        # 严格来说是 BE，也有用 Assets 的。这里用 BE 和 FF 定义尽量一致 (Book Equity)
        op_prof_pivot = op_pivot / equity_pivot
        
        # [FF5] 投资因子 (Asset Growth = d(Assets) / Assets)
        # 使用 252 交易日同比
        asset_growth_pivot = asset_pivot.pct_change(periods=252)
        
        return ret_pivot, mcap_pivot, bm_pivot, op_prof_pivot, asset_growth_pivot

    def build_factors(self, start_date='2018-01-01'):
        """
        核心构建逻辑 (FF5)
        """
        logger.info("🏗️  Constructing Fama-French 5 Factors from local DB...")
        
        data = self.get_full_universe_data(start_date)
        if isinstance(data, pd.DataFrame) and data.empty:
            return pd.DataFrame()
            
        ret, mcap, bm, op_prof, inv = data
        
        factors = []
        
        # 辅助函数：计算市值加权收益
        def calc_ret(grp):
            if grp.empty: return 0.0
            return (grp['ret'] * grp['mcap']).sum() / grp['mcap'].sum()

        for date in ret.index[1:]:
            # 当日切片
            try:
                r = ret.loc[date]
                mc = mcap.loc[date]
                b = bm.loc[date]
                op = op_prof.loc[date]
                iv = inv.loc[date]
            except KeyError:
                continue
            
            # 对齐
            valid = pd.concat([r, mc, b, op, iv], axis=1, join='inner')
            valid.columns = ['ret', 'mcap', 'bm', 'op_prof', 'inv']
            
            valid.dropna(inplace=True)
            if len(valid) < 10: continue
                
            # 1. Market
            total_mcap = valid['mcap'].sum()
            mkt_ret = (valid['ret'] * valid['mcap']).sum() / total_mcap
            
            # 2. SMB
            median_size = valid['mcap'].median()
            small = valid[valid['mcap'] <= median_size]
            big = valid[valid['mcap'] > median_size]
            smb = calc_ret(small) - calc_ret(big)
            
            # 3. HML (Value vs Growth)
            p30_bm = valid['bm'].quantile(0.3)
            p70_bm = valid['bm'].quantile(0.7)
            hml = calc_ret(valid[valid['bm'] >= p70_bm]) - calc_ret(valid[valid['bm'] <= p30_bm])
            
            # 4. RMW (Robust vs Weak Profitability)
            p30_op = valid['op_prof'].quantile(0.3)
            p70_op = valid['op_prof'].quantile(0.7)
            rmw = calc_ret(valid[valid['op_prof'] >= p70_op]) - calc_ret(valid[valid['op_prof'] <= p30_op])
            
            # 5. CMA (Conservative vs Aggressive Investment)
            # Conservative = Low Investment (Low Growth)
            p30_inv = valid['inv'].quantile(0.3)
            p70_inv = valid['inv'].quantile(0.7)
            cma = calc_ret(valid[valid['inv'] <= p30_inv]) - calc_ret(valid[valid['inv'] >= p70_inv])
            
            factors.append({
                'Date': date,
                'Mkt-RF': mkt_ret - (0.04/252),
                'SMB': smb,
                'HML': hml,
                'RMW': rmw,
                'CMA': cma,
                'RF': (0.04/252)
            })
            
        if not factors:
            logger.warning("No factors generated. Check data density.")
            return pd.DataFrame()
            
        df_factors = pd.DataFrame(factors).set_index('Date')
        logger.info(f"✅ FF5 Factors constructed! ({len(df_factors)} days)")
        
        return df_factors
