import pandas as pd
import numpy as np
import logging
import yfinance as yf
from src.data_manager import DataManager

logger = logging.getLogger("FactorBuilder")

class FactorBuilder:
    def __init__(self):
        self.db = DataManager()
        
    def _fetch_risk_free_rate(self, start_date=None, end_date=None):
        """获取动态无风险利率 (^IRX)"""
        try:
            # 下载 13-week Treasury Bill
            irx = yf.download("^IRX", start=start_date, end=end_date, progress=False, auto_adjust=True)
            if irx.empty: return pd.Series()
            
            # yfinance 返回的是百分比 (e.g. 4.5 -> 4.5%)
            # 转为日度小数: (4.5 / 100) / 252
            close_rates = irx['Close']
            if isinstance(close_rates, pd.DataFrame):
                close_rates = close_rates.squeeze()
            
            rf_daily = (close_rates / 100) / 252
            return rf_daily
        except Exception as e:
            logger.error(f"Failed to fetch RF: {e}")
            return pd.Series()
        
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
        # [FF5] 投资因子 (Asset Growth = d(Assets) / Assets)
        # 使用 252 交易日同比
        asset_growth_pivot = asset_pivot.pct_change(periods=252)
        
        # [FF6] 动量因子 (Momentum = Return (t-12 -> t-1))
        # 逻辑：过去12个月的累计收益，剔除最近1个月 (252 - 21 = 231天窗口，滞后21天)
        # P(t-21) / P(t-252) - 1
        p_lag1 = price_pivot.shift(21)
        p_lag12 = price_pivot.shift(252)
        p_lag1 = price_pivot.shift(21)
        p_lag12 = price_pivot.shift(252)
        mom_pivot = (p_lag1 / p_lag12) - 1.0

        # [FF6 Update] Fetch Dynamic RF
        start_dt = price_pivot.index[0]
        end_dt = price_pivot.index[-1]
        rf_daily = self._fetch_risk_free_rate(start_dt, end_dt)
        # Reindex to match universe dates (ffill for holidays)
        rf_daily = rf_daily.reindex(price_pivot.index).ffill().fillna(0.04/252)

        return ret_pivot, mcap_pivot, bm_pivot, op_prof_pivot, asset_growth_pivot, mom_pivot, rf_daily

    def build_factors(self, start_date='2018-01-01'):
        """
        核心构建逻辑 (FF5) - 升级版：月度重平衡
        """
        logger.info("🏗️  Constructing Fama-French 5 Factors (Monthly Rebalancing)...")
        
        data = self.get_full_universe_data(start_date)
        if isinstance(data, pd.DataFrame) and data.empty:
            return pd.DataFrame()
            
        if isinstance(data, pd.DataFrame) and data.empty:
            return pd.DataFrame()
            
        ret, mcap, bm, op_prof, inv, mom, rf_series = data
        
        factors = []
        
        # 辅助函数：计算市值加权收益
        def calc_ret(daily_ret_df, weights):
            # weights is a Series of mcap at formation date
            # daily_ret_df is a DataFrame (Time x Stocks) for the month
            # Align weights to columns
            common = daily_ret_df.columns.intersection(weights.index)
            if len(common) == 0: return pd.Series(0.0, index=daily_ret_df.index)
            
            w = weights[common]
            r = daily_ret_df[common]
            
            # Weighted average per day
            return (r * w).sum(axis=1) / w.sum()

        # 1. 获取所有月份的结束日期
        # 我们使用重采样找到每个月的最后一天 (大致)
        # 注意：ret 的 index 是交易日
        month_groups = ret.groupby(pd.Grouper(freq='M'))
        
        for month_end_dt, group in month_groups:
            if group.empty: continue
            
            # 2. 确定 "Formation Date" (上个月的最后一天)
            # month_end_dt 是这个月的最后一天 (比如 1月31日)，group 是 1月的数据
            # 我们需要用 *上个月底* 的数据来构建 1月的组合
            # 由于数据全部是对齐的，我们可以直接找 group 第一天之前的那个有效交易日
            
            first_day_of_month = group.index[0]
            # 找到全量数据里，在 first_day 之前的最近一天
            prev_days = mcap.index[mcap.index < first_day_of_month]
            
            if prev_days.empty:
                # 如果没有前一天（比如数据的第一个月），则无法构建因子（不知道谁是大盘谁是小盘）
                continue
                
            formation_date = prev_days[-1]
            
            # 3. 获取 Formation Date 的截面数据
            try:
                mc = mcap.loc[formation_date]
                b = bm.loc[formation_date]
                op = op_prof.loc[formation_date]
                iv = inv.loc[formation_date]
            except KeyError:
                continue

            # 4. 构建组合 (Sorting)
            # 我们需要把所有 signal 拼在一起，注意：如果有任何一个因子确实，该股票就会被 drop
            # 对于 FF6，如果缺少 Mom 数据，也会被剔除。这是标准做法。
            valid = pd.concat([mc, b, op, iv, mom.loc[formation_date]], axis=1, join='inner')
            valid.columns = ['mcap', 'bm', 'op_prof', 'inv', 'mom']
            valid.dropna(inplace=True)
            
            if len(valid) < 10: continue

            # --- Sorts ---
            
            # Size Split (Median)
            median_size = valid['mcap'].median()
            small_mask = valid['mcap'] <= median_size
            big_mask = valid['mcap'] > median_size
            
            # BM Split (30/70)
            p30_bm = valid['bm'].quantile(0.3)
            p70_bm = valid['bm'].quantile(0.7)
            value_mask = valid['bm'] >= p70_bm
            growth_mask = valid['bm'] <= p30_bm
            
            # Op Split
            p30_op = valid['op_prof'].quantile(0.3)
            p70_op = valid['op_prof'].quantile(0.7)
            robust_mask = valid['op_prof'] >= p70_op
            weak_mask = valid['op_prof'] <= p30_op
            
            # Inv Split
            # Inv Split
            p30_inv = valid['inv'].quantile(0.3)
            p70_inv = valid['inv'].quantile(0.7)
            consv_mask = valid['inv'] <= p30_inv
            aggr_mask = valid['inv'] >= p70_inv
            
            # Mom Split
            p30_mom = valid['mom'].quantile(0.3)
            p70_mom = valid['mom'].quantile(0.7)
            high_mom_mask = valid['mom'] >= p70_mom
            low_mom_mask = valid['mom'] <= p30_mom
            
            # 5. 计算当月每一天的因子收益
            # 注意：在这个月内，Constituents 不变，Weight (shares) 也不变
            # 但 Value Weight 的 'Value' (Market Cap) 每天会随股价变动？
            # 简化版 FF：通常使用 Formation Date 的 Market Cap 作为权重固定一个月，或者每月根据上月市值重置权重
            # 这里我们使用 Formation Date 的 Mcap 作为这个月的固定权重
            
            curr_month_ret = group # DataFrame: Dates x Tickers
            w = valid['mcap']      # Series: Tickers (Fixed for month)
            
            # Factor 1: Market
            mkt = calc_ret(curr_month_ret, w)
            
            # Factor 2: SMB
            r_small = calc_ret(curr_month_ret, w[small_mask])
            r_big = calc_ret(curr_month_ret, w[big_mask])
            smb = r_small - r_big
            
            # Factor 3: HML
            r_val = calc_ret(curr_month_ret, w[value_mask])
            r_gro = calc_ret(curr_month_ret, w[growth_mask])
            hml = r_val - r_gro
            
            # Factor 4: RMW
            r_rob = calc_ret(curr_month_ret, w[robust_mask])
            r_weak = calc_ret(curr_month_ret, w[weak_mask])
            rmw = r_rob - r_weak
            
            # Factor 5: CMA
            r_con = calc_ret(curr_month_ret, w[consv_mask])
            r_agg = calc_ret(curr_month_ret, w[aggr_mask])
            cma = r_con - r_agg

            # Factor 6: MOM (Up - Down)
            # Sm-Hi, Sm-Lo, Big-Hi, Big-Lo
            # MOM = (Small-High + Big-High)/2 - (Small-Low + Big-Low)/2
            # 注意：MOM 也是控制了 Size 的
            
            # Intersection masks
            sh_mask = small_mask & high_mom_mask
            sl_mask = small_mask & low_mom_mask
            bh_mask = big_mask & high_mom_mask
            bl_mask = big_mask & low_mom_mask
            
            r_sh = calc_ret(curr_month_ret, w[sh_mask])
            r_sl = calc_ret(curr_month_ret, w[sl_mask])
            r_bh = calc_ret(curr_month_ret, w[bh_mask])
            r_bl = calc_ret(curr_month_ret, w[bl_mask])
            
            mom_factor = (r_sh + r_bh) / 2 - (r_sl + r_bl) / 2
            
            # 组合 DataFrame
            # 获取当月每一天的 RF 值
            curr_rf = rf_series.reindex(curr_month_ret.index).fillna(0.04/252)
            
            month_df = pd.DataFrame({
                'Mkt-RF': mkt - curr_rf,
                'SMB': smb,
                'HML': hml,
                'RMW': rmw,
                'CMA': cma,
                'MOM': mom_factor,
                'RF': curr_rf
            })
            
            factors.append(month_df)
            
        if not factors:
            logger.warning("No factors generated.")
            return pd.DataFrame()
            
        df_factors = pd.concat(factors).sort_index()
        logger.info(f"✅ FF5 Factors constructed! ({len(df_factors)} days)")
        
        return df_factors
