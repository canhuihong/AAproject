import pandas as pd
import numpy as np
import logging
import pandas_datareader.data as web
import statsmodels.api as sm
from src.data_manager import DataManager
from src.config import FULL_BLOCKLIST, FF_CACHE_PATH, PROXY_URL, DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FactorEngine")

class FactorEngine:
    def __init__(self):
        self.db = DataManager()
        self.ff_factors = None
        
    def fetch_ff_factors(self):
        """
        获取 Fama-French 3因子数据 (自建版)
        """
        from src.factor_builder import FactorBuilder
        
        # 也可以加缓存逻辑
        if FF_CACHE_PATH.exists():
             df = pd.read_csv(FF_CACHE_PATH, index_col=0, parse_dates=True)
             # 简单的过期检查：如果最近一天太久远，就重算 (可选)
             if (pd.Timestamp.now() - df.index[-1]).days < 5:
                # df = df[~df.index.duplicated(keep='first')] 
                # 这里不需要重复检查了，builder生成的肯定是干净的，但保留也可
                logger.info(f"📂 Loaded FF Factors from cache ({len(df)} rows)")
                return df
                 
        # 现场构建
        builder = FactorBuilder()
        df = builder.build_factors(start_date='2018-01-01')
        
        if not df.empty:
            df.to_csv(FF_CACHE_PATH)
            
        return df

    def get_price_history_all(self, end_date):
        """一次性获取所有股票的历史价格 (优化版)"""
        # 为了保证有足够的窗口做回归，我们取 2 年的数据 (approx 504 trading days)
        start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=730)).strftime('%Y-%m-%d')
        
        conn = self.db._get_conn()
        try:
            # 只取需要的字段，且只取还在截面里的股票？这里为了简单，取全量
            query = f"SELECT date, ticker, close FROM prices WHERE date >= '{start_date}' AND date <= '{end_date}'"
            df = pd.read_sql(query, conn)
            df['date'] = pd.to_datetime(df['date'])
            # 这里的 pivot 可能会消耗内存，但对几百只股票还好
            return df.pivot(index='date', columns='ticker', values='close')
        except Exception as e:
            logger.error(f"Error reading prices: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def calculate_alpha(self, stock_returns, ff_data, min_obs=126):
        """
        核心回归逻辑
        Rx - Rf = Alpha + b1*(Rm-Rf) + b2*SMB + b3*HML + epsilon
        """
        # 1. 索引对齐 (Inner Join)
        if not stock_returns.index.is_unique:
            stock_returns = stock_returns[~stock_returns.index.duplicated(keep='first')]
        if not ff_data.index.is_unique:
            ff_data = ff_data[~ff_data.index.duplicated(keep='first')]
            
        # 防止 Ticker 名字与因子名字 (如 RF) 冲突
        stock_returns.name = "StockRet"
            
        # axis=1 join，自动对其日期
        data = pd.concat([stock_returns, ff_data], axis=1, join='inner').dropna()
        
        if len(data) < min_obs:
            return -np.inf, None  # 数据太少，直接置为负无穷
        
        # 2. 准备 Y 和 X
        # Y: 股票超额收益 (Ri - Rf)
        Y = data['StockRet'] - data['RF']
        
        # X: 因子 (Mkt-RF, SMB, HML, RMW, CMA)
        # 兼容性检查：如果新因子存在则加入回归
        factors = ['Mkt-RF', 'SMB', 'HML']
        if 'RMW' in data.columns: factors.append('RMW')
        if 'CMA' in data.columns: factors.append('CMA')
        
        X = data[factors]
        X = sm.add_constant(X)
        
        try:
            model = sm.OLS(Y, X).fit()
            alpha = model.params['const']
            
            # 年化 Alpha (252天)
            # 我们通常比较 年化Alpha，更直观
            alpha_annual = (1 + alpha) ** 252 - 1
            
            # 也可以返回 t-stat 看显著性
            # t_alpha = model.tvalues['const']
            
            return alpha_annual, model
        except Exception:
            return -np.inf, None

    def get_scored_universe(self, analysis_date=None, top_n=10):
        """
        主流程: 
        1. 获取 FF 因子
        2. 获取所有股票价格 -> 算日收益率
        3. 循环跑回归 -> 算出 Alpha
        4. 排序返回
        """
        if not analysis_date:
            analysis_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
        logger.info(f"⚙️  Starting FF Alpha selection (FF5 Model) for {analysis_date}...")

        # 1. 准备因子
        ff_factors = self.fetch_ff_factors()
        if ff_factors.empty:
            logger.error("FF factors unavailable. Implementation Aborted.")
            return pd.DataFrame()
            
        # 截取到分析日
        ff_factors = ff_factors[ff_factors.index <= analysis_date]

        # 2. 准备股票收益率
        prices_df = self.get_price_history_all(analysis_date)
        if prices_df.empty:
            return pd.DataFrame()
            
        # 计算日收益率 (过滤极端值 - 使用 Winsorization)
        # 1. 计算 1% 和 99% 分位数 (针对整个截面或时间序列，这里简单对整个 DataFrame 做处理)
        # 注意：每一列是一只股票，我们在时间维度上并没有太大意义做 winsorize，
        # 但这里是全量数据的预处理。更精细的做法是每天做截面 winsorize
        # 这里为了效率，先计算 returns
        returns_df = prices_df.pct_change()
        
        # 2. 截面 Winsorization (按天)
        # 对于每天的数据，将超过 1% / 99% 的值压缩到边界
        # 这是一个 Pandas 这种 apply 操作可能会慢，但比循环快
        def winsorize_series(s, lower=0.01, upper=0.99):
            if s.empty: return s
            q_low = s.quantile(lower)
            q_high = s.quantile(upper)
            return s.clip(lower=q_low, upper=q_high)

        # Apply winsorization row-by-row (axis=1) -> Cross-sectional
        returns_df = returns_df.apply(winsorize_series, axis=1)
        
        # 3. 再次过滤掉无效行
        returns_df = returns_df.dropna(how='all')
        
        results = []
        
        # 3. 逐个回归 (这里可以优化用 GroupBy Apply 或者矩阵运算，但循环更直观)
        tickers = returns_df.columns
        total = len(tickers)
        
        # 可以在生产环境加 tqdm，这里为了日志清爽简单打 print
        from tqdm import tqdm
        
        valid_count = 0
        for ticker in tqdm(tickers, desc="Regressing"):
            if ticker in FULL_BLOCKLIST: continue
            
            series = returns_df[ticker].dropna()
            if series.empty: continue
            
            try:
                # 提高最小观测数据量到 126 (半年)
                alpha, model = self.calculate_alpha(series, ff_factors, min_obs=126)
                
                # 过滤条件
                # 1. alpha > -1.0 (非负无穷)
                # 2. alpha < 5.0 (年化 500% 以上通常是伪回归)
                if alpha > -1.0 and alpha < 5.0: 
                    # 我们同时保存 Beta (作为参考)
                    # 我们同时保存 Beta (作为参考)
                    beta_mkt = model.params.get('Mkt-RF', 0)
                    beta_smb = model.params.get('SMB', 0)
                    beta_hml = model.params.get('HML', 0)
                    beta_rmw = model.params.get('RMW', 0) # [FF5]
                    beta_cma = model.params.get('CMA', 0) # [FF5]
                    r_squared = model.rsquared
                    
                    results.append({
                        'ticker': ticker,
                        'final_score': alpha, # 将 Alpha 作为最终得分
                        'alpha_annual': alpha,
                        'beta_mkt': beta_mkt,
                        'beta_smb': beta_smb,
                        'beta_hml': beta_hml,
                        'beta_rmw': beta_rmw,
                        'beta_cma': beta_cma,
                        'r2': r_squared
                    })
                    valid_count += 1
            except Exception as e:
                logger.warning(f"Skipping {ticker} due to error: {e}")
                continue

        if not results:
            logger.warning("No valid regression results found.")
            return pd.DataFrame()

        # 4. 排序与输出
        df_res = pd.DataFrame(results)
        df_res.set_index('ticker', inplace=True)
        
        # 过滤掉 R2 太低的？(比如噪音太大，Alpha 不可信)
        # 这里暂时不过滤，全凭 Alpha 说话
        
        df_res = df_res.sort_values('final_score', ascending=False)
        
        logger.info(f"✅ Regression completed for {valid_count} stocks. Top Alpha: {df_res.iloc[0]['alpha_annual']:.2%}")
        
        return df_res