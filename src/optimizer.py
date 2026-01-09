import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging
from src.data_manager import DataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Optimizer")

class PortfolioOptimizer:
    def __init__(self, tickers, analysis_date, risk_free_rate=0.04):
        self.tickers = tickers
        self.analysis_date = analysis_date
        self.rfr = risk_free_rate
        self.db = DataManager()
        # 初始化时直接清洗数据
        self.data, self.valid_tickers = self._fetch_and_clean_data()
        
    def _fetch_and_clean_data(self):
        """
        极度宽容的数据获取与清洗
        """
        if not self.tickers:
            return pd.DataFrame(), []

        conn = self.db._get_conn()
        placeholders = ",".join([f"'{t}'" for t in self.tickers])
        
        query = f"""
            SELECT date, ticker, close 
            FROM prices 
            WHERE ticker IN ({placeholders}) AND date <= '{self.analysis_date}'
        """
        try:
            df = pd.read_sql(query, conn)
        except Exception as e:
            logger.error(f"DB Error: {e}")
            return pd.DataFrame(), []
        finally:
            conn.close()
        
        if df.empty:
            logger.warning("Optimizer received empty price data from DB.")
            return pd.DataFrame(), []
            
        df['date'] = pd.to_datetime(df['date'])
        
        # 数据透视
        pivot_df = df.pivot(index='date', columns='ticker', values='close')
        
        # 1. 填充空值 (ffill)
        pivot_df = pivot_df.ffill()
        
        # 2. 只有当有效数据长度 > 60天 (约3个月) 才保留
        # 之前的 126天可能太严格，导致所有股票都被剔除
        min_history = 60 
        valid_cols = []
        for col in pivot_df.columns:
            if pivot_df[col].count() >= min_history:
                # 再次检查：是否是死股 (方差为0)
                if pivot_df[col].std() > 1e-4:
                    valid_cols.append(col)
        
        if not valid_cols:
            logger.warning(f"No stocks have >{min_history} days of history.")
            return pd.DataFrame(), []
            
        # 3. 截取最后一段公共时间
        cleaned_df = pivot_df[valid_cols].dropna()
        
        # 如果取交集后剩下的太短，也放弃
        if len(cleaned_df) < 30: 
            logger.warning("Common history too short (<30 days).")
            return pd.DataFrame(), []
            
        return cleaned_df, valid_cols

    def get_portfolio_metrics(self, weights):
        weights = np.array(weights)
        returns = self.data.pct_change().dropna()
        
        mean_returns = returns.mean()
        # 协方差正则化 (防止 Singular Matrix)
        cov_matrix = returns.cov() + np.eye(len(returns.columns)) * 1e-5
        
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        # 安全除法
        if portfolio_volatility < 1e-6:
            return 0.0, 0.0, 0.0
            
        sharpe_ratio = (portfolio_return - self.rfr) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def optimize_sharpe_ratio(self, max_weight=1.0):
        """
        核心方法：无论如何返回一个 DataFrame，绝不报错
        """
        # 1. 如果数据不足，直接返回等权重 (Equal Weight)
        if self.data.empty or not self.valid_tickers:
            logger.warning("Optimization skipped (bad data). Defaulting to Equal Weights.")
            return self._get_equal_weights(self.tickers)

        num_assets = len(self.valid_tickers)
        init_guess = [1.0 / num_assets] * num_assets
        
        # 2. 定义优化问题
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, max_weight) for _ in range(num_assets))
        
        def neg_sharpe(weights):
            try:
                r, v, s = self.get_portfolio_metrics(weights)
                if np.isnan(s) or np.isinf(s): return 1e6
                return -s
            except:
                return 1e6

        # 3. 尝试优化
        try:
            result = minimize(
                neg_sharpe, 
                init_guess, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 50, 'ftol': 1e-6}
            )
            
            # 如果优化成功，使用结果；否则使用初始猜测
            final_weights = result.x if result.success else init_guess
            
            # 结果清洗
            final_weights = np.where(final_weights < 0.001, 0, final_weights)
            if np.sum(final_weights) == 0: 
                final_weights = init_guess # 防止全变0
            else:
                final_weights = final_weights / np.sum(final_weights)
            
            return pd.DataFrame({'ticker': self.valid_tickers, 'weight': final_weights}).sort_values('weight', ascending=False)

        except Exception as e:
            logger.error(f"Optimization crashed ({e}). Fallback to Equal Weights.")
            return self._get_equal_weights(self.valid_tickers)

    def _get_equal_weights(self, ticker_list):
        """辅助函数：生成等权重配置"""
        if not ticker_list:
            return pd.DataFrame(columns=['ticker', 'weight'])
        
        n = len(ticker_list)
        return pd.DataFrame({
            'ticker': ticker_list,
            'weight': [1.0/n] * n
        })