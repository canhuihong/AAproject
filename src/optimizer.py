import logging
import pandas as pd
import numpy as np
import scipy.optimize as sco
from scipy.optimize import minimize
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from src.data_manager import DataManager

from src.config import OPTIMIZER_PARAMS, logger as global_logger
# Use the global logger instead of creating a new one to keep config consistent
logger = global_logger

class PortfolioOptimizer:
    def __init__(self, tickers, analysis_date, risk_free_rate=0.04):
        self.tickers = tickers
        self.analysis_date = analysis_date
        self.rfr = risk_free_rate
        self.db = DataManager()
        # 初始化时直接清洗数据
        self.data, self.valid_tickers = self._fetch_and_clean_data()
        
        # [Phase 2] Load Sector Map
        self.sector_map = self.db.get_sector_map()
        
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
        # 之前可能硬编码了，现在统一走 Config
        min_history = OPTIMIZER_PARAMS['MIN_HISTORY_DAYS'] 
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
        [Phase 2 Upgrade] Adds Sector Constraints (Max 30% per sector)
        """
        # 1. 如果数据不足，直接返回等权重 (Equal Weight)
        if self.data.empty or not self.valid_tickers:
            logger.warning("Optimization skipped (bad data). Defaulting to Equal Weights.")
            return self._get_equal_weights(self.tickers)

        num_assets = len(self.valid_tickers)
        
        # --- [Phase 2 Improvement] Feasible Initial Guess (Balanced Sector) ---
        
        # 1. Identify Sectors
        sector_indices = {} 
        for idx, ticker in enumerate(self.valid_tickers):
            sec = self.sector_map.get(ticker, 'Unknown')
            if sec not in sector_indices: sector_indices[sec] = []
            sector_indices[sec].append(idx)
            
        num_sectors = len(sector_indices)
        
        # 2. Determine Feasible Constraint
        # Min feasible max_weight = 1.0 / num_sectors
        # Buffer to allow optimization room
        min_feasible = 1.01 / max(1, num_sectors)
        MAX_SECTOR_WEIGHT = max(0.30, min_feasible)
        
        if MAX_SECTOR_WEIGHT > 0.30:
            logger.warning(f"⚠️ Only {num_sectors} sectors found. Relaxing constraint to {MAX_SECTOR_WEIGHT:.1%}")

        # 3. Construct Balanced Initial Guess (Sector Equal Weight)
        # Assign 1/N weight to each sector, divided by M stocks in that sector
        init_guess = np.zeros(num_assets)
        target_sector_weight = 1.0 / max(1, num_sectors)
        
        for sec, indices in sector_indices.items():
            n_in_sec = len(indices)
            if n_in_sec > 0:
                w = target_sector_weight / n_in_sec
                # If this violates individual max_weight, we might have an issue, 
                # but max_weight is usually 0.1 or 0.15, and w is likely smaller.
                # If w > max_weight, clip it (but then sum < 1).
                # Let's simple clip and normalize later if needed.
                init_guess[indices] = w
                
        # Final Normalize (to be safe against floating point errors)
        if np.sum(init_guess) > 0:
            init_guess /= np.sum(init_guess)
        
        # ----------------------------------------------------
        
        # 4. Define Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        for sec, indices in sector_indices.items():
            def sector_constraint(x, idxs=indices, limit=MAX_SECTOR_WEIGHT):
                return limit - np.sum(x[idxs])
            constraints.append({'type': 'ineq', 'fun': sector_constraint})
            
        logger.info(f"Optimization Constraints: {len(constraints)} (1 Eq + {len(constraints)-1} Sector Ineq)")

        bounds = tuple((0.0, max_weight) for _ in range(num_assets))
        
        def neg_sharpe(weights):
            try:
                r, v, s = self.get_portfolio_metrics(weights)
                if np.isnan(s) or np.isinf(s): return 1e6
                return -s
            except:
                return 1e6

        # 5. Attempt Optimization
        try:
            result = minimize(
                neg_sharpe, 
                init_guess, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            # 如果优化成功，使用结果；否则使用初始猜测
            final_weights = result.x if result.success else init_guess
            
            # 结果清洗
            final_weights = np.where(final_weights < 0.001, 0, final_weights)
            if np.sum(final_weights) == 0: 
                final_weights = init_guess # 防止全变0
            else:
                final_weights = final_weights / np.sum(final_weights)
            
            # Log valid result
            if result.success:
                logger.info("✅ Optimization Succeeded!")
            else:
                logger.warning(f"⚠️ Optimization Failed: {result.message}")
            
            return pd.DataFrame({'ticker': self.valid_tickers, 'weight': final_weights}).sort_values('weight', ascending=False)

        except Exception as e:
            logger.error(f"Optimization crashed ({e}). Fallback to Equal Weights.")
            return self._get_equal_weights(self.valid_tickers)

    def optimize_portfolio(self):
        """
        Generic entry point. Dispatches to HRP or Sharpe based on Config.
        """
        if OPTIMIZER_PARAMS.get('USE_HRP', False):
            logger.info("🧠 Optimization Mode: HRP (Hierarchical Risk Parity)")
            return self.optimize_hrp()
        else:
            logger.info("🧠 Optimization Mode: MVO (Max Sharpe)")
            return self.optimize_sharpe_ratio()

    # ==========================================
    # HRP Implementation
    # ==========================================
    def optimize_hrp(self):
        if self.data.empty or not self.valid_tickers:
            return self._get_equal_weights(self.tickers)

        try:
            # 1. Prepare Data
            returns = self.data.pct_change().dropna()
            if returns.empty: return self._get_equal_weights(self.valid_tickers)
            
            cov = returns.cov()
            corr = returns.corr()
            
            # 2. Clustering
            dist = ssd.pdist(corr, metric='euclidean') # Or correlation distance (1-corr)
            # More standard HRP uses (1-corr) as distance *matrix* then condensed?
            # Lopez de Prado uses: d[i,j] = sqrt(0.5 * (1 - rho[i,j])) ? 
            # Simplified: Use Euclidean distance of correlation matrix rows
            link = sch.linkage(dist, method='single')
            
            # 3. Quasi Diagonalization (Sort)
            sort_ix = self._get_quasi_diag(link)
            sort_ix = [self.valid_tickers[i] for i in sort_ix] # Map index to ticker
            
            # Reorder cov/corr
            cov = cov.loc[sort_ix, sort_ix]
            
            # 4. Recursive Bisection
            hrp_weights = self._get_rec_bisection(cov, sort_ix)
            
            # Convert to DataFrame
            weights_series = pd.Series(hrp_weights).sort_values(ascending=False)
            df_w = pd.DataFrame({'ticker': weights_series.index, 'weight': weights_series.values})
            
            logger.info("✅ HRP Optimization Succeeded!")
            return df_w

        except Exception as e:
            logger.error(f"HRP Optimization failed: {e}. Fallback to Equal.")
            return self._get_equal_weights(self.valid_tickers)

    def _get_quasi_diag(self, link):
        # Sort clustered items by distance
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3] # Total items
        
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2) # Make space
            df0 = sort_ix[sort_ix >= num_items] # Clusters
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0] # Item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0]) # Item 2
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
            
        return sort_ix.tolist()

    def _get_rec_bisection(self, cov, sort_ix):
        w = pd.Series(1, index=sort_ix)
        c_items = [sort_ix]
        
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c1 = c_items[i] 
                c2 = c_items[i + 1]
                v1 = self._get_cluster_var(cov, c1)
                v2 = self._get_cluster_var(cov, c2)
                alpha = 1 - v1 / (v1 + v2)
                w[c1] *= alpha
                w[c2] *= 1 - alpha
        return w

    def _get_cluster_var(self, cov, c_items):
        # Calculate variance of a cluster
        cov_slice = cov.loc[c_items, c_items]
        w = self._get_ivp(cov_slice).values.reshape(-1, 1)
        c_var = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
        return c_var

    def _get_ivp(self, cov):
        # Inverse Variance Portfolio
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return pd.Series(ivp, index=cov.index)

    def _get_equal_weights(self, ticker_list):
        """辅助函数：生成等权重配置"""
        if not ticker_list:
            return pd.DataFrame(columns=['ticker', 'weight'])
        
        n = len(ticker_list)
        return pd.DataFrame({
            'ticker': ticker_list,
            'weight': [1.0/n] * n
        })