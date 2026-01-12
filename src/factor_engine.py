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
    """
    Fama-French 3-Factor Model Engine.
    Reverts to the regression-based approach to find Alpha.
    """
    def __init__(self):
        self.db = DataManager()
        self.lookback_days = 252  # 1 year regression window
        self.min_observations = 50 # Reduced to 50 (approx 2.5 months) to allow faster entry
        # self.rfr removed; now dynamic

    def _get_historical_data(self, analysis_date):
        """Fetch price and fundamentals for the window"""
        conn = self.db._get_conn()
        
        # 1. Get Prices
        # We need enough buffer for lookback
        start_date = (pd.Timestamp(analysis_date) - pd.Timedelta(days=self.lookback_days * 2)).strftime('%Y-%m-%d')
        query_prices = f"SELECT date, ticker, close FROM prices WHERE date >= '{start_date}' AND date <= '{analysis_date}'"
        
        # 2. Get Fundamentals (for SMB/HML construction)
        # We need the latest snapshot relative to analysis_date
        # Fix: Ensure we don't pick up "hollow" rows (shares update only) that have NULL financials
        query_fund = f"""
            SELECT f.ticker, f.total_equity, f.shares_count, f.net_income, f.total_revenue
            FROM fundamentals f
            INNER JOIN (
                SELECT ticker, MAX(date) as max_date
                FROM fundamentals
                WHERE date <= '{analysis_date}' AND total_equity IS NOT NULL AND net_income IS NOT NULL
                GROUP BY ticker
            ) latest ON f.ticker = latest.ticker AND f.date = latest.max_date
        """
        
        try:
            df_price = pd.read_sql(query_prices, conn)
            df_fund = pd.read_sql(query_fund, conn)
        except Exception as e:
            logger.error(f"DB Read Error: {e}")
            return pd.DataFrame(), pd.DataFrame()
        finally:
            conn.close()
            
        return df_price, df_fund

    def _construct_factors(self, df_returns, df_fund_snapshot, rfr_series):
        """
        Construct Local Fama-French 5-Factors (Mkt, SMB, HML, RMW, CMA) + Momentum (MOM)
        rfr_series: pandas Series of daily risk-free rates, aligned with df_returns index
        """
        """
        Construct Local Fama-French 5-Factors (Mkt, SMB, HML, RMW, CMA) + Momentum (MOM)
        """
        # 1. Market Factor (Mkt-RF)
        if 'SPY' in df_returns.columns:
            mkt_ret = df_returns['SPY']
        else:
            mkt_ret = df_returns.mean(axis=1) 
            
        # 2. Prepare Fundamental Data
        try:
            latest_prices = df_returns.iloc[-1]
        except IndexError:
            return pd.DataFrame()
        
        valid_tickers = df_returns.columns.intersection(df_fund_snapshot['ticker'])
        
        if len(valid_tickers) < 10:
            return pd.DataFrame({'Mkt-RF': mkt_ret - rfr_series})

        # Feature Vector
        metrics = df_fund_snapshot.set_index('ticker').loc[valid_tickers].copy()
        
        # --- Basic Metrics ---
        metrics['price'] = latest_prices[valid_tickers]
        metrics['mkt_cap'] = metrics['price'] * metrics['shares_count']
        metrics['bm_ratio'] = metrics['total_equity'] / metrics['mkt_cap'] # Book-to-Market (for HML)
        
        # --- 5-Factor Specifics ---
        # RMW (Profitability): Operating Profit / Book Equity
        # Proxy: Net Income / Total Equity (ROE)
        metrics['op_profit'] = metrics['net_income'] / metrics['total_equity'] 
        
        # CMA (Investment): Asset Growth
        # We don't have Total Assets, use Total Equity growth as proxy or simply ignore if not available
        # Simplified: We will focus on RMW and MOM if CMA data is lacking, but let's try to use Rev Growth if possible?
        # Standard: CMA is Change in Total Assets.
        # Let's use Total Revenue Growth as a proxy for "Aggressiveness" (Investment)
        metrics['inv_aggr'] = metrics['total_revenue'] # Ideally we need previous year's asset but let's stick to valid proxies
        
        # --- Momentum (MOM) ---
        # 12-1 month return
        mom_ret = (df_returns.iloc[-1] / df_returns.iloc[0]) - 1
        metrics['momentum'] = mom_ret[valid_tickers]

        # Filter bad data
        metrics = metrics[metrics['mkt_cap'] > 0]
        metrics.replace([np.inf, -np.inf], np.nan, inplace=True)
        metrics.dropna(inplace=True)

        if len(metrics) < 10: return pd.DataFrame()

        # Helper for Factor Return Calc
        def get_factor_return(metric_col, ascending=True):
            """
            Generic Long-Short Portfolio Construction
            ascending=True: Long Bottom 30%, Short Top 30% (e.g. SMB)
            ascending=False: Long Top 30%, Short Bottom 30% (e.g. HML, RMW, MOM)
            """
            p30 = metrics[metric_col].quantile(0.3)
            p70 = metrics[metric_col].quantile(0.7)
            
            if ascending:
                long_tickers = metrics[metrics[metric_col] <= p30].index
                short_tickers = metrics[metrics[metric_col] >= p70].index
            else:
                long_tickers = metrics[metrics[metric_col] >= p70].index
                short_tickers = metrics[metrics[metric_col] <= p30].index
                
            return df_returns[long_tickers].mean(axis=1) - df_returns[short_tickers].mean(axis=1)

        # --- Factor Construction ---
        
        # SMB: Small Minus Big (Long Small)
        smb = get_factor_return('mkt_cap', ascending=True)
        
        # HML: High Minus Low B/M (Long Value)
        hml = get_factor_return('bm_ratio', ascending=False)
        
        # RMW: Robust Minus Weak Profitability (Long High Profit)
        rmw = get_factor_return('op_profit', ascending=False)
        
        # CMA: Conservative Minus Aggressive (Long Conservative/Low Inv)
        # Low Investment -> High Return
        cma = get_factor_return('inv_aggr', ascending=True) 
        
        # MOM: Winners Minus Losers (Long High Mom)
        mom = get_factor_return('momentum', ascending=False)
        
        # Combine
        factors = pd.DataFrame({
            'Mkt-RF': mkt_ret - rfr_series,
            'SMB': smb,
            'HML': hml,
            'RMW': rmw, 
            'CMA': cma,
            'MOM': mom
        })
        
        return factors.dropna()

    def get_scored_universe(self, analysis_date=None, top_n=None):
        """
        Runs the regression (5-Factor + Mom) and returns stocks ranked by Alpha.
        """
        if not analysis_date:
            analysis_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
        logger.info(f"Running Fama-French 5-Factor + MOM Regression for {analysis_date}...")
        
        # 1. Load Data
        df_price, df_fund = self._get_historical_data(analysis_date)
        
        logger.warning(f"DEBUG: {analysis_date} | Price Rows: {len(df_price)} | Fund Rows: {len(df_fund)}")
        
        if df_price.empty: 
            logger.warning("DEBUG: df_price is empty!")
            return pd.DataFrame()
        
        # 2. Pivot
        df_price['date'] = pd.to_datetime(df_price['date'])
        try:
            pivot_prices = df_price.drop_duplicates(subset=['date', 'ticker']).pivot(index='date', columns='ticker', values='close')
        except Exception:
            return pd.DataFrame()
        
        pivot_prices = pivot_prices.sort_index()
        df_returns = pivot_prices.pct_change().dropna(how='all').tail(self.lookback_days)
        
        if len(df_returns) < self.min_observations:
            logger.warning(f"Insufficient history. Found {len(df_returns)} days.")
            return pd.DataFrame()
            
        # DEBUG PRINTS
        logger.warning(f"DEBUG: df_returns shape: {df_returns.shape}")
        logger.warning(f"DEBUG: df_fund shape: {df_fund.shape}")
        if not df_fund.empty:
            logger.warning(f"DEBUG: df_fund sample ticker: {df_fund.iloc[0]['ticker']}")

        # 2.5 Prepare RFR
        # Dynamic Risk Free Rate Logic
        if not df_returns.empty:
            s_date = df_returns.index[0].strftime('%Y-%m-%d')
            e_date = df_returns.index[-1].strftime('%Y-%m-%d')
            rfr_raw = self.db.get_risk_free_rate_series(s_date, e_date)
            # Align and Fill
            rfr_series = rfr_raw.reindex(df_returns.index, method='ffill').fillna(0.04/252)
        else:
             rfr_series = pd.Series(0.04/252, index=df_returns.index)

        # 3. Construct 6 Factors
        factors = self._construct_factors(df_returns, df_fund, rfr_series)
        
        logger.warning(f"DEBUG: Factors shape: {factors.shape}")
        if factors.empty: 
            logger.warning("DEBUG: Factors construction returned empty.")
            return pd.DataFrame()
        
        # Align
        common_dates = df_returns.index.intersection(factors.index)
        Y_all = df_returns.loc[common_dates]
        X = factors.loc[common_dates]
        
        # 4. Regression (Matrix Operation)
        # X: [1, Mkt, SMB, HML, RMW, CMA, MOM]
        X_design = np.column_stack([np.ones(len(X)), X.values])
        
        results = []
        full_blocklist_set = set(FULL_BLOCKLIST)
        
        for ticker in Y_all.columns:
            if ticker in full_blocklist_set or ticker == 'SPY': continue
            
            # Use dynamic RFR for Excess Return
            y = Y_all[ticker].values - rfr_series.loc[common_dates].values
            mask = ~np.isnan(y)
            if np.sum(mask) < self.min_observations: continue
            
            try:
                beta, residuals, rank, s = np.linalg.lstsq(X_design[mask], y[mask], rcond=None)
                
                # Calculate Volatility (Std Dev of residuals or raw content?)
                # We use raw volatility for filtering high risk stocks
                volatility = np.std(y) * np.sqrt(252)

                # Unwrap 7 params
                res = {
                    'ticker': ticker,
                    'alpha': beta[0] * 252, # Annualized
                    'volatility': volatility,
                    'beta_mkt': beta[1],
                    'beta_smb': beta[2],
                    'beta_hml': beta[3],
                    'beta_rmw': beta[4],
                    'beta_cma': beta[5],
                    'beta_mom': beta[6],
                    'close': pivot_prices[ticker].iloc[-1]
                }
                results.append(res)
            except:
                continue
                
        # 5. Result
        if not results: return pd.DataFrame()
        
        df_res = pd.DataFrame(results).set_index('ticker')
        
        # --- Risk Filters ---
        # 1. Beta Filter (New): Focus on Active Share
        # Exclude High Beta (> 1.3): Avoid leveraged market plays
        # Exclude Low Beta (< 0.1): Avoid dead money
        df_res = df_res[(df_res['beta_mkt'] > 0.1) & (df_res['beta_mkt'] < 1.3)]
        
        # 2. Alpha Outlier Filter
        # Removed strict Volatility filter to ensure we actually trade
        # Relaxed Alpha Outlier Filter
        df_res = df_res[(df_res['alpha'] < 5.0) & (df_res['alpha'] > -2.0)]
        
        df_res.sort_values('alpha', ascending=False, inplace=True)
        
        # Compatibility
        df_res['final_score'] = df_res['alpha']
        
        logger.info(f"Regression complete. Top: {df_res.index[0]} (Alpha: {df_res.iloc[0]['alpha']:.2%})")
        return df_res