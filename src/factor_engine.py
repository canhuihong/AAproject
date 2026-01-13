import pandas as pd
import numpy as np
import logging
import pandas_datareader.data as web
import statsmodels.api as sm
from src.data_manager import DataManager
from src.config import FULL_BLOCKLIST, FF_CACHE_PATH, PROXY_URL, DATA_DIR, FACTOR_PARAMS, STRATEGY_PARAMS, REGIME_WEIGHTS, LIQUIDITY_PARAMS, MACRO_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FactorEngine")

class FactorEngine:
    """
    Fama-French 3-Factor Model Engine.
    Reverts to the regression-based approach to find Alpha.
    """
    def __init__(self):
        self.db = DataManager()
        self.lookback_days = FACTOR_PARAMS['LOOKBACK_DAYS']
        self.min_observations = FACTOR_PARAMS['MIN_OBSERVATIONS']
        # self.rfr removed; now dynamic
        
        # [Cache] Store computed snapshots to avoid redundant regression calls
        # Key: analysis_date (str), Value: DataFrame
        self._score_cache = {}

    def _detect_market_regime(self, analysis_date):
        """
        [Macro Awareness Update]
        Determine Market Regime based on:
        1. VIX Level (Fear)
        2. SPY Trend (Technical)
        3. Yield Curve (Recession Warning)
        
        Returns: 'STRONG_BULL' | 'BULL' | 'CORRECTION' | 'CRISIS'
        """
        if not STRATEGY_PARAMS['REGIME_SWITCHING']:
            return 'NEUTRAL'
            
        try:
            # 1. Fetch Data Inputs
            # Lookback window for technicals (SMA200)
            start_date_tech = (pd.Timestamp(analysis_date) - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
            
            conn = self.db._get_conn()
            
            # A. Get SPY for Trend
            df_spy = pd.read_sql(f"SELECT date, close FROM prices WHERE ticker = 'SPY' AND date >= '{start_date_tech}' AND date <= '{analysis_date}' ORDER BY date", conn)
            
            # B. Get Macro Indicators (Single Day Snapshot is usually enough, but we take last 5 days to be safe against missing data)
            start_date_macro = (pd.Timestamp(analysis_date) - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
            
            # Fetch VIX, TNX (10Y), IRX (13W ~ 3M)
            # Note: IRX is yield * 10 (e.g., 4.5 index = 4.5% yield? No, Yahoo ^IRX is usually standard yield)
            # Let's assume standard % format.
            df_macro = pd.read_sql(f"SELECT date, ticker, close FROM prices WHERE ticker IN ('^VIX', '^TNX', '^IRX') AND date >= '{start_date_macro}' AND date <= '{analysis_date}'", conn)
            
            conn.close()
            
            # --- Technical Score ---
            if len(df_spy) < 200:
                tech_score = 0
            else:
                sma200 = df_spy['close'].rolling(200).mean().iloc[-1]
                curr_price = df_spy['close'].iloc[-1]
                if pd.isna(sma200): tech_score = 0
                elif curr_price > sma200: tech_score = 1
                else: tech_score = -1
                
            # --- VIX Score ---
            vix_score = 0
            last_vix = 20.0 # Default fallback
            
            vix_data = df_macro[df_macro['ticker'] == '^VIX'].sort_values('date')
            if not vix_data.empty:
                last_vix = vix_data.iloc[-1]['close']
                
            if last_vix > MACRO_PARAMS['VIX_PANIC_THRESHOLD']: # > 30
                vix_score = -2
            elif last_vix > MACRO_PARAMS['VIX_HIGH_THRESHOLD']: # > 25
                vix_score = -1
            elif last_vix < MACRO_PARAMS['VIX_LOW_THRESHOLD']: # < 15
                vix_score = 1
            else:
                vix_score = 0
                
            # --- Yield Curve Score (10Y - 3M) ---
            # 10Y-2Y is standard, but 10Y-3M is often considered more predictive by Fed.
            # We use ^TNX (10Y) and ^IRX (13W).
            yield_score = 0
            
            tnx_data = df_macro[df_macro['ticker'] == '^TNX'].sort_values('date')
            irx_data = df_macro[df_macro['ticker'] == '^IRX'].sort_values('date')
            
            if not tnx_data.empty and not irx_data.empty:
                t10 = tnx_data.iloc[-1]['close']
                t3m = irx_data.iloc[-1]['close']
                
                # Check inversion
                diff = t10 - t3m
                if diff < MACRO_PARAMS['YIELD_CURVE_INVERSION_THRESHOLD']:
                    yield_score = -1
                elif diff > 0.5: # Healthy curve
                    yield_score = 0 # Neutral/Normal. (Could be +1 if steepening, but keep simple)
                    
            # --- Total Macro Score ---
            total_score = tech_score + vix_score + yield_score
            
            # Map to Regime
            # Range: 
            # Max: +1 (Tech) + 1 (Vix) + 0 (Yield) = +2
            # Min: -1 (Tech) - 2 (Vix) - 1 (Yield) = -4
            
            regime = 'NEUTRAL'
            
            if total_score >= 1:
                regime = 'STRONG_BULL'
            elif total_score == 0:
                regime = 'NEUTRAL' # or Weak Bull
            elif total_score >= -2:
                regime = 'CORRECTION'
            else:
                regime = 'CRISIS' # <= -3
                
            # Fallback for "BULL" in config if needed, but our new config has specific keys.
            # If mapped key doesn't exist in config, fallback to closest.
            
            diff_display = f"{diff:.2f}" if 'diff' in locals() else "N/A"
            logger.info(f"🧩 Macro Awareness: Score={total_score} (Tech:{tech_score}, VIX:{last_vix:.1f}, YieldDiff:{diff_display}) -> {regime}")
            
            return regime

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return 'NEUTRAL'

    def _get_historical_data(self, analysis_date):
        """Fetch price and fundamentals for the window"""
        conn = self.db._get_conn()
        
        # 1. Get Prices (and Volume for liquidity filter)
        # We need enough buffer for lookback
        start_date = (pd.Timestamp(analysis_date) - pd.Timedelta(days=self.lookback_days * 2)).strftime('%Y-%m-%d')
        query_prices = f"SELECT date, ticker, close, volume FROM prices WHERE date >= '{start_date}' AND date <= '{analysis_date}'"
        
        # 2. Get Fundamentals (for SMB/HML construction + Accruals)
        # We need the latest snapshot relative to analysis_date
        # Fix: Ensure we don't pick up "hollow" rows (shares update only) that have NULL financials
        query_fund = f"""
            SELECT f.ticker, f.total_equity, f.shares_count, f.net_income, f.total_revenue, f.total_assets, f.operating_cash_flow
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

    def _compute_single_snapshot(self, analysis_date, weights=None):
        """
        Internal: Runs the regression for a specific date with specific weights
        """
        if weights is None:
            # Fallback to default
            weights = {
                'WEIGHT_ALPHA': FACTOR_PARAMS['WEIGHT_ALPHA'],
                'WEIGHT_QUALITY': FACTOR_PARAMS['WEIGHT_QUALITY'],
                'WEIGHT_LOW_VOL': FACTOR_PARAMS['WEIGHT_LOW_VOL']
            }
            
        # [Cache Check]
        if analysis_date in self._score_cache:
            # logger.debug(f"⚡ Cache Hit: {analysis_date}")
            return self._score_cache[analysis_date].copy()

        logger.info(f"Running Fama-French Regression for {analysis_date}...")
        
        # 1. Load Data
        df_price, df_fund = self._get_historical_data(analysis_date)
        
        logger.debug(f"DEBUG: {analysis_date} | Price Rows: {len(df_price)} | Fund Rows: {len(df_fund)}")
        
        if df_price.empty: 
            logger.debug("DEBUG: df_price is empty!")
            return pd.DataFrame()
        
        # 2. Pivot Price & Volume
        df_price['date'] = pd.to_datetime(df_price['date'])
        try:
            df_dedup = df_price.drop_duplicates(subset=['date', 'ticker'])
            pivot_prices = df_dedup.pivot(index='date', columns='ticker', values='close')
            pivot_volume = df_dedup.pivot(index='date', columns='ticker', values='volume')
        except Exception:
            return pd.DataFrame()
        
        pivot_prices = pivot_prices.sort_index()
        
        # --- 2.1 Liquidity Filter ---
        # Calculate Average Dollar Volume (last 20 days)
        last_prices = pivot_prices.ffill().iloc[-1]
        avg_volume = pivot_volume.rolling(20).mean().iloc[-1]
        avg_dollar_vol = last_prices * avg_volume
        
        # Filter Mask
        # A. Price > MIN_PRICE
        mask_price = last_prices >= LIQUIDITY_PARAMS['MIN_PRICE']
        
        # B. Dollar Vol > MIN_DOLLAR_VOLUME
        mask_vol = avg_dollar_vol >= LIQUIDITY_PARAMS['MIN_DOLLAR_VOLUME']
        
        valid_liquidity_tickers = last_prices.index[mask_price & mask_vol]
        
        logger.info(f"   💧 Liquidity Filter: {len(pivot_prices.columns)} -> {len(valid_liquidity_tickers)} stocks")
        
        if len(valid_liquidity_tickers) < 10:
             return pd.DataFrame()
             
        # Filter the pivot tables
        pivot_prices = pivot_prices[valid_liquidity_tickers]
        
        # 2.2 Calculate Returns
        df_returns = pivot_prices.pct_change(fill_method=None).dropna(how='all').tail(self.lookback_days)
        
        if len(df_returns) < self.min_observations:
            logger.warning(f"Insufficient history. Found {len(df_returns)} days.")
            return pd.DataFrame()
            
        # DEBUG PRINTS
        logger.debug(f"DEBUG: df_returns shape: {df_returns.shape}")
        logger.debug(f"DEBUG: df_fund shape: {df_fund.shape}")
        if not df_fund.empty:
            logger.debug(f"DEBUG: df_fund sample ticker: {df_fund.iloc[0]['ticker']}")

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
        
        logger.debug(f"DEBUG: Factors shape: {factors.shape}")
        if factors.empty: 
            logger.debug("DEBUG: Factors construction returned empty.")
            return pd.DataFrame()
        
        # Align
        common_dates = df_returns.index.intersection(factors.index)
        Y_all = df_returns.loc[common_dates]
        X = factors.loc[common_dates]
        
        # 4. Regression (Numba Accelerated)
        try:
            from src.numba_engine import run_parallel_ols
        except ImportError:
            logger.error("Numba engine missing. Reinstall numba.")
            return pd.DataFrame()

        # Prepare X (T x K)
        # Add Constant to X
        X_design = np.column_stack([np.ones(len(X)), X.values])
        
        # Prepare Y (T x N)
        # We need to subtract RF from ALL columns efficiently
        # RFR is aligned with common_dates
        rfr_values = rfr_series.loc[common_dates].values
        
        full_blocklist_set = set(FULL_BLOCKLIST)
        
        # Align Y to common dates
        # Note: Y_all might contain columns that are in blocklist
        valid_tickers = [t for t in Y_all.columns if t not in full_blocklist_set and t != 'SPY']
        
        if not valid_tickers:
            return pd.DataFrame()
            
        Y_subset = Y_all[valid_tickers].loc[common_dates]
        
        # Vectorized Subtraction (Excess Returns)
        # Y_subset is (T, N), rfr_values is (T,). Numpy broadcasting works if shape matches.
        # rfr needs (T, 1) to broadcast across N columns
        Y_excess = Y_subset.values - rfr_values[:, np.newaxis]
        
        # --- EXECUTE NUMBA ENGINE ---
        logger.info(f"⚡ Starting Numba Parallel Regression for {len(valid_tickers)} stocks...")
        import time
        t0 = time.time()
        
        # Output: [Alpha, Vol, Beta_const, Beta_Mkt, Beta_SMB, ...]
        raw_results = run_parallel_ols(Y_excess, X_design, min_obs=self.min_observations)
        
        t1 = time.time()
        logger.info(f"⚡ Regression Finished in {t1-t0:.4f}s")
        
        # 5. Pack Results
        # raw_results matches valid_tickers order
        # Cols: 0: Alpha, 1: Vol, 2: Beta_Const (same as Alpha/252), 3: Beta_Mkt, ...
        
        # Map betas to names. X_design cols: [Const, Mkt, SMB, HML, RMW, CMA, MOM]
        # raw_results cols: [Alpha, Vol, B_Const, B_Mkt, B_SMB, B_HML, B_RMW, B_CMA, B_MOM]
        
        df_res = pd.DataFrame(index=valid_tickers)
        df_res['alpha'] = raw_results[:, 0]
        df_res['volatility'] = raw_results[:, 1]
        df_res['beta_mkt'] = raw_results[:, 3]
        df_res['beta_smb'] = raw_results[:, 4]
        df_res['beta_hml'] = raw_results[:, 5]
        df_res['beta_rmw'] = raw_results[:, 6]
        df_res['beta_cma'] = raw_results[:, 7]
        df_res['beta_mom'] = raw_results[:, 8]
        
        # We also need 'close' price for final report
        # pivot_prices columns -> tickers
        curr_prices = pivot_prices.iloc[-1]
        df_res['close'] = curr_prices.reindex(valid_tickers).values
        
        # Filter NaNs (failed regressions)
        df_res.dropna(subset=['alpha'], inplace=True)

        if df_res.empty: return pd.DataFrame()
        
        if df_res.empty: return pd.DataFrame()
        
        # --- Risk Filters (Beta) ---
        df_res = df_res[(df_res['beta_mkt'] >= FACTOR_PARAMS['BETA_MIN']) & 
                        (df_res['beta_mkt'] <= FACTOR_PARAMS['BETA_MAX'])]
        
        # --- Alpha Outlier Filter ---
        df_res = df_res[(df_res['alpha'] < FACTOR_PARAMS['ALPHA_UPPER_BOUND']) & 
                        (df_res['alpha'] > FACTOR_PARAMS['ALPHA_LOWER_BOUND'])]
        
        if df_res.empty: return pd.DataFrame()

        # ==========================================
        # 6. Composite Scoring (Multi-Factor)
        # ==========================================
        valid_fund = df_fund.set_index('ticker').reindex(df_res.index)
        
        # A. Quality Factor: Accruals
        # Accruals = (Net Income - Operating Cash Flow) / Total Assets
        # Interpretation: High Accruals = Low Quality (Fake Earnings) -> We want LOW Accruals
        try:
            accruals = (valid_fund['net_income'] - valid_fund['operating_cash_flow']) / valid_fund['total_assets']
            accruals = accruals.fillna(0) # If missing, assume neutral
        except KeyError:
            # If OCF column missing (old DB), assume 0
            accruals = pd.Series(0, index=df_res.index)
            
        df_res['accruals'] = accruals
        
        # B. Z-Score Standardization (Robust to scale differences)
        
        # Helper: Rank and Z-Score (Percentile might be better but Z-score is standard)
        def robust_zscore(series):
            if len(series) < 2: return pd.Series(0, index=series.index)
            # Clip outliers first to avoiding skewing the mean
            shrunk = series.clip(lower=series.quantile(0.01), upper=series.quantile(0.99))
            return (shrunk - shrunk.mean()) / (shrunk.std() + 1e-6)

        z_alpha = robust_zscore(df_res['alpha'])
        
        # For Volatility and Accruals, LOWER is BETTER. So we negate them.
        z_low_vol = robust_zscore(df_res['volatility']) * -1.0
        z_quality = robust_zscore(df_res['accruals']) * -1.0 
        
        # C. Weighted Sum
        w_alpha = weights['WEIGHT_ALPHA']
        w_quality = weights['WEIGHT_QUALITY']
        w_vol = weights['WEIGHT_LOW_VOL']
        
        # [MODIFIED] Store Raw Z-Scores for External Optimization
        df_res['z_alpha'] = z_alpha
        df_res['z_quality'] = z_quality
        df_res['z_low_vol'] = z_low_vol
        
        df_res['final_score'] = (z_alpha * w_alpha) + (z_quality * w_quality) + (z_low_vol * w_vol)
        
        df_res.sort_values('final_score', ascending=False, inplace=True)
        
        logger.info(f"Scoring Complete. Top: {df_res.index[0]} (Score: {df_res.iloc[0]['final_score']:.2f} | Alpha: {df_res.iloc[0]['alpha']:.2%})")
        
        # [Cache Save]
        self._score_cache[analysis_date] = df_res.copy()
        
        return df_res

    def get_scored_universe(self, analysis_date=None, top_n=None):
        """
        Public API: Returns the final scored universe.
        Includes:
        1. Regime Switching (Dynamic Weights)
        2. Signal Smoothing (Tri-Month Average)
        """
        if not analysis_date:
            analysis_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
        # 1. Regime Detection
        regime = self._detect_market_regime(analysis_date)
        weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS['NEUTRAL'])
        
        logger.info(f"🔮 Strategy Mode: {regime} | Weights: {weights}")
        
        # 2. Signal Smoothing
        if not STRATEGY_PARAMS['ENABLE_SMOOTHING']:
            return self._compute_single_snapshot(analysis_date, weights)
            
        logger.info(f"🌊 Smoothing Enabled: Averaging last {STRATEGY_PARAMS['SMOOTHING_MONTHS']} months...")
        
        score_frames = []
        
        # Loop back N months (approx 30 days per month)
        for i in range(STRATEGY_PARAMS['SMOOTHING_MONTHS']):
            lookback_date = (pd.Timestamp(analysis_date) - pd.Timedelta(days=30 * i)).strftime('%Y-%m-%d')
            
            logger.info(f"   > Snapshot {i+1}: {lookback_date}")
            df = self._compute_single_snapshot(lookback_date, weights)
            
            if not df.empty:
                score_frames.append(df[['final_score', 'alpha', 'close']])
                
        if not score_frames:
            return pd.DataFrame()
            
        # 3. Combine and Average
        # Concat all frames
        combined = pd.concat(score_frames, axis=1)
        # We need to average 'final_score' across columns. 
        # But 'final_score' appears multiple times.
        # Safer way: average by index
        
        # Create a DF containing only final_scores
        scores_only = pd.DataFrame()
        for i, df in enumerate(score_frames):
            scores_only[f'score_{i}'] = df['final_score']
            
        # Average Score
        avg_score = scores_only.mean(axis=1)
        
        # Get the latest other data (alpha, close) from the *first* frame (analysis_date)
        # We only keep stocks that exist in the LATEST snapshot (analysis_date).
        # Stocks that dropped out are ignored.
        latest_df = score_frames[0].copy()
        
        # Update final_score with average
        # We intersection with avg_score to handle cases where stock didn't exist in past (less smoothing)
        common_idx = latest_df.index.intersection(avg_score.index)
        latest_df.loc[common_idx, 'final_score'] = avg_score.loc[common_idx]
        
        # Sort
        latest_df.sort_values('final_score', ascending=False, inplace=True)
        
        logger.info(f"✅ Smoothing Complete. Top: {latest_df.index[0]} (Avg Score: {latest_df.iloc[0]['final_score']:.2f})")
        
        return latest_df