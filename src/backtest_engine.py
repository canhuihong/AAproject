import pandas as pd
import numpy as np
import logging
from src.factor_engine import FactorEngine
from src.optimizer import PortfolioOptimizer
from src.data_manager import DataManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backtest")

class BacktestEngine:
    def __init__(self, start_date='2023-01-01', initial_capital=100000.0, transaction_cost=0.001):
        self.start_date = pd.to_datetime(start_date)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.db = DataManager()
        self.factor_engine = FactorEngine()
        
    def get_rebalance_schedule(self):
        """获取调仓日列表 (月末)"""
        conn = self.db._get_conn()
        try:
            df_dates = pd.read_sql("SELECT DISTINCT date FROM prices ORDER BY date", conn)
        except Exception as e:
            logger.error(f"Database read error: {e}")
            return []
        finally:
            conn.close()
        
        if df_dates.empty:
            logger.error("No price data found in database!")
            return []

        df_dates['date'] = pd.to_datetime(df_dates['date'])
        
        # 筛选回测区间
        valid_dates = df_dates.loc[df_dates['date'] >= self.start_date, 'date']
        
        if valid_dates.empty:
            logger.warning(f"No dates found after start_date {self.start_date}")
            return []
            
        # 取每个月最后一天
        rebalance_dates = valid_dates.groupby(valid_dates.dt.to_period('M')).max()
        return rebalance_dates.sort_values().tolist()

    def _get_period_price_data(self, tickers, start_date, end_date):
        """
        获取价格数据，并进行高强度的清洗和填充
        """
        conn = self.db._get_conn()
        placeholders = ",".join([f"'{t}'" for t in tickers])
        s_str = start_date.strftime('%Y-%m-%d')
        e_str = end_date.strftime('%Y-%m-%d')
        
        query = f"""
            SELECT date, ticker, close 
            FROM prices 
            WHERE ticker IN ({placeholders}) 
            AND date >= '{s_str}' AND date <= '{e_str}'
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty: 
            return pd.DataFrame()
        
        # 数据透视
        df['date'] = pd.to_datetime(df['date'])
        pivot = df.pivot(index='date', columns='ticker', values='close')
        
        # 【关键修复】
        # 1. 前向填充 (ffill): 如果某天停牌，用昨天价格
        pivot = pivot.ffill()
        
        # 2. 后向填充 (bfill): 如果第一天就停牌，用后面价格补（极少情况）
        pivot = pivot.bfill()
        
        # 3. 如果还有 NaN (说明这只股票这段时间完全没数据)，直接丢弃该列
        pivot = pivot.dropna(axis=1, how='any')
        
        return pivot

    def run(self):
        rebalance_dates = self.get_rebalance_schedule()
        if len(rebalance_dates) < 2:
            logger.warning("Not enough rebalance dates to run backtest.")
            return pd.DataFrame()
            
        logger.info(f"📅 Backtest Range: {rebalance_dates[0].date()} -> {rebalance_dates[-1].date()}")
        
        full_curve = []
        current_capital = self.initial_capital
        prev_weights = {} # 用于计算换手率
        
        # 遍历每个调仓周期
        for i in range(len(rebalance_dates) - 1):
            curr_date = rebalance_dates[i]
            next_date = rebalance_dates[i+1]
            date_str = curr_date.strftime('%Y-%m-%d')
            
            logger.info(f"🔄 Processing {date_str} | Capital: ${current_capital:,.0f}")
            
            # --- 1. 选股 ---
            try:
                scored_df = self.factor_engine.get_scored_universe(analysis_date=date_str)
            except Exception as e:
                logger.warning(f"   ⚠️ Factor Engine error on {date_str}: {e}")
                scored_df = pd.DataFrame()

            if scored_df.empty:
                logger.warning(f"   ⚠️ No stocks selected. Holding Cash.")
                # If we haven't started trading yet, don't accumulate "flat line"
                # Just skip this date basically moving the start_date forward
                if len(full_curve) == 0:
                    logger.info("   ⏳ Waiting for data to warm up... Skipping period.")
                    continue
                    
                # Otherwise, hold cash
                dates = pd.date_range(curr_date, next_date)
                full_curve.append(pd.Series(current_capital, index=dates))
                continue
                
            top_tickers = scored_df.head(20).index.tolist()
            
            # --- 2. 优化权重 (Max 10%) ---
            try:
                optimizer = PortfolioOptimizer(top_tickers, analysis_date=date_str)
                allocation_df = optimizer.optimize_sharpe_ratio(max_weight=0.1)
            except Exception as e:
                logger.warning(f"   ⚠️ Optimizer crashed: {e}")
                allocation_df = pd.DataFrame()
                
            # 如果优化失败，尝试等权重兜底
            if allocation_df.empty and top_tickers:
                logger.info("   ⚠️ Fallback to Equal Weight.")
                allocation_df = pd.DataFrame({'ticker': top_tickers, 'weight': 1.0/len(top_tickers)})
            elif allocation_df.empty:
                continue

            # 提取权重字典
            weights = dict(zip(allocation_df['ticker'], allocation_df['weight']))
            weights = {k: v for k, v in weights.items() if v > 0.001}
            active_tickers = list(weights.keys())
            
            if not active_tickers:
                continue

            logger.info(f"   ✅ Position: {len(active_tickers)} stocks (Top: {active_tickers[:3]}...)")

            # --- 2.5 交易成本计算 ---
            # Turnover = sum(|w_new - w_old|)
            all_tickers = set(weights.keys()) | set(prev_weights.keys())
            turnover = sum(abs(weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_tickers)
            
            # Cost = Turnover * Cap * Rate
            # 注意：这里的 turnover 是双边的总变动比例 (比如卖10%买10%，turnover=20%)
            # 这里的 transaction_cost 如果是单边的 (比如 10bps)，那么对于买和卖都要收
            # 所以 0.001 * 20% = 0.02% 的总资产
            cost = turnover * val * self.transaction_cost if (val := current_capital) > 0 else 0
            
            # 首日建仓 (prev_weights为空) 也算 Turnover (即 100% 买入)
            if not prev_weights:
                # 初始建仓只算买入的一边成本?
                # 通常 Backtest 假设初始资金是现金，所以是买入 100%，Turnover=100%
                # Cost = 1.0 * cost_rate
                pass

            current_capital -= cost
            logger.info(f"   💸 Cost: ${cost:.2f} (Turnover: {turnover:.1%}) -> Net Cap: ${current_capital:,.0f}")
            
            # 更新 prev_weights
            prev_weights = weights

            # --- 3. 模拟持有 ---
            # 【修复未来函数】
            # 我们在 curr_date 收盘后做决策，所以在下一天 (curr_date + 1) 开始持有
            try:
                trade_start_date = curr_date + pd.Timedelta(days=1)
                
                # 获取从 交易日 到 下个调仓日 的数据
                # 注意：如果 next_date 也是 T+1，那这里会取不到数据，但在月度调仓下一般没事
                price_data = self._get_period_price_data(active_tickers, trade_start_date, next_date)
            except Exception as e:
                logger.warning(f"   ⚠️ Error getting price data: {e}")
                continue
            
            # 二次检查：确保我们买的股票在价格数据里真的存在
            # (get_period_price_data 可能会因为缺数据而丢弃某些列)
            valid_tickers = [t for t in active_tickers if t in price_data.columns]
            
            if not valid_tickers:
                logger.warning("   ⚠️ No price data for selected stocks! Holding Cash.")
                dates = pd.date_range(curr_date, next_date)
                full_curve.append(pd.Series(current_capital, index=dates))
                continue
                
            # 重新归一化权重 (因为有些股票可能被丢了)
            valid_weights = pd.Series({t: weights[t] for t in valid_tickers})
            valid_weights = valid_weights / valid_weights.sum()
            
            # 计算净值曲线
            period_prices = price_data[valid_tickers]
            
            # 归一化价格 (Base 1.0)
            # 这里的 iloc[0] 极其重要，必须非零
            start_prices = period_prices.iloc[0]
            if (start_prices == 0).any():
                logger.warning("   ⚠️ Found zero price, dropping bad columns.")
                period_prices = period_prices.loc[:, (start_prices != 0)]
                valid_weights = valid_weights[period_prices.columns]
                valid_weights = valid_weights / valid_weights.sum()
                start_prices = period_prices.iloc[0]

            normalized_prices = period_prices / start_prices
            
            # 每日组合价值
            period_portfolio_value = (normalized_prices * valid_weights).sum(axis=1) * current_capital
            
            # 拼接
            if i > 0:
                full_curve.append(period_portfolio_value.iloc[1:]) # 避免日期重复
            else:
                full_curve.append(period_portfolio_value)
            
            current_capital = period_portfolio_value.iloc[-1]
            
       # --- 结束 ---
        if not full_curve:
            logger.warning("Backtest produced no curve points.")
            return pd.DataFrame()
            
        equity_curve = pd.concat(full_curve)
        
        # --- 修复点：更稳健的基准对齐逻辑 ---
        # 1. 创建结果 DataFrame，以策略为准
        result_df = pd.DataFrame({'Strategy': equity_curve})
        
        # 2. 尝试获取基准 (SPY)
        try:
            spy_data = self._get_period_price_data(['SPY'], rebalance_dates[0], rebalance_dates[-1])
            if not spy_data.empty and 'SPY' in spy_data.columns:
                # 计算 SPY 净值
                # 关键修复：基准必须和策略在同一天归一化 (Rebase)
                # 否则如果策略从 2024 年才开始 (前面跳过了)，基准却从 2023 年开始算，起点就不一样了
                spy_series = spy_data['SPY']
                
                # 找到策略真正的起始日期
                strategy_start_date = result_df.index[0]
                
                # 截取该日期之后的 SPY
                spy_series = spy_series[spy_series.index >= strategy_start_date]
                
                if not spy_series.empty:
                    # 归一化：让 SPY 在策略开始的那一天，净值等于策略的初始资金
                    base_value = spy_series.iloc[0]
                    start_capital = result_df['Strategy'].iloc[0] # 应该是 initial_capital，但为了保险取第一笔
                    
                    spy_curve = (spy_series / base_value) * start_capital
                    
                    result_df = result_df.join(spy_curve.rename('Benchmark (SPY)'), how='left')
                    result_df['Benchmark (SPY)'] = result_df['Benchmark (SPY)'].ffill()
            else:
                logger.warning("Benchmark (SPY) data missing. Skipping benchmark comparison.")
        except Exception as e:
            logger.warning(f"Failed to process benchmark: {e}")

        # 3. 最终清理
        # 移除任何因为数据拼接导致的 NaN，但打印警告
        original_len = len(result_df)
        final_df = result_df.dropna(subset=['Strategy']) # 只要策略有值就行
        
        if len(final_df) < original_len:
             logger.warning(f"Dropped {original_len - len(final_df)} rows due to missing strategy data.")

        return final_df