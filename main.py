import sys
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 强制 UTF-8 输出，防止 emoji 报错
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# 引入我们刚才写好的 Reporting 模块
from src.reporting import ReportManager
from src.factor_engine import FactorEngine
from src.optimizer import PortfolioOptimizer
from src.backtest_engine import BacktestEngine

# 修复路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def calculate_max_drawdown(series):
    """辅助函数：计算最大回撤"""
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

def run_live_mode(report):
    """实盘模式：基于最新数据推荐当前持仓"""
    print("\n" + "="*60)
    print("📢 [PART 1] LIVE MARKET RECOMMENDATION")
    print("="*60)
    
    report.add_heading("Live Portfolio Recommendation")
    report.add_text("Based on the latest available market and fundamental data.")
    
    engine = FactorEngine()
    # 动态获取昨天的日期作为分析日（确保有收盘价）
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    scored_df = engine.get_scored_universe(analysis_date=yesterday)
    
    if scored_df.empty:
        msg = "❌ No data found for scoring. Please run init_data.py first."
        print(msg)
        report.add_text(msg)
        return

    # 选 Top 10
    top_picks = scored_df.head(10)
    top_tickers = top_picks.index.tolist()
    
    # 1. 保存打分结果
    report.save_data(scored_df, "factor_scores_latest.csv")
    report.add_dataframe(top_picks.reset_index(), "Top 10 Scored Stocks (Raw)", max_rows=10)

    # 优化权重
    print(f"⚙️  Optimizing allocation for {yesterday}...")
    optimizer = PortfolioOptimizer(top_tickers, analysis_date=yesterday)
    allocation_df = optimizer.optimize_sharpe_ratio()
    
    if allocation_df.empty:
        print("⚠️ Optimization failed.")
        report.add_text("Optimization failed due to insufficient data history.")
        return

    # 2. 输出最终建议
    print("\n🏆 Final Recommended Portfolio:")
    final = allocation_df[allocation_df['weight'] > 0.001].copy()
    
    # 格式化输出到控制台
    print(final)
    
    # 保存结果
    report.save_data(final, "final_allocation.csv")
    
    # 在报告中展示
    # 格式化百分比显示
    final_display = final.copy()
    final_display['weight'] = final_display['weight'].apply(lambda x: f"{x:.2%}")
    report.add_dataframe(final_display, "🏆 Optimal Portfolio Weights")
    
    # 画个饼图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(final['weight'], labels=final['ticker'], autopct='%1.1f%%', startangle=90)
    ax.set_title("Recommended Allocation")
    report.add_figure(fig, "allocation_pie_chart")

def run_backtest_mode(report):
    """回测模式"""
    print("\n\n" + "="*60)
    print("⏳ [PART 2] HISTORICAL BACKTEST VERIFICATION")
    print("="*60)
    
    report.add_heading("Historical Backtest Results")
    
    # 设定回测起点
    backtester = BacktestEngine(start_date='2023-01-01', initial_capital=100000)
    results = backtester.run()
    
    if results.empty:
        print("❌ Backtest failed.")
        report.add_text("Backtest produced no trades/results.")
        return
        
    # 1. 保存回测曲线数据
    report.save_data(results, "backtest_equity_curve.csv")
    
    # 2. 计算关键指标
    strategy_ret = results['Strategy'].pct_change().dropna()
    total_ret = (results['Strategy'].iloc[-1] / 100000 - 1)
    
    # 年化收益 (简单估算)
    days = (results.index[-1] - results.index[0]).days
    ann_ret = (1 + total_ret) ** (365/days) - 1
    
    # 夏普比率 (假设无风险利率 4%)
    rfr_daily = 0.04 / 252
    excess_ret = strategy_ret - rfr_daily
    sharpe = (excess_ret.mean() / excess_ret.std()) * np.sqrt(252)
    
    # 最大回撤
    mdd = calculate_max_drawdown(results['Strategy'])
    
    # 打印到控制台
    print(f"\n📈 Performance Summary:")
    print(f"Total Return: {total_ret:.2%}")
    print(f"Annualized:   {ann_ret:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {mdd:.2%}")
    
    # 添加到报告 (漂亮的指标卡片)
    metrics = {
        "Total Return": f"{total_ret:.2%}",
        "CAGR": f"{ann_ret:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{mdd:.2%}",
        "Final Capital": f"${results['Strategy'].iloc[-1]:,.0f}"
    }
    report.add_metrics_panel(metrics)
    
    # 3. 绘制并保存曲线图
    fig = plt.figure(figsize=(12, 6))
    plt.plot(results.index, results['Strategy'], label='My Strategy', linewidth=2, color='#3498db')
    
    if 'Benchmark (SPY)' in results.columns:
        # 对标收益
        bench_ret = (results['Benchmark (SPY)'].iloc[-1] / 100000 - 1)
        report.add_text(f"Benchmark (SPY) Return: {bench_ret:.2%}")
        
        plt.plot(results.index, results['Benchmark (SPY)'], label='S&P 500 (SPY)', linestyle='--', color='gray', alpha=0.7)
        
    plt.title(f"Equity Curve ({backtester.start_date.date()} - Present)")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    report.add_figure(fig, "equity_curve")
    
    # 4. 只有在有Benchmark时才画相对收益图
    if 'Benchmark (SPY)' in results.columns:
        # 相对强弱 (RS)
        rs = results['Strategy'] / results['Benchmark (SPY)']
        fig2 = plt.figure(figsize=(12, 4))
        plt.plot(results.index, rs, color='purple', alpha=0.8)
        plt.axhline(1.0, linestyle='--', color='black', alpha=0.5)
        plt.title("Relative Strength vs SPY ( > 1.0 means Outperformance)")
        plt.grid(True, alpha=0.3)
        report.add_figure(fig2, "relative_strength")

if __name__ == "__main__":
    # 初始化报告管理器
    report = ReportManager()
    
    print(f"📂 Output Directory: {report.report_dir}")
    
    try:
        run_live_mode(report)
        run_backtest_mode(report)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        report.add_text(f"CRITICAL ERROR: {e}")
    finally:
        # 无论如何都要生成报告
        html_path = report.generate_html()
        if html_path:
            # 尝试在 Windows 下自动打开浏览器 (可选)
            if os.name == 'nt':
                os.startfile(html_path)