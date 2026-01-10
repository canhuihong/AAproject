import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# 1. 环境与密钥加载 (The Vault Opener)
# ==========================================
# 这一步非常关键：它去读取 .env 文件，把里面的文本加载到系统的内存中
# 如果没有这一行，os.getenv() 就读不到你在 .env 里写的密码
load_dotenv()

# ==========================================
# 2. 项目路径导航 (GPS)
# ==========================================
# 动态获取当前文件的父目录的父目录 -> 即项目根目录
# 这样做的好处是：无论你在哪个文件夹下运行 python 命令，路径永远是对的
ROOT_DIR = Path(__file__).resolve().parent.parent

# 定义标准数据存储位置
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
DB_PATH = DATA_DIR / "quant_lab_v2.db"  # 统一数据库路径

# 自动创建目录（如果不存在），省去手动 mkdir 的麻烦
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 3. 网络代理设置 (The "Bridge")
# ==========================================
# 逻辑说明：
# 1. 从 .env 获取端口号（默认 7897 是常见的 Clash 端口）
# 2. 拼接成标准的 HTTP 代理地址
# 3. 【核心】直接注入到 Python 进程的环境变量中
#    这样 yfinance, requests 等所有库都会自动走这个梯子，无需单独配置

PROXY_PORT = int(os.getenv("PROXY_PORT", 7897))
PROXY_URL = f"http://127.0.0.1:{PROXY_PORT}"

# 强制应用代理 (解决了国内无法下载美股数据的问题)
os.environ["HTTP_PROXY"] = PROXY_URL
os.environ["HTTPS_PROXY"] = PROXY_URL

print(f"System Proxy Configured: {PROXY_URL}")

# ==========================================
# 4. 业务常量与黑名单 (Business Logic)
# ==========================================
# 数据清洗规则：在选股时，我们要剔除 ETF 和非股票资产
# 继承自旧项目，这部分逻辑是正确的，保留！

# A. 指数 ETF (干扰选股模型)
ETF_BLOCKLIST = [
    'SPY', 'QQQ', 'TLT', 'GLD', 'IWM', 'USDOLLAR', '^GSPC', '^VIX', '^IXIC',
    'IVV', 'VOO', 'AGG', 'BND', 'LQD', 'HYG', 'EEM', 'EFA', 'SLV', 'USO',
    'SHY', 'IEF', 'TIP', 'VNQ', 'XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP'
]

# B. 宏观指标 (有些 API 会混在一起，需要剔除)
MACRO_BLOCKLIST = [
    'DGS10', 'T5YIE', 'T10Y2Y', 'BAMLC0A0CM', 'VIXCLS'
]

# C. 合并后的完整排除列表
FULL_BLOCKLIST = list(set(ETF_BLOCKLIST + MACRO_BLOCKLIST))

# ==========================================
# 5. 全局日志配置 (Logging)
# ==========================================
# 设定日志格式，方便调试
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 给项目定义一个主 Logger
logger = logging.getLogger("QML")

# ==========================================
# 6. 其他配置 (Optional)
# ==========================================
# 如果未来要接盈透证券 (IBKR)，配置预留在这里
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", 7497))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", 1))

# ==========================================
# 7. 调试与测试配置 (Debug Limits)
# ==========================================
# 设置每次下载的股票数量上限 (测试用)
# ⚠️ 注意：正式全量运行时，请将这两个值设为 None
SP500_LIMIT = None   # 测试模式：只下 50 只
SP600_LIMIT = None   # 测试模式：只下 50 只
SP400_LIMIT = None   # 中盘股
NASDAQ_LIMIT = None  # 科技股

# Fama-French Data Cache
FF_CACHE_PATH = DATA_DIR / "ff_factors.csv"

if __name__ == "__main__":
    # 测试代码：直接运行 python src/config.py 可以检查配置是否正确
    print("-" * 30)
    print(f"✅ Config Loaded Successfully")
    print(f"📂 Project Root: {ROOT_DIR}")
    print(f"📡 Proxy:        {os.environ.get('HTTP_PROXY')}")
    print(f"🚫 Blocklist:    {len(FULL_BLOCKLIST)} items")
    print(f"🚧 SP500 Limit:  {SP500_LIMIT}")
    print(f"🚧 SP600 Limit:  {SP600_LIMIT}")
    print(f"🚧 SP400 Limit:  {SP400_LIMIT}")
    print(f"🚧 NASDAQ Limit: {NASDAQ_LIMIT}")
    print("-" * 30)

# ==========================================
# 8. 回测与策略配置 (Strategy Config)
# ==========================================
# 集中管理策略参数，方便调优
BACKTEST_CONFIG = {
    'INITIAL_CAPITAL': 100000.0,
    'START_DATE': '2021-12-01',
    'TRANSACTION_COST': 0.001,  # 双边各千分之一 (10 bps)
    'MAX_POSITION_WEIGHT': 0.10, # 单只股票最大仓位 10%
    'REBALANCE_FREQ': 'M',      # 调仓频率 (M=Month, W=Week)
}