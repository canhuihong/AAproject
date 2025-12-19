import os
import datetime
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 尝试从 config 导入，如果失败则使用默认值 (兼容性处理)
try:
    from src.config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = Path("outputs")

logger = logging.getLogger("QML.Reporting")

class ReportManager:
    def __init__(self, output_dir=None):
        """
        初始化报告管理器
        """
        if output_dir:
            self.report_dir = Path(output_dir)
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.report_dir = OUTPUT_DIR / timestamp
        
        # 定义子目录结构
        self.images_dir = self.report_dir / "images"
        self.data_dir = self.report_dir / "data"
        
        # 自动创建目录
        try:
            self.images_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📝 Report initialized. Path: {self.report_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to create report directories: {e}")
        
        self.html_content = []
        self._init_html()

    def _init_html(self):
        """写入 HTML 头部与 CSS 样式"""
        header = f"""
        <html>
        <head>
            <title>Quant Strategy Report</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background-color: #f9f9f9; }}
                h1 {{ border-bottom: 3px solid #2c3e50; padding-bottom: 10px; color: #2c3e50; }}
                h2 {{ color: #34495e; margin-top: 30px; border-left: 5px solid #3498db; padding-left: 10px; }}
                h3 {{ color: #7f8c8d; }}
                p {{ line-height: 1.6; color: #333; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); background-color: white; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; box-shadow: 0 0 20px rgba(0, 0, 0, 0.15); }}
                table thead tr {{ background-color: #009879; color: #ffffff; text-align: left; }}
                table th, table td {{ padding: 12px 15px; border: 1px solid #ddd; }}
                table tbody tr {{ border-bottom: 1px solid #dddddd; }}
                table tbody tr:nth-of-type(even) {{ background-color: #f3f3f3; }}
                table tbody tr:last-of-type {{ border-bottom: 2px solid #009879; }}
                .timestamp {{ color: #7f8c8d; font-size: 0.85em; text-align: right; }}
                .metric-box {{ display: inline-block; background: #fff; border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 5px; min-width: 150px; text-align: center; }}
                .metric-val {{ font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 0.8em; color: #7f8c8d; text-transform: uppercase; }}
            </style>
        </head>
        <body>
            <h1>📊 Quant Strategy Research Report</h1>
            <p class="timestamp">Generated at: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
        """
        self.html_content.append(header)

    def add_text(self, text):
        if text: self.html_content.append(f"<p>{text}</p>")

    def add_heading(self, text, level=2):
        if text: self.html_content.append(f"<h{level}>{text}</h{level}>")

    def add_metrics_panel(self, metrics_dict):
        """添加一行漂亮的指标卡片"""
        html = '<div style="display: flex; flex-wrap: wrap; justify-content: space-around;">'
        for k, v in metrics_dict.items():
            html += f"""
            <div class="metric-box">
                <div class="metric-val">{v}</div>
                <div class="metric-label">{k}</div>
            </div>
            """
        html += '</div>'
        self.html_content.append(html)

    def add_figure(self, fig, filename_tag):
        """保存 matplotlib 图片并嵌入 HTML"""
        if fig is None: return
        filename = f"{filename_tag}.png"
        filepath = self.images_dir / filename
        try:
            fig.savefig(filepath, bbox_inches='tight', dpi=100)
            plt.close(fig) 
            rel_path = f"images/{filename}"
            self.html_content.append(f"<img src='{rel_path}' alt='{filename_tag}'>")
        except Exception as e:
            logger.error(f"❌ Failed to save image {filename}: {e}")

    def add_dataframe(self, df, title=None, max_rows=20):
        """将 DataFrame 转换为 HTML 表格"""
        if df is None or df.empty:
            self.add_text(f"⚠️ No data available for: {title}")
            return
            
        if title:
            self.add_heading(title, level=3)
            
        # 截取前 max_rows 行
        display_df = df.head(max_rows)
        html_table = display_df.to_html(index=False, border=0, classes="styled-table")
        self.html_content.append(html_table)
        
        if len(df) > max_rows:
            self.add_text(f"*(Showing top {max_rows} of {len(df)} rows)*")

    def save_data(self, df, filename):
        """保存数据到 CSV"""
        if df is None or df.empty: return
        if not filename.endswith('.csv'): filename += '.csv'
        filepath = self.data_dir / filename
        try:
            df.to_csv(filepath) # 建议保留 index，特别是时间序列
            logger.info(f"💾 CSV Saved: {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save csv {filename}: {e}")

    def generate_html(self):
        """完成并写入 report.html"""
        self.html_content.append("</body></html>")
        report_path = self.report_dir / "report.html"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.html_content))
            logger.info(f"✅ Report Generated: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"❌ Failed to write HTML: {e}")
            return None