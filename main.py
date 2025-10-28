import akshare as ak
import pandas as pd
import logging
import argparse
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict

# 日志配置
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_format = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(current_dir, "logs")
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir, exist_ok=True)

log_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
log_file_path = os.path.join(logs_dir, log_filename)

file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def get_china_stock_list() -> Optional[pd.DataFrame]:
    try:
        stock_list = ak.stock_info_a_code_name()
        logger.debug(f"获取的股票列表预览:\n{stock_list}")
        stock_list.rename(columns={"代码": "code", "名称": "name"}, inplace=True)
        if not stock_list.empty and all(
            col in stock_list.columns for col in ["code", "name"]
        ):
            logger.info(f"成功获取A股股票列表，共 {len(stock_list)} 只股票")
            logger.debug(f"股票列表预览:\n{stock_list}")
            return stock_list
        logger.error("获取的股票列表格式不正确")
        return None
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return None


def load_local_list(max_age_days: int = 7) -> Optional[pd.DataFrame]:
    if os.path.exists("stock.csv"):
        try:
            file_mtime = os.path.getmtime("stock.csv")
            file_age_days = (datetime.now().timestamp() - file_mtime) / (24 * 3600)

            if file_age_days > max_age_days:
                logger.warning(
                    f"本地股票列表已超过 {max_age_days} 天，可能不是最新数据"
                )

            stock_list = pd.read_csv("stock.csv", encoding="utf-8-sig")
            if not all(col in stock_list.columns for col in ["code", "name"]):
                logger.error("本地股票列表格式不正确，缺少 code/name 列")
                return None
            stock_list["code"] = stock_list["code"].astype(str).str.zfill(6)

            logger.info(
                f"从本地文件加载股票列表，共 {len(stock_list)} 只股票（文件距今 {file_age_days:.1f} 天）"
            )
            return stock_list
        except Exception as e:
            logger.error(f"读取本地股票列表失败: {e}")
            return None
    else:
        logger.info("本地股票列表文件不存在")
        return None


def save_list_to_local(stock_list: pd.DataFrame) -> bool:
    try:
        if not all(col in stock_list.columns for col in ["code", "name"]):
            logger.error("无法保存，股票列表缺少 code/name 列")
            return False

        stock_list.to_csv("stock.csv", index=False, encoding="utf-8-sig")
        logger.info("股票列表已保存到 stock.csv 文件")
        return True
    except Exception as e:
        logger.error(f"保存股票列表到本地文件失败: {e}")
        return False


def search_security_in_list(
    security_list: pd.DataFrame, query: str
) -> Optional[pd.Series]:
    """
    在证券列表中搜索证券（股票或ETF）
    """
    if security_list is None or security_list.empty:
        return None

    code_match = security_list[security_list["code"] == query]
    if not code_match.empty:
        return code_match.iloc[0]

    name_match = security_list[security_list["name"] == query]
    if not name_match.empty:
        return name_match.iloc[0]

    fuzzy_match = security_list[
        security_list["name"].str.contains(query, na=False, case=False)
    ]
    if not fuzzy_match.empty:
        count = len(fuzzy_match)
        logger.info(f"找到 {count} 个匹配的证券:")
        for _, row in fuzzy_match.head(5).iterrows():
            security_type = row.get("type", "unknown")
            logger.info(f"  {row['code']} - {row['name']} [{security_type}]")
        if count > 5:
            logger.info(f"  ... 还有 {count - 5} 个结果未显示，请使用更精确的查询")
        if count == 1:
            return fuzzy_match.iloc[0]

    return None


def add_stock_suffix(code: str) -> str:
    code_clean = code.strip()
    if len(code_clean) != 6:
        logger.warning(f"股票代码 {code_clean} 格式不正确（应为6位数字）")
        return code_clean

    prefix = code_clean[:2]
    if prefix in ["60", "68"]:
        return f"{code_clean}.sh"
    elif prefix in ["00", "30"]:
        return f"{code_clean}.sz"
    else:
        logger.warning(f"未知市场的股票代码 {code_clean}，默认使用.sh")
        return f"{code_clean}.sh"


def get_stock_historical_data(symbol: str, years: int) -> Optional[pd.DataFrame]:
    try:
        symbol_with_suffix = add_stock_suffix(symbol)
        logger.info(f"正在获取股票 {symbol_with_suffix} 的历史数据...")

        end_date = datetime.now()
        start_date = datetime(end_date.year - years + 1, 1, 1)

        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")
        logger.info(f"获取日期范围: {start_date_str} 至 {end_date_str}")

        stock_hist = ak.stock_zh_a_hist(
            symbol=str(symbol),
            period="daily",
            start_date=start_date_str,
            end_date=end_date_str,
            adjust="qfq",
        )

        if stock_hist is None or stock_hist.empty:
            logger.error(f"未获取到股票 {symbol_with_suffix} 的历史数据")
            return None

        required_columns = ["日期", "开盘", "最高", "最低", "收盘", "成交量"]
        if not all(col in stock_hist.columns for col in required_columns):
            logger.error(f"历史数据缺少必要列 {required_columns}")
            return None

        logger.info(
            f"成功获取股票 {symbol_with_suffix} 的历史数据，共 {len(stock_hist)} 条记录 "
            f"({stock_hist['日期'].min()} 至 {stock_hist['日期'].max()})"
        )
        return stock_hist
    except Exception as e:
        logger.error(f"获取股票 {symbol} 历史数据失败: {e}")
        return None


def get_etf_list() -> Optional[pd.DataFrame]:
    try:
        etf_list = ak.fund_etf_category_sina(symbol="ETF基金")

        if etf_list is not None and not etf_list.empty:
            etf_list.rename(columns={"代码": "code", "名称": "name"}, inplace=True)
            etf_list = etf_list[["code", "name"]]

            logger.info(f"成功获取ETF列表，共 {len(etf_list)} 只ETF")
            logger.debug(f"ETF列表预览:\n{etf_list}")
            return etf_list
        else:
            logger.warning("获取ETF列表失败或返回空数据")
            return None
    except Exception as e:
        logger.error(f"获取ETF列表失败: {e}")
        return None


def get_etf_historical_data(symbol: str, years: int) -> Optional[pd.DataFrame]:
    try:
        logger.info(f"正在获取ETF {symbol} 的历史数据...")

        end_date = datetime.now()
        start_date = datetime(end_date.year - years + 1, 1, 1)

        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")
        logger.info(f"获取日期范围: {start_date_str} 至 {end_date_str}")

        etf_hist = ak.fund_etf_hist_sina(symbol=symbol)

        if etf_hist is None or etf_hist.empty:
            logger.error(f"未获取到ETF {symbol} 的历史数据")
            return None

        required_columns = ["date", "open", "high", "low", "close", "volume"]
        if not all(col in etf_hist.columns for col in required_columns):
            logger.error(f"ETF历史数据缺少必要列 {required_columns}")
            return None

        etf_hist["date"] = pd.to_datetime(etf_hist["date"])
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        etf_hist = etf_hist[
            (etf_hist["date"] >= start_date) & (etf_hist["date"] <= end_date)
        ]

        etf_hist.rename(
            columns={
                "date": "日期",
                "open": "开盘",
                "high": "最高",
                "low": "最低",
                "close": "收盘",
                "volume": "成交量",
            },
            inplace=True,
        )

        logger.info(
            f"成功获取ETF {symbol} 的历史数据，共 {len(etf_hist)} 条记录 "
            f"({etf_hist['日期'].min()} 至 {etf_hist['日期'].max()})"
        )
        return etf_hist
    except Exception as e:
        logger.error(f"获取ETF {symbol} 历史数据失败: {e}")
        return None


def group_data_by_year(stock_data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    try:
        stock_data["日期"] = pd.to_datetime(stock_data["日期"])
        stock_data["年份"] = stock_data["日期"].dt.year

        yearly_groups = stock_data.groupby("年份")
        yearly_data = {
            year: group.drop(columns=["年份"]) for year, group in yearly_groups
        }
        yearly_data = {k: v for k, v in yearly_data.items() if not v.empty}

        logger.info(f"数据按年份分组完成，共 {len(yearly_data)} 个有效年份")
        return yearly_data
    except Exception as e:
        logger.error(f"按年份分组数据失败: {e}")
        return {}


def calculate_technical_indicators(stock_data: pd.DataFrame) -> dict:
    """
    计算股票的技术指标：MA、VOL_MA、BOLL、OBV、WR、MACD、RSI、CCI
    """
    if stock_data is None or stock_data.empty:
        return {}

    try:
        # 确保数据按日期排序并转换日期格式
        stock_data = stock_data.sort_values("日期").reset_index(drop=True)
        stock_data["日期"] = pd.to_datetime(stock_data["日期"])

        # 提取核心数据列
        high = stock_data["最高"]
        low = stock_data["最低"]
        close = stock_data["收盘"]
        volume = stock_data["成交量"]
        indicators = {}

        # 1. 移动平均线 (MA) - 5/10/20日周期（常用短/中周期）
        def calculate_ma(prices, periods=[5, 10, 20]):
            ma_dict = {}
            for period in periods:
                ma_dict[f"MA{period}"] = prices.rolling(
                    window=period, min_periods=1
                ).mean()
            return ma_dict

        ma_dict = calculate_ma(close)
        indicators.update(ma_dict)

        # 2. 成交量均线 (VOL_MA) - 5/10日周期（判断量能趋势）
        def calculate_vol_ma(volume, periods=[5, 10]):
            vol_ma_dict = {}
            for period in periods:
                vol_ma_dict[f"VOL_MA{period}"] = volume.rolling(
                    window=period, min_periods=1
                ).mean()
            return vol_ma_dict

        vol_ma_dict = calculate_vol_ma(volume)
        indicators.update(vol_ma_dict)

        # 3. 布林带 (BOLL) - 默认20日周期
        def calculate_boll(high, low, close, period=20, dev=2):
            mid_band = close.rolling(
                window=period, min_periods=1
            ).mean()  # 中轨（MA20）
            std = close.rolling(window=period, min_periods=1).std()  # 标准差
            upper_band = mid_band + (std * dev)  # 上轨（中轨+2倍标准差）
            lower_band = mid_band - (std * dev)  # 下轨（中轨-2倍标准差）
            return mid_band, upper_band, lower_band

        indicators["BOLL_MID"], indicators["BOLL_UPPER"], indicators["BOLL_LOWER"] = (
            calculate_boll(high, low, close)
        )

        # 4. 能量潮 (OBV) - 反映资金流向
        def calculate_obv(close, volume):
            obv = pd.Series(0.0, index=close.index)
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i - 1]
            return obv

        indicators["OBV"] = calculate_obv(close, volume)

        # 5. 威廉指标 (WR) - 修复：用真实高/低价计算
        def williams_r(high, low, close, period=14):
            highest_high = high.rolling(window=period, min_periods=1).max()
            lowest_low = low.rolling(window=period, min_periods=1).min()
            wr = -100 * (highest_high - close) / (highest_high - lowest_low)
            return wr

        indicators["WR"] = williams_r(high, low, close)

        # 6. MACD指标
        def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
            exp1 = prices.ewm(span=fast_period, adjust=False).mean()
            exp2 = prices.ewm(span=slow_period, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=signal_period, adjust=False).mean()
            histogram = macd - signal
            return macd, signal, histogram

        indicators["MACD"], indicators["MACD_Signal"], indicators["MACD_Histogram"] = (
            calculate_macd(close)
        )

        # 7. 相对强弱指数 (RSI)
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (
                (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            )
            loss = (
                (-delta.where(delta < 0, 0))
                .rolling(window=period, min_periods=1)
                .mean()
            )
            rs = gain / loss.replace(0, np.nan)  # 避免除零
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # 初始值填充为50（中性）

        indicators["RSI"] = calculate_rsi(close)

        # 8. 商品通道指数 (CCI) - 修复：用真实高/低价计算
        def calculate_cci(high, low, close, period=20):
            tp = (high + low + close) / 3  # 典型价格
            sma_tp = tp.rolling(window=period, min_periods=1).mean()
            mean_deviation = tp.rolling(window=period, min_periods=1).apply(
                lambda x: np.abs(x - x.mean()).mean(), raw=True
            )
            cci = (tp - sma_tp) / (0.015 * mean_deviation)
            return cci.replace([np.inf, -np.inf], np.nan).fillna(0)

        indicators["CCI"] = calculate_cci(high, low, close)

        return indicators
    except Exception as e:
        logger.error(f"计算技术指标时出错: {e}")
        return {}


def display_technical_indicators(indicators: dict, symbol: str, stock_name: str):
    """
    显示技术指标的最新值及实用解读
    """
    if not indicators:
        logger.warning("没有可显示的技术指标")
        return

    try:
        logger.info(f"======= {symbol} {stock_name} 技术指标（最新值）=======")

        # 1. 移动平均线 (MA) - 趋势判断
        logger.info("【趋势类指标】")
        for ma_key in [k for k in indicators.keys() if k.startswith("MA")]:
            latest_ma = (
                indicators[ma_key].iloc[-1] if not indicators[ma_key].empty else "N/A"
            )
            if isinstance(latest_ma, (int, float)):
                # 短期MA（MA5/MA10）与长期MA（MA20）对比
                trend = ""
                if ma_key == "MA5" and "MA20" in indicators:
                    ma20 = indicators["MA20"].iloc[-1]
                    trend = (
                        "→ 短期趋势强于长期"
                        if latest_ma > ma20
                        else "→ 短期趋势弱于长期"
                    )
                logger.info(f"{ma_key}: {latest_ma:.2f} {trend}")

        # 2. 成交量均线 (VOL_MA) - 量能判断
        logger.info("【量能类指标】")
        latest_volume = (
            indicators["VOL_MA5"].iloc[-1] if "VOL_MA5" in indicators else "N/A"
        )
        if isinstance(latest_volume, (int, float)):
            vol_trend = (
                "→ 量能放大"
                if latest_volume > indicators["VOL_MA10"].iloc[-1]
                else "→ 量能缩小"
            )
            logger.info(f"VOL_MA5: {latest_volume:.0f} {vol_trend}")
            logger.info(f"VOL_MA10: {indicators['VOL_MA10'].iloc[-1]:.0f}")

        # 3. 布林带 (BOLL) - 波动区间
        logger.info("【波动类指标】")
        boll_keys = ["BOLL_MID", "BOLL_UPPER", "BOLL_LOWER"]
        if all(k in indicators for k in boll_keys):
            mid = indicators["BOLL_MID"].iloc[-1]
            upper = indicators["BOLL_UPPER"].iloc[-1]
            lower = indicators["BOLL_LOWER"].iloc[-1]
            close = indicators["MA5"].iloc[-1]  # 用MA5近似收盘价
            position = (
                "→ 价格靠近上轨（强市）" if close > mid else "→ 价格靠近下轨（弱市）"
            )
            logger.info(
                f"BOLL中轨: {mid:.2f}, 上轨: {upper:.2f}, 下轨: {lower:.2f} {position}"
            )

        # 4. 能量潮 (OBV) - 资金流向
        if "OBV" in indicators:
            latest_obv = indicators["OBV"].iloc[-1]
            obv_trend = (
                "→ 资金流入"
                if latest_obv > indicators["OBV"].iloc[-5]
                else "→ 资金流出"
            )  # 与5日前对比
            logger.info(f"OBV能量潮: {latest_obv:.0f} {obv_trend}（5日对比）")

        # 5. 其他原有指标
        logger.info("【强弱/趋势类指标】")
        # WR指标（-80超卖，-20超买）
        latest_wr = indicators["WR"].iloc[-1] if not indicators["WR"].empty else "N/A"
        wr_note = ""
        if isinstance(latest_wr, (int, float)):
            wr_note = (
                "→ 接近超卖区"
                if latest_wr < -80
                else "→ 接近超买区" if latest_wr > -20 else ""
            )
            logger.info(f"威廉指标(WR): {latest_wr:.2f} {wr_note}")

        # MACD
        latest_macd = (
            indicators["MACD"].iloc[-1] if not indicators["MACD"].empty else "N/A"
        )
        latest_signal = (
            indicators["MACD_Signal"].iloc[-1]
            if not indicators["MACD_Signal"].empty
            else "N/A"
        )
        latest_histogram = (
            indicators["MACD_Histogram"].iloc[-1]
            if not indicators["MACD_Histogram"].empty
            else "N/A"
        )
        macd_note = (
            "→ 金叉（多头信号）"
            if isinstance(latest_macd, (int, float))
            and isinstance(latest_signal, (int, float))
            and latest_macd > latest_signal
            else ""
        )
        logger.info(f"MACD: {latest_macd:.2f}, 信号线: {latest_signal:.2f} {macd_note}")
        logger.info(f"MACD柱状图: {latest_histogram:.2f}")

        # RSI（30超卖，70超买）
        latest_rsi = (
            indicators["RSI"].iloc[-1] if not indicators["RSI"].empty else "N/A"
        )
        rsi_note = ""
        if isinstance(latest_rsi, (int, float)):
            rsi_note = (
                "→ 超卖区"
                if latest_rsi < 30
                else "→ 超买区" if latest_rsi > 70 else "→ 中性区"
            )
            logger.info(f"相对强弱指数(RSI): {latest_rsi:.2f} {rsi_note}")

        # CCI（+100强势，-100弱势）
        latest_cci = (
            indicators["CCI"].iloc[-1] if not indicators["CCI"].empty else "N/A"
        )
        cci_note = ""
        if isinstance(latest_cci, (int, float)):
            cci_note = (
                "→ 强势区"
                if latest_cci > 100
                else "→ 弱势区" if latest_cci < -100 else "→ 常态区"
            )
            logger.info(f"商品通道指数(CCI): {latest_cci:.2f} {cci_note}")

        logger.info("=" * 60 + "\n")
    except Exception as e:
        logger.error(f"显示技术指标时出错: {e}")


def plot_stock_analysis_chart(
    stock_data: pd.DataFrame, indicators: dict, symbol: str, stock_name: str
) -> None:
    if stock_data.empty or not indicators:
        logger.error("没有数据可供绘制组合分析图")
        return

    try:
        stock_data = stock_data.sort_values("日期").reset_index(drop=True)
        stock_data["日期"] = pd.to_datetime(stock_data["日期"])
        plot_data = stock_data.tail(90).copy()
        plot_indicators = {k: v.tail(90) for k, v in indicators.items()}
        logger.debug(f"用于绘图的数据预览:\n{plot_data}")
        logger.debug(f"用于绘图的指标预览:\n{plot_indicators}")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Technical Analysis Chart ({symbol}) (Recent 90 Days)",
            fontsize=14,
            fontweight="bold",
        )

        # 1. 子图1：价格 + MA5 + MA10 + MA20
        ax1.plot(
            plot_data["日期"],
            plot_data["收盘"],
            label="Closing Price",
            color="#1f77b4",
            linewidth=2,
        )
        if "MA5" in plot_indicators:
            ax1.plot(
                plot_data["日期"],
                plot_indicators["MA5"],
                label="MA5",
                color="#ff7f0e",
                linewidth=1.5,
                alpha=0.8,
            )
        if "MA10" in plot_indicators:
            ax1.plot(
                plot_data["日期"],
                plot_indicators["MA10"],
                label="MA10",
                color="#2ca02c",
                linewidth=1.5,
                alpha=0.8,
            )
        if "MA20" in plot_indicators:
            ax1.plot(
                plot_data["日期"],
                plot_indicators["MA20"],
                label="MA20",
                color="#d62728",
                linewidth=1.5,
                alpha=0.8,
            )
        ax1.set_title("Price and Moving Averages", fontsize=12)
        ax1.set_ylabel("Price (CNY)")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        # 2. 子图2：成交量 + VOL_MA5
        ax2.bar(
            plot_data["日期"],
            plot_data["成交量"],
            label="Volume",
            color="#1f77b4",
            alpha=0.6,
        )
        if "VOL_MA5" in plot_indicators:
            ax2.plot(
                plot_data["日期"],
                plot_indicators["VOL_MA5"],
                label="VOL_MA5",
                color="#ff7f0e",
                linewidth=2,
            )
        ax2.set_title("Volume", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        # 3. 子图3：MACD + 信号线 + 柱状图
        if all(k in plot_indicators for k in ["MACD", "MACD_Signal", "MACD_Histogram"]):
            ax3.plot(
                plot_data["日期"],
                plot_indicators["MACD"],
                label="MACD",
                color="#1f77b4",
                linewidth=2,
            )
            ax3.plot(
                plot_data["日期"],
                plot_indicators["MACD_Signal"],
                label="Signal",
                color="#ff7f0e",
                linewidth=2,
            )
            # MACD柱状图（红涨绿跌）
            colors = [
                "#d62728" if x > 0 else "#2ca02c"
                for x in plot_indicators["MACD_Histogram"]
            ]
            ax3.bar(
                plot_data["日期"],
                plot_indicators["MACD_Histogram"],
                color=colors,
                alpha=0.6,
                label="Histogram",
            )
            ax3.axhline(y=0, color="black", linewidth=0.5, alpha=0.8)
            ax3.set_title("MACD", fontsize=12)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis="x", rotation=45)

        # 4. 子图4：RSI（标注超买超卖区）
        if "RSI" in plot_indicators:
            ax4.plot(
                plot_data["日期"],
                plot_indicators["RSI"],
                label="RSI",
                color="#d62728",
                linewidth=2,
            )
            ax4.axhline(
                y=70, color="red", linewidth=1, alpha=0.7, label="Overbought (70)"
            )
            ax4.axhline(
                y=30, color="green", linewidth=1, alpha=0.7, label="Oversold (30)"
            )
            ax4.set_title("RSI", fontsize=12)
            ax4.set_ylim(0, 100)
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        output_file = f"{logs_dir}/stock_technical_analysis_{symbol}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.show()
        logger.info(f"技术分析组合图已保存为 {output_file}")

    except Exception as e:
        logger.error(f"绘制技术分析图失败: {e}")


def plot_stock_price_comparison(
    yearly_data: Dict[int, pd.DataFrame], symbol: str, stock_name: str
) -> None:
    """
    保留原年度价格对比图，优化显示效果
    """
    if not yearly_data:
        logger.error("没有数据可供绘制年度对比图")
        return

    try:
        plt.figure(figsize=(14, 8))

        years = sorted(yearly_data.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(years)))

        for i, year in enumerate(years):
            data = yearly_data[year].copy()
            if "收盘" not in data.columns or "日期" not in data.columns:
                logger.warning(f"跳过年份 {year}：缺少必要数据列")
                continue
            data["日期"] = pd.to_datetime(data["日期"])
            data["Day of Year"] = data["日期"].dt.dayofyear
            plt.plot(
                data["Day of Year"],
                data["收盘"],
                label=f"{year}",
                color=colors[i],
                marker="o" if len(data) < 100 else None,
                markersize=3,
                linewidth=2,
                alpha=0.9,
            )

        plt.xlabel("Day of Year", fontsize=11)
        plt.ylabel("Stock Price (CNY)", fontsize=11)
        plt.title(
            f"Stock Price Annual Comparison ({symbol})",
            fontsize=12,
            fontweight="bold",
        )
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
        )
        plt.grid(True, alpha=0.3)

        # 月份标签优化
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        month_positions = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        plt.xticks(month_positions, month_names, rotation=30, fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        output_file = f"{logs_dir}/stock_annual_comparison_{symbol}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.show()
        logger.info(f"年度价格对比图已保存为 {output_file}")

    except Exception as e:
        logger.error(f"绘制年度价格对比图失败: {e}")


def merge_lists(
    stock_list: pd.DataFrame, etf_list: pd.DataFrame
) -> Optional[pd.DataFrame]:
    try:
        if stock_list is None and etf_list is None:
            logger.warning("股票列表和ETF列表都为空，无法合并")
            return None

        if stock_list is None:
            logger.info("股票列表为空，仅使用ETF列表")
            etf_list["type"] = "etf"
            return etf_list

        if etf_list is None:
            logger.info("ETF列表为空，仅使用股票列表")
            stock_list["type"] = "stock"
            return stock_list

        stock_list["type"] = "stock"
        etf_list["type"] = "etf"
        merged_list = pd.concat([stock_list, etf_list], ignore_index=True)
        merged_list = merged_list.drop_duplicates(subset=["code"], keep="first")
        merged_list = merged_list.sort_values("code").reset_index(drop=True)
        logger.info(
            f"成功合并列表，共 {len(merged_list)} 只证券（股票 {len(stock_list)} 只，ETF {len(etf_list)} 只）"
        )
        logger.debug(f"合并列表结果：{merged_list}")
        return merged_list
    except Exception as e:
        logger.error(f"合并证券列表时出错: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="查询股票/ETF数据并生成技术分析报告")
    parser.add_argument("query", help="股票/ETF代码或名称")
    parser.add_argument("--update", action="store_true", help="强制更新本地列表")
    parser.add_argument(
        "--years", type=int, default=5, help="获取的历史数据年份数，默认5年"
    )
    parser.add_argument(
        "--plot-tech",
        action="store_true",
        help="生成技术分析组合图",
    )
    args = parser.parse_args()

    query = args.query
    logger.info(f"开始查询: {query}")

    # 加载证券列表
    if args.update:
        logger.info("强制更新本地列表...")
        online_stock_list = get_china_stock_list()
        online_etf_list = get_etf_list()
        online_list = merge_lists(online_stock_list, online_etf_list)
        if online_list is not None:
            save_list_to_local(online_list)
            security_list = online_list
        else:
            logger.error("强制更新失败，尝试加载本地列表")
            security_list = load_local_list()
    else:
        security_list = load_local_list()
        if security_list is None:
            logger.info("尝试从在线数据源获取证券列表...")
            stock_list = get_china_stock_list()
            etf_list = get_etf_list()
            security_list = merge_lists(stock_list, etf_list)
            if security_list is not None:
                save_list_to_local(security_list)
            else:
                logger.error("无法获取证券列表，程序退出")
                return

    # 搜索证券
    security_info = search_security_in_list(security_list, query)
    if security_info is None:
        logger.error(f"未找到匹配的证券: {query}")
        return

    symbol = str(security_info["code"])
    security_name = security_info["name"]
    security_type = security_info.get("type", "stock")  # 默认为股票
    logger.info(f"找到{security_type.upper()}: {symbol} - {security_name}")

    if security_type == "etf":
        security_data = get_etf_historical_data(symbol, args.years)
    else:
        security_data = get_stock_historical_data(symbol, args.years)

    if security_data is None:
        logger.error("无法继续处理，缺少历史数据")
        return
    logger.debug(f"历史数据预览:\n{security_data}")

    # 计算并显示技术指标
    logger.info("正在计算技术指标...")
    technical_indicators = calculate_technical_indicators(security_data)
    display_technical_indicators(technical_indicators, symbol, security_name)

    # 生成技术分析组合图（可选）
    if args.plot_tech:
        logger.info("正在生成技术分析组合图...")
        plot_stock_analysis_chart(
            security_data, technical_indicators, symbol, security_name
        )

    # 生成年度价格对比图（保留原功能）
    yearly_data = group_data_by_year(security_data)
    if yearly_data:
        logger.info("正在生成年度价格对比图...")
        plot_stock_price_comparison(yearly_data, symbol, security_name)
    else:
        logger.error("无法生成年度价格对比图，分组后无有效数据")


if __name__ == "__main__":
    main()
