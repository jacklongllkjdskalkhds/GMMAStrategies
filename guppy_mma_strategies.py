# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Union
import numpy as np
import talib as ta
import pandas as pd
from datetime import datetime
import sqlite3
import logging
from howtrader.app.cta_strategy import (
    CtaTemplate,
    StopOrder
)

from howtrader.trader.object import TickData, BarData, TradeData, OrderData
from howtrader.trader.utility import BarGenerator, ArrayManager
from decimal import Decimal
from howtrader.trader.constant import Interval, Direction, Offset, Status, OrderType
from howtrader.app.cta_strategy.engine import CtaEngine
from other_documents import data_calculation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("GuppyMMAStrategies.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GuppyMMAStrategies")

class GuppyMMAStrategy(CtaTemplate):
    """
    Guppy Multiple Moving Average (GMMA) Strategy
    基于Guppy均线组的交易策略，包括短期均线组和长期均线组
    结合价格结构、趋势强度、震荡指标等多重条件优化交易
    """
    author = "Claude"

    # 添加数据库相关参数
    db_name = "guppy_strategy.db"

    # 策略参数
    # 短期EMA参数
    short_ema1_period = 3
    short_ema2_period = 5
    short_ema3_period = 8
    short_ema4_period = 10
    short_ema5_period = 12
    short_ema6_period = 15

    # 长期EMA参数
    long_ema1_period = 30
    long_ema2_period = 35
    long_ema3_period = 40
    long_ema4_period = 45
    long_ema5_period = 50
    long_ema6_period = 60

    # 止损百分比
    stop_loss_percent = 2.0

    # 追踪止损
    trailing_stop_enabled = False
    trailing_stop_activation = 1.0  # 追踪止损激活比例 (%)
    trailing_stop_distance = 1.5  # 追踪止损距离 (%)

    # 获利设置
    take_profit_enabled = False
    take_profit_percent = 5.0  # 获利目标百分比

    # 部分止盈策略参数
    first_tp_percent = 2.0  # 第一止盈目标 (%)
    first_tp_size = 50  # 第一止盈比例 (%)

    # 震荡指标过滤
    use_rsi_filter = False
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30

    # 趋势强度评分系统参数
    use_trend_score = True
    adx_threshold = 25  # ADX趋势阈值
    min_trend_score = 50  # 最小趋势强度分数

    # 震荡指标过滤参数
    use_oscillators = True
    macd_fast_length = 12  # MACD快线
    macd_slow_length = 26  # MACD慢线
    macd_signal_length = 9  # MACD信号线

    # 入场优化参数
    use_entry_optimization = True
    required_confirmations = 3  # 满足条件数量
    breakout_bars = 2  # 突破确认K线数

    # 变量
    target_pos = 0
    entry_price = 0.0

    # 是否启用相关功能的参数
    use_trailing_stop = True    # 是否启用追踪止损
    use_partial_tp = True   # 是否启用分批止盈
    use_trend_exit = True   # 是否使用趋势反转平仓
    use_indicator_exit = True   # 是否使用指标反转平仓
    use_cooldown_period = True  # 是否使用交易冷静期

    # 参数映射表
    parameters = [
        "short_ema1", "short_ema2", "short_ema3", "short_ema4", "short_ema5", "short_ema6",
        "long_ema1", "long_ema2", "long_ema3", "long_ema4", "long_ema5", "long_ema6",
        "stop_loss_percent", "trailing_stop_enabled", "trailing_stop_activation", "trailing_stop_distance",
        "take_profit_enabled", "take_profit_percent",
        "use_partial_tp", "first_tp_percent", "first_tp_size",
        "use_rsi_filter", "rsi_period", "rsi_overbought", "rsi_oversold",
        "use_trend_score", "adx_threshold", "min_trend_score",
        "use_oscillators", "macd_fast_length", "macd_slow_length", "macd_signal_length",
        "use_entry_optimization", "required_confirmations", "breakout_bars",
        "db_name", "log_level"  # 添加新参数
    ]

    # 变量映射表
    variables = [
        "target_pos",
        "entry_price",
        "in_long_position",
        "in_short_position",
        "highest_since_entry",
        "lowest_since_entry",
        "trailing_long_stop_level",
        "trailing_short_stop_level",
        "fixed_long_stop_level",
        "fixed_short_stop_level",
        "long_take_profit_level",
        "short_take_profit_level",
        "breakout_level",
        "waiting_for_breakout_confirm",
        "breakout_confirm_count",
        "breakout_direction",
        "win_count",
        "loss_count",
        "current_win_rate",
        "partial_tp_triggered"
    ]

    def __init__(self, cta_engine: CtaEngine, strategy_name: str, vt_symbol: str, setting: dict):
        """
        初始化策略
        """
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # 初始化数据库连接
        self.init_database()

        # 创建K线合成器和K线管理器
        self.bg5 = BarGenerator(self.on_bar, 1, self.on_5min_bar, Interval.HOUR)  # 合成1小时K线
        self.am5 = ArrayManager(size=100)  # 保留更多数据用于计算均线和指标

        self.minute_interval = 60

        # 策略状态变量
        self.in_long_position = False
        self.in_short_position = False
        self.highest_since_entry = 0.0
        self.lowest_since_entry = float('inf')
        self.entry_price = 0.0
        self.trailing_long_stop_level = 0.0
        self.trailing_short_stop_level = 0.0
        self.fixed_long_stop_level = 0.0
        self.fixed_short_stop_level = 0.0
        self.long_take_profit_level = 0.0
        self.short_take_profit_level = 0.0
        self.long_trend_up = None
        self.datetime = None

        # 添加平仓逻辑所需的属性初始化
        self.in_cooldown = False
        self.cooldown_bars_remaining = 0
        self.cooldown_bars = setting.get("cooldown_bars", 5)  # 默认冷静期为5个Bar

        # 追踪止损相关参数
        self.use_trailing_stop = setting.get("use_trailing_stop", False)
        self.trailing_stop_percent = setting.get("trailing_stop_percent", 2.0)

        # 突破相关变量
        self.breakout_level = 0.0
        self.waiting_for_breakout_confirm = False
        self.breakout_confirm_count = 0
        self.breakout_direction = "none"

        # 统计相关变量
        self.win_count = 0
        self.loss_count = 0
        self.current_win_rate = 50.0

        # 部分止盈状态
        self.use_partial_tp = setting.get("use_partial_tp", False)
        self.partial_tp_percent = setting.get("partial_tp_percent", 2.0)
        self.partial_tp_triggered = False

        # 趋势和指标退出相关参数
        self.use_trend_exit = setting.get("use_trend_exit", False)
        self.use_indicator_exit = setting.get("use_indicator_exit", False)
        self.use_cooldown_period = setting.get("use_cooldown_period", False)

        # 趋势持续时间
        self.uptrend_duration = 0
        self.downtrend_duration = 0

        # 价格结构
        self.resistance_levels = []
        self.support_levels = []


        self.order_data = None
        self.trade = None
    def on_init(self):
        """
        策略初始化时调用
        """
        self.write_log("策略初始化")
        self.load_bar(10)  # 加载10天的历史数据

    def on_start(self):
        """
        策略启动时调用
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        策略停止时调用
        """
        self.write_log("策略停止")

        # 关闭数据库连接
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")

        self.put_event()

    def on_tick(self, tick: TickData):
        """
        Tick数据更新
        """
        self.bg.update_tick(tick)

        self.last_price = Decimal(tick.last_price)

    def on_bar(self, bar: BarData):
        """
      通过该函数收到新的1分钟k线推送，必须由用户继承实现
        """
        # print("1分钟的K线数据",bar)
        # 更新K线到时间序列容器中
        self.bg5.update_bar(bar)
        self.put_event()

    def on_5min_bar(self, bar: BarData):
        """
        合成5分钟的K线数据
        """
        logger.info(f"5分钟的Bar: {bar}")
        am5 = self.am5
        am5.update_bar(bar)
        if not am5.inited:
            return
        self.datetime = bar.datetime
        # 增加调试日志
        logger.info(
            f"新K线: 时间={bar.datetime}, 开={bar.open_price}, 高={bar.high_price}, 低={bar.low_price}, 收={bar.close_price}")
        # 计算均线和指标
        self.calculate_indicators()
        # 更新价格结构
        self.update_price_structure()
        # 计算趋势强度评分
        if hasattr(self, 'use_trend_score') and self.use_trend_score:
            self.calculate_trend_score()
        # 检查止损和止盈
        self.manage_positions(bar)
        # 生成交易信号
        self.generate_signals(bar)
        # 数据库记录
        self.log_indicators(bar)
        # 记录状态
        self.put_event()

    def calculate_indicators(self):
        """
        计算技术指标
        """
        # 获取K线数据数组
        close_array = self.am5.close_array
        high_array = self.am5.high_array
        low_array = self.am5.low_array

        # 1. 计算Guppy多重均线系统
        # 短期均线组
        short_ema1_array = ta.EMA(close_array, timeperiod=self.short_ema1_period)
        short_ema2_array = ta.EMA(close_array, timeperiod=self.short_ema2_period)
        short_ema3_array = ta.EMA(close_array, timeperiod=self.short_ema3_period)
        short_ema4_array = ta.EMA(close_array, timeperiod=self.short_ema4_period)
        short_ema5_array = ta.EMA(close_array, timeperiod=self.short_ema5_period)
        short_ema6_array = ta.EMA(close_array, timeperiod=self.short_ema6_period)

        # 长期均线组
        long_ema1_array = ta.EMA(close_array, timeperiod=self.long_ema1_period)
        long_ema2_array = ta.EMA(close_array, timeperiod=self.long_ema2_period)
        long_ema3_array = ta.EMA(close_array, timeperiod=self.long_ema3_period)
        long_ema4_array = ta.EMA(close_array, timeperiod=self.long_ema4_period)
        long_ema5_array = ta.EMA(close_array, timeperiod=self.long_ema5_period)
        long_ema6_array = ta.EMA(close_array, timeperiod=self.long_ema6_period)

        # 获取当前价格
        current_price = close_array[-1]

        # 计算短期均线和长期均线的平均值（用于信号生成）
        self.short_ema1 = short_ema1_array[-1].round(4)
        self.short_ema2 = short_ema2_array[-1].round(4)
        self.short_ema3 = short_ema3_array[-1].round(4)
        self.short_ema4 = short_ema4_array[-1].round(4)
        self.short_ema5 = short_ema5_array[-1].round(4)
        self.short_ema6 = short_ema6_array[-1].round(4)
        self.long_ema1 = long_ema1_array[-1].round(4)
        self.long_ema2 = long_ema2_array[-1].round(4)
        self.long_ema3 = long_ema3_array[-1].round(4)
        self.long_ema4 = long_ema4_array[-1].round(4)
        self.long_ema5 = long_ema5_array[-1].round(4)
        self.long_ema6 = long_ema6_array[-1].round(4)
        self.short_avg = (short_ema1_array[-1] + short_ema2_array[-1] + short_ema3_array[-1] +
                          short_ema4_array[-1] + short_ema5_array[-1] + short_ema6_array[-1]) / 6

        self.long_avg = (long_ema1_array[-1] + long_ema2_array[-1] + long_ema3_array[-1] +
                         long_ema4_array[-1] + long_ema5_array[-1] + long_ema6_array[-1]) / 6

        # 检查短期均线是否趋势一致
        short_emas_aligned_up = (
                short_ema1_array[-1] > short_ema2_array[-1] > short_ema3_array[-1] >
                short_ema4_array[-1] > short_ema5_array[-1] > short_ema6_array[-1]
        )

        short_emas_aligned_down = (
                short_ema1_array[-1] < short_ema2_array[-1] < short_ema3_array[-1] <
                short_ema4_array[-1] < short_ema5_array[-1] < short_ema6_array[-1]
        )

        # 检查长期均线是否趋势一致
        long_emas_aligned_up = (
                long_ema1_array[-1] > long_ema2_array[-1] > long_ema3_array[-1] >
                long_ema4_array[-1] > long_ema5_array[-1] > long_ema6_array[-1]
        )

        long_emas_aligned_down = (
                long_ema1_array[-1] < long_ema2_array[-1] < long_ema3_array[-1] <
                long_ema4_array[-1] < long_ema5_array[-1] < long_ema6_array[-1]
        )

        # 存储多空排列状态
        self.short_emas_aligned = short_emas_aligned_up
        self.short_emas_aligned_down = short_emas_aligned_down
        self.long_emas_aligned = long_emas_aligned_up
        self.long_emas_aligned_down = long_emas_aligned_down

        # 均线交叉信号
        self.prev_short_avg = getattr(self, 'prev_short_avg', self.short_avg)
        self.prev_long_avg = getattr(self, 'prev_long_avg', self.long_avg)

        self.crossover_short_long = self.prev_short_avg <= self.prev_long_avg and self.short_avg > self.long_avg
        self.crossunder_short_long = self.prev_short_avg >= self.prev_long_avg and self.short_avg < self.long_avg

        # 更新上一个值
        self.prev_short_avg = self.short_avg
        self.prev_long_avg = self.long_avg

        # 2. 计算RSI
        if self.use_rsi_filter:
            rsi_array = ta.RSI(close_array, timeperiod=self.rsi_period)
            self.rsi_prev = getattr(self, 'rsi', rsi_array[-1])
            self.rsi = rsi_array[-1]

        # 3. 计算MACD
        if self.use_oscillators:
            macd_array, signal_array, hist_array = ta.MACD(
                close_array,
                fastperiod=self.macd_fast_length,
                slowperiod=self.macd_slow_length,
                signalperiod=self.macd_signal_length
            )

            self.macd_line = macd_array[-1]
            self.macd_signal = signal_array[-1]
            self.macd_hist = hist_array[-1]

            # 计算MACD交叉信号
            self.macd_prev_line = getattr(self, 'macd_line', self.macd_line)
            self.macd_prev_signal = getattr(self, 'macd_signal', self.macd_signal)

            self.macd_cross_up = self.macd_prev_line <= self.macd_prev_signal and self.macd_line > self.macd_signal
            self.macd_cross_down = self.macd_prev_line >= self.macd_prev_signal and self.macd_line < self.macd_signal

            self.macd_above_zero = self.macd_line > 0
            self.macd_below_zero = self.macd_line < 0

            # 更新上一个值
            self.macd_prev_line = self.macd_line
            self.macd_prev_signal = self.macd_signal

        # 4. 计算ADX趋势强度
        if self.use_trend_score:
            adx_array = ta.ADX(high_array, low_array, close_array, timeperiod=14)
            di_plus_array = ta.PLUS_DI(high_array, low_array, close_array, timeperiod=14)
            di_minus_array = ta.MINUS_DI(high_array, low_array, close_array, timeperiod=14)

            self.adx = adx_array[-1]
            self.di_plus = di_plus_array[-1]
            self.di_minus = di_minus_array[-1]

            # 计算趋势强度分数
            di_diff = abs(self.di_plus - self.di_minus)
            self.trend_score = (self.adx + di_diff) / 2

            # 确定强趋势条件
            self.strong_uptrend = self.adx > self.adx_threshold and self.di_plus > self.di_minus
            self.strong_downtrend = self.adx > self.adx_threshold and self.di_minus > self.di_plus

        # 5. 计算震荡市场标志
        # 使用价格在短期均线组附近震荡但没有明确方向作为震荡市场的标志
        price_deviation = 0
        for ema in [short_ema1_array[-1], short_ema2_array[-1], short_ema3_array[-1],
                    short_ema4_array[-1], short_ema5_array[-1], short_ema6_array[-1]]:
            price_deviation += abs(current_price - ema) / current_price

        price_deviation = price_deviation / 6 * 100  # 转换为百分比

        # 如果价格与短期均线组的偏差小于0.5%且ADX低于20，认为是震荡市场
        adx_value = self.adx if hasattr(self, 'adx') else 25
        self.is_choppy_market = price_deviation < 0.5 and adx_value < 20

        # 6. 计算价格结构
        # 检查价格是否在支撑位或阻力位附近
        # 简单地使用前期高低点作为支撑阻力
        lookback = 10
        local_high = max(high_array[-lookback:-1])
        local_low = min(low_array[-lookback:-1])

        tolerance = 0.005  # 0.5%的容差
        self.is_near_resistance = abs(current_price - local_high) / current_price < tolerance
        self.is_near_support = abs(current_price - local_low) / current_price < tolerance

        # 7. 跟踪趋势持续时间
        if not hasattr(self, 'uptrend_duration'):
            self.uptrend_duration = 0
            self.downtrend_duration = 0

        if self.short_avg > self.long_avg:
            self.uptrend_duration += 1
            self.downtrend_duration = 0
        elif self.short_avg < self.long_avg:
            self.downtrend_duration += 1
            self.uptrend_duration = 0

        # 记录一些调试信息
        logger.info(f"技术指标: 短均={self.short_avg:.4f}, 长均={self.long_avg:.4f}, " +
                     f"趋势得分={getattr(self, 'trend_score', 'N/A')}, " +
                     f"是否震荡={self.is_choppy_market}, " +
                     f"强多头={getattr(self, 'strong_uptrend', False)}, " +
                     f"强空头={getattr(self, 'strong_downtrend', False)}")


    def calculate_trend_score(self):
        """
        计算趋势强度评分
        """
        trend_score = 0.0

        # ADX评分 (最多40分)
        trend_score += min(40, self.adx / 50 * 40)

        # 趋势方向一致性 (最多20分)
        direction_consistency = (self.long_trend_up and self.di_plus > self.di_minus) or (
                    not self.long_trend_up and self.di_minus > self.di_plus)
        trend_score += 20 if direction_consistency else 0

        # 均线排列评分 (最多20分)
        good_alignment = (self.long_trend_up and self.short_emas_aligned) or (
                    not self.long_trend_up and self.short_emas_aligned_down)
        trend_score += 20 if good_alignment else 0

        # 价格与均线关系 (最多20分)
        price_above_short_ema = self.am5.close_array[-1] > self.short_avg and self.short_avg > self.long_avg
        price_below_short_ema = self.am5.close_array[-1] < self.short_avg and self.short_avg < self.long_avg
        trend_score += 20 if ((self.long_trend_up and price_above_short_ema) or (
                    not self.long_trend_up and price_below_short_ema)) else 0

        self.trend_score = trend_score

        # 市场状态分类
        self.strong_trend_market = self.trend_score >= 70
        self.moderate_trend_market = 50 <= self.trend_score < 70
        self.weak_trend_market = 30 <= self.trend_score < 50
        self.chop_market = self.trend_score < 30

        logger.info(f"市场状态: 强趋势={self.strong_trend_market}, " +
                     f"中趋势={self.moderate_trend_market}, " +
                     f"弱趋势={self.weak_trend_market}, " +
                     f"震荡市场={self.chop_market}")

    def update_price_structure(self):
        """
        更新价格结构（支撑位和阻力位）
        """
        # 使用简化方法查找最近的高点和低点
        high_array = self.am5.high_array
        low_array = self.am5.low_array
        lookback = 10  # 查看前10根K线

        # 寻找局部高点作为阻力位
        for i in range(-lookback, -1):
            if i + 1 < len(high_array) and i - 1 >= -len(high_array):
                if high_array[i] > high_array[i + 1] and high_array[i] > high_array[i - 1]:
                    if high_array[i] not in self.resistance_levels:
                        self.resistance_levels.append(high_array[i])

        # 寻找局部低点作为支撑位
        for i in range(-lookback, -1):
            if i + 1 < len(low_array) and i - 1 >= -len(low_array):
                if low_array[i] < low_array[i + 1] and low_array[i] < low_array[i - 1]:
                    if low_array[i] not in self.support_levels:
                        self.support_levels.append(low_array[i])

        # 保持数组大小可控
        if len(self.resistance_levels) > 5:
            self.resistance_levels = self.resistance_levels[-5:]
        if len(self.support_levels) > 5:
            self.support_levels = self.support_levels[-5:]

        # 检查是否接近支撑阻力位
        self.is_near_resistance = False
        self.is_near_support = False

        current_close = self.am5.close_array[-1]

        for level in self.resistance_levels:
            if abs(current_close - level) / level * 100 < 1:
                self.is_near_resistance = True
                break

        for level in self.support_levels:
            if abs(current_close - level) / level * 100 < 1:
                self.is_near_support = True
                break

        logger.info(f"价格结构: 是否接近阻力={self.is_near_resistance}, " +
                     f"是否接近支撑={self.is_near_support}")

    def update_trade_stats(self, is_win: bool):
        """
        更新交易统计
        """
        if not hasattr(self, 'total_trades'):
            self.total_trades = 0
            self.win_trades = 0
            self.loss_trades = 0

        self.total_trades += 1
        if is_win:
            self.win_trades += 1
        else:
            self.loss_trades += 1

        win_rate = self.win_trades / self.total_trades * 100 if self.total_trades > 0 else 0
        logger.info(
            f"交易统计: 总交易={self.total_trades}, 盈利={self.win_trades}, 亏损={self.loss_trades}, 胜率={win_rate:.2f}%")

    def generate_signals(self, bar: BarData):
        """
        生成交易信号并处理仓位管理，包括开仓和平仓逻辑
        """
        # ==========处理现有仓位的平仓逻辑==========
        self.manage_positions(bar)

        # 如果在冷静期或已经有仓位，不再生成新信号
        if (hasattr(self, 'in_cooldown') and self.in_cooldown) or self.in_long_position or self.in_short_position:
            return

        # ==========以下是原有的开仓逻辑==========

        # 如果等待突破确认中
        if self.waiting_for_breakout_confirm:
            if self.breakout_direction == "long":
                # 检查是否继续确认突破
                if bar.close_price > self.breakout_level:
                    self.breakout_confirm_count += 1
                    if self.breakout_confirm_count >= self.breakout_bars:
                        self.buy(bar.close_price, 1)
                        self.entry_price = bar.close_price
                        self.target_pos = 1
                        self.in_long_position = True
                        self.in_short_position = False
                        self.highest_since_entry = bar.high_price
                        self.trailing_long_stop_level = 0.0
                        self.fixed_long_stop_level = bar.close_price * (1 - self.stop_loss_percent / 100)
                        self.long_take_profit_level = bar.close_price * (1 + self.take_profit_percent / 100)
                        self.partial_tp_triggered = False
                        self.waiting_for_breakout_confirm = False
                        self.write_log(f"多头突破确认进场: {bar.close_price}")
                        logger.info(f"多头突破确认进场: {bar.close_price}")

                        # 记录开仓事件
                        self.log_position_event(
                            "LONG",
                            bar.close_price,
                            "ENTRY",
                            f"突破确认进场，确认次数: {self.breakout_bars}"
                        )
                else:
                    # 突破失败，重置确认计数
                    self.waiting_for_breakout_confirm = False
                    self.breakout_confirm_count = 0
                    logger.info(f"多头突破确认失败: 当前价格 {bar.close_price} 低于突破水平 {self.breakout_level}")

            elif self.breakout_direction == "short":
                # 检查是否继续确认突破
                if bar.close_price < self.breakout_level:
                    self.breakout_confirm_count += 1
                    if self.breakout_confirm_count >= self.breakout_bars:
                        self.short(bar.close_price, 1)
                        self.entry_price = bar.close_price
                        self.target_pos = -1
                        self.in_short_position = True
                        self.in_long_position = False
                        self.lowest_since_entry = bar.low_price
                        self.trailing_short_stop_level = 0.0
                        self.fixed_short_stop_level = bar.close_price * (1 + self.stop_loss_percent / 100)
                        self.short_take_profit_level = bar.close_price * (1 - self.take_profit_percent / 100)
                        self.partial_tp_triggered = False
                        self.waiting_for_breakout_confirm = False
                        self.write_log(f"空头突破确认进场: {bar.close_price}")
                        logger.info(f"空头突破确认进场: {bar.close_price}")

                        # 记录开仓事件
                        self.log_position_event(
                            "SHORT",
                            bar.close_price,
                            "ENTRY",
                            f"突破确认进场，确认次数: {self.breakout_bars}"
                        )
                else:
                    # 突破失败，重置确认计数
                    self.waiting_for_breakout_confirm = False
                    self.breakout_confirm_count = 0
                    logger.info(f"空头突破确认失败: 当前价格 {bar.close_price} 高于突破水平 {self.breakout_level}")

            return

        # 计算入场条件得分
        long_score = 0
        short_score = 0

        # 1. 均线相关条件
        if self.short_avg > self.long_avg:
            long_score += 1
        else:
            short_score += 1

        if self.short_emas_aligned:
            long_score += 1
        elif self.short_emas_aligned_down:
            short_score += 1

        # 2. 突破相关条件
        if self.crossover_short_long:
            long_score += 1
        elif self.crossunder_short_long:
            short_score += 1

        # 3. 趋势相关条件
        if self.use_trend_score:
            if self.strong_uptrend:
                long_score += 2
            elif self.strong_downtrend:
                short_score += 2

            if self.di_plus > self.di_minus:
                long_score += 1
            elif self.di_minus > self.di_plus:
                short_score += 1

            if self.trend_score >= self.min_trend_score and self.di_plus > self.di_minus:
                long_score += 1
            elif self.trend_score >= self.min_trend_score and self.di_minus > self.di_plus:
                short_score += 1

        # 4. 震荡指标条件
        if self.use_oscillators:
            if self.macd_cross_up and self.macd_above_zero:
                long_score += 1
            elif self.macd_cross_down and self.macd_below_zero:
                short_score += 1

        # 5. RSI条件
        if self.use_rsi_filter:
            if self.rsi < self.rsi_oversold and self.rsi > self.rsi_prev:
                long_score += 1
            elif self.rsi > self.rsi_overbought and self.rsi < self.rsi_prev:
                short_score += 1

        # 6. 价格结构条件
        if self.is_near_support:
            long_score += 1
        elif self.is_near_resistance:
            short_score += 1

        # 7. 趋势持续时间
        if self.uptrend_duration > 5:
            long_score += 1
        elif self.downtrend_duration > 5:
            short_score += 1

        # 避免震荡行情中的错误信号
        if self.is_choppy_market:
            long_score = max(0, long_score - 2)
            short_score = max(0, short_score - 2)

        # 保存得分供数据库记录使用
        self.long_score = long_score
        self.short_score = short_score

        # 检查是否满足最低得分要求
        required_score = self.required_confirmations if self.use_entry_optimization else 3

        # 根据得分和优化设置生成信号
        if long_score >= required_score and long_score > short_score:
            if self.use_entry_optimization and self.breakout_bars > 0:
                # 设置突破确认条件
                self.breakout_level = bar.close_price
                self.waiting_for_breakout_confirm = True
                self.breakout_direction = "long"
                self.breakout_confirm_count = 0
                self.write_log(f"等待多头突破确认: {bar.close_price}, 得分: {long_score}")
                logger.info(f"多头信号生成，等待突破确认: 价格={bar.close_price}, 得分={long_score}/{required_score}")
            else:
                # 直接入场
                self.buy(bar.close_price, 1)
                self.entry_price = bar.close_price
                self.target_pos = 1
                self.in_long_position = True
                self.in_short_position = False
                self.highest_since_entry = bar.high_price
                self.trailing_long_stop_level = 0.0
                self.fixed_long_stop_level = bar.close_price * (1 - self.stop_loss_percent / 100)
                self.long_take_profit_level = bar.close_price * (1 + self.take_profit_percent / 100)
                self.partial_tp_triggered = False
                self.write_log(f"多头直接进场: {bar.close_price}, 得分: {long_score}")
                logger.info(f"多头信号生成，直接入场: 价格={bar.close_price}, 得分={long_score}/{required_score}")

                # 记录开仓事件
                self.log_position_event(
                    "LONG",
                    bar.close_price,
                    "ENTRY",
                    f"直接进场，得分: {long_score}/{required_score}"
                )

        elif short_score >= required_score and short_score > long_score:
            if self.use_entry_optimization and self.breakout_bars > 0:
                # 设置突破确认条件
                self.breakout_level = bar.close_price
                self.waiting_for_breakout_confirm = True
                self.breakout_direction = "short"
                self.breakout_confirm_count = 0
                self.write_log(f"等待空头突破确认: {bar.close_price}, 得分: {short_score}")
                logger.info(f"空头信号生成，等待突破确认: 价格={bar.close_price}, 得分={short_score}/{required_score}")
            else:
                # 直接入场
                self.short(bar.close_price, 1)
                self.entry_price = bar.close_price
                self.target_pos = -1
                self.in_short_position = True
                self.in_long_position = False
                self.lowest_since_entry = bar.low_price
                self.trailing_short_stop_level = 0.0
                self.fixed_short_stop_level = bar.close_price * (1 + self.stop_loss_percent / 100)
                self.short_take_profit_level = bar.close_price * (1 - self.take_profit_percent / 100)
                self.partial_tp_triggered = False
                self.write_log(f"空头直接进场: {bar.close_price}, 得分: {short_score}")
                logger.info(f"空头信号生成，直接入场: 价格={bar.close_price}, 得分={short_score}/{required_score}")

                # 记录开仓事件
                self.log_position_event(
                    "SHORT",
                    bar.close_price,
                    "ENTRY",
                    f"直接进场，得分: {short_score}/{required_score}"
                )
        else:
            logger.info(f"没有交易信号: 多头得分={long_score}, 空头得分={short_score}, 要求得分={required_score}")

    def manage_positions(self, bar: BarData):
        """
        统一的仓位管理函数，处理止损、止盈和平仓逻辑
        """
        # 如果没有仓位，直接返回
        if not (self.in_long_position or self.in_short_position):
            return

        current_price = bar.close_price

        # 计算ATR用于动态止损（如果启用）
        dynamic_stop_loss_percent = self.stop_loss_percent
        if hasattr(self, 'use_dynamic_stop_loss') and self.use_dynamic_stop_loss:
            atr_array = ta.ATR(
                self.am5.high_array,
                self.am5.low_array,
                self.am5.close_array,
                timeperiod=14
            )
            atr_value = atr_array[-1]
            atr_stop_multiplier = 2.0

            # 更灵活的止损计算
            dynamic_stop_loss_percent = min(
                self.stop_loss_percent,
                (atr_value / current_price * 100) * atr_stop_multiplier
            )

            logger.info(
                f"动态止损计算: 固定={self.stop_loss_percent}%, "
                f"基于ATR={dynamic_stop_loss_percent:.2f}%,"
                f" ATR值={atr_value}")

        # 处理多头仓位
        if self.in_long_position:
            # 更新最高价
            self.highest_since_entry = max(self.highest_since_entry, bar.high_price)

            # 设置固定止损（可能是动态的）
            self.fixed_long_stop_level = self.entry_price * (1 - dynamic_stop_loss_percent / 100)

            # 追踪止损逻辑
            if hasattr(self, 'use_trailing_stop') and self.use_trailing_stop:
                activation_threshold_reached = True
                if hasattr(self, 'trailing_stop_activation'):
                    activation_threshold_reached = bar.high_price >= self.entry_price * (
                                1 + self.trailing_stop_activation / 100)

                if activation_threshold_reached:
                    trailing_stop_percent = self.trailing_stop_percent if hasattr(self,
                                                                                  'trailing_stop_percent') else self.trailing_stop_distance
                    new_trailing_stop = self.highest_since_entry * (1 - trailing_stop_percent / 100)

                    if self.trailing_long_stop_level == 0.0 or new_trailing_stop > self.trailing_long_stop_level:
                        old_level = self.trailing_long_stop_level
                        self.trailing_long_stop_level = new_trailing_stop
                        self.write_log(f"更新多头追踪止损: {self.trailing_long_stop_level}")
                        logger.info(f"更新多头追踪止损: {old_level:.4f} -> {self.trailing_long_stop_level:.4f}")

                        # 记录追踪止损更新
                        if hasattr(self, 'log_position_event'):
                            self.log_position_event(
                                "LONG",
                                current_price,
                                "TRAILING_STOP_UPDATE",
                                f"新高: {self.highest_since_entry}, 追踪距离: {trailing_stop_percent}%"
                            )

            # 在强趋势中调整止损（如果适用）
            if hasattr(self, 'strong_uptrend') and self.strong_uptrend and hasattr(self, 'long_avg'):
                old_stop = self.fixed_long_stop_level
                self.fixed_long_stop_level = min(self.fixed_long_stop_level, self.long_avg)
                if old_stop != self.fixed_long_stop_level:
                    logger.info(f"强多头趋势中调整止损: {old_stop:.4f} -> {self.fixed_long_stop_level:.4f}")

            # 设置止盈水平
            take_profit_percent = self.take_profit_percent
            if hasattr(self, 'strong_uptrend') and self.strong_uptrend:
                take_profit_percent = self.take_profit_percent * 1.5
            self.long_take_profit_level = self.entry_price * (1 + take_profit_percent / 100)

            # 确定最终止损水平
            long_stop_level = self.fixed_long_stop_level
            if hasattr(self, 'use_trailing_stop') and self.use_trailing_stop and self.trailing_long_stop_level > 0:
                if self.trailing_long_stop_level > self.fixed_long_stop_level:
                    long_stop_level = self.trailing_long_stop_level

            # 检查止损条件
            exit_long_stop = False
            ignore_stop_loss = False
            if hasattr(self, 'strong_uptrend') and self.strong_uptrend:
                ignore_stop_loss = True  # 强趋势不触发止损

            if not ignore_stop_loss and bar.low_price < long_stop_level:
                exit_long_stop = True

            # 检查止盈条件
            exit_long_take_profit = False
            ignore_take_profit = False
            if hasattr(self, 'strong_uptrend') and self.strong_uptrend and hasattr(self, 'adx') and self.adx > 30:
                ignore_take_profit = True  # 强趋势且ADX高不触发止盈

            if not ignore_take_profit and hasattr(self,
                                                  'take_profit_enabled') and self.take_profit_enabled and bar.high_price > self.long_take_profit_level:
                exit_long_take_profit = True

            # 检查趋势反转条件
            exit_long_trend_reversal = False
            if hasattr(self, 'use_trend_exit') and self.use_trend_exit:
                if (hasattr(self, 'crossunder_short_long') and self.crossunder_short_long and
                        not (hasattr(self, 'strong_uptrend') and self.strong_uptrend)):
                    exit_long_trend_reversal = True
                elif hasattr(self, 'trend_score') and hasattr(self, 'min_trend_score'):
                    if self.trend_score < -self.min_trend_score:
                        exit_long_trend_reversal = True

            # 检查指标反转条件
            exit_long_indicator_reversal = False
            if hasattr(self, 'use_indicator_exit') and self.use_indicator_exit:
                if ((hasattr(self, 'macd_cross_down') and hasattr(self, 'macd_above_zero') and
                     self.macd_cross_down and self.macd_above_zero) or
                        (hasattr(self, 'rsi') and hasattr(self, 'rsi_overbought') and hasattr(self, 'rsi_prev') and
                         self.rsi > self.rsi_overbought and self.rsi < self.rsi_prev)):
                    exit_long_indicator_reversal = True

            # 执行平仓
            exit_reason = None
            if exit_long_stop:
                exit_reason = f"止损触发: 价格={bar.close_price}, 止损线={long_stop_level:.4f}"
            elif exit_long_take_profit:
                exit_reason = f"止盈触发: 价格={bar.close_price}, 止盈线={self.long_take_profit_level:.4f}"
            elif exit_long_trend_reversal:
                exit_reason = "趋势反转平仓"
            elif exit_long_indicator_reversal:
                exit_reason = "指标反转平仓"

            if exit_reason:
                position_size = abs(self.target_pos)
                self.sell(bar.close_price, position_size)
                profit_percent = (bar.close_price / self.entry_price - 1) * 100
                self.write_log(f"多头平仓: {bar.close_price}, 获利/亏损: {profit_percent:.2f}%, 原因: {exit_reason}")
                logger.info(f"多头平仓: {bar.close_price}, 获利/亏损: {profit_percent:.2f}%, 原因: {exit_reason}")

                # 记录交易与平仓事件
                if hasattr(self, 'log_trade'):
                    exit_type = "STOP_LOSS" if exit_long_stop else (
                        "TAKE_PROFIT" if exit_long_take_profit else "TREND_REVERSAL")
                    self.log_trade(
                        "LONG",
                        self.entry_price,
                        bar.close_price,
                        position_size,
                        exit_type,
                        trade_duration=1  # 可以添加实际持仓时间计算
                    )

                if hasattr(self, 'log_position_event'):
                    self.log_position_event(
                        "LONG",
                        bar.close_price,
                        "EXIT",
                        f"{exit_reason}, 获利/亏损: {profit_percent:.2f}%"
                    )

                # 重置仓位状态
                self.in_long_position = False
                self.target_pos = 0
                self.partial_tp_triggered = False

                # 更新交易统计（如果有此功能）
                if hasattr(self, 'update_trade_stats'):
                    self.update_trade_stats(not exit_long_stop)  # 非止损为成功交易

                # 设置冷静期（如果有此功能）
                if hasattr(self, 'use_cooldown_period') and self.use_cooldown_period:
                    self.in_cooldown = True
                    self.cooldown_bars_remaining = self.cooldown_bars
                    self.write_log(f"进入冷静期: {self.cooldown_bars}个周期")
                    logger.info(f"进入冷静期: {self.cooldown_bars}个周期")

                return

            # 分批止盈逻辑
            if ((hasattr(self, 'use_partial_tp') and self.use_partial_tp) or
                (hasattr(self, 'first_tp_percent') and hasattr(self,
                                                               'first_tp_size'))) and not self.partial_tp_triggered:

                first_tp_percent = self.partial_tp_percent if hasattr(self,
                                                                      'partial_tp_percent') else self.first_tp_percent
                first_tp_size = 0.5 if not hasattr(self, 'first_tp_size') else self.first_tp_size

                if current_price > self.entry_price * (1 + first_tp_percent / 100):
                    partial_size = round(abs(self.target_pos) * first_tp_size, 2)
                    if partial_size > 0:
                        self.sell(bar.close_price, partial_size)
                        self.write_log(f"多头部分止盈: {bar.close_price}, 数量: {partial_size}")
                        logger.info(f"多头部分止盈: {bar.close_price}, 数量: {partial_size}")

                        # 更新剩余持仓量
                        self.target_pos -= partial_size
                        self.partial_tp_triggered = True

                        # 可选: 更新止损到成本位
                        old_stop = self.fixed_long_stop_level
                        self.fixed_long_stop_level = max(self.fixed_long_stop_level, self.entry_price)

                        # 记录部分止盈事件
                        if hasattr(self, 'log_position_event'):
                            self.log_position_event(
                                "LONG",
                                bar.close_price,
                                "PARTIAL_EXIT" if not hasattr(self, 'log_trade') else "PARTIAL_TAKE_PROFIT",
                                f"部分止盈: {(bar.close_price / self.entry_price - 1) * 100:.2f}%, 止损更新: {old_stop:.4f} -> {self.fixed_long_stop_level:.4f}"
                            )

                        # 记录部分止盈交易
                        if hasattr(self, 'log_trade'):
                            self.log_trade(
                                "LONG",
                                self.entry_price,
                                bar.close_price,
                                partial_size,
                                "PARTIAL_TAKE_PROFIT",
                                trade_duration=1
                            )

        # 处理空头仓位
        elif self.in_short_position:
            # 更新最低价
            self.lowest_since_entry = min(self.lowest_since_entry, bar.low_price)

            # 设置固定止损（可能是动态的）
            self.fixed_short_stop_level = self.entry_price * (1 + dynamic_stop_loss_percent / 100)

            # 追踪止损逻辑
            if hasattr(self, 'use_trailing_stop') and self.use_trailing_stop:
                activation_threshold_reached = True
                if hasattr(self, 'trailing_stop_activation'):
                    activation_threshold_reached = bar.low_price <= self.entry_price * (
                                1 - self.trailing_stop_activation / 100)

                if activation_threshold_reached:
                    trailing_stop_percent = self.trailing_stop_percent if hasattr(self,
                                                                                  'trailing_stop_percent') else self.trailing_stop_distance
                    new_trailing_stop = self.lowest_since_entry * (1 + trailing_stop_percent / 100)

                    if self.trailing_short_stop_level == 0.0 or new_trailing_stop < self.trailing_short_stop_level:
                        old_level = self.trailing_short_stop_level
                        self.trailing_short_stop_level = new_trailing_stop
                        self.write_log(f"更新空头追踪止损: {self.trailing_short_stop_level}")
                        logger.info(f"更新空头追踪止损: {old_level:.4f} -> {self.trailing_short_stop_level:.4f}")

                        # 记录追踪止损更新
                        if hasattr(self, 'log_position_event'):
                            self.log_position_event(
                                "SHORT",
                                current_price,
                                "TRAILING_STOP_UPDATE",
                                f"新低: {self.lowest_since_entry}, 追踪距离: {trailing_stop_percent}%"
                            )

            # 在强趋势中调整止损（如果适用）
            if hasattr(self, 'strong_downtrend') and self.strong_downtrend and hasattr(self, 'long_avg'):
                old_stop = self.fixed_short_stop_level
                self.fixed_short_stop_level = max(self.fixed_short_stop_level, self.long_avg)
                if old_stop != self.fixed_short_stop_level:
                    logger.info(f"强空头趋势中调整止损: {old_stop:.4f} -> {self.fixed_short_stop_level:.4f}")

            # 设置止盈水平
            take_profit_percent = self.take_profit_percent
            if hasattr(self, 'strong_downtrend') and self.strong_downtrend:
                take_profit_percent = self.take_profit_percent * 1.5
            self.short_take_profit_level = self.entry_price * (1 - take_profit_percent / 100)

            # 确定最终止损水平
            short_stop_level = self.fixed_short_stop_level
            if hasattr(self, 'use_trailing_stop') and self.use_trailing_stop and self.trailing_short_stop_level > 0:
                if self.trailing_short_stop_level < self.fixed_short_stop_level:
                    short_stop_level = self.trailing_short_stop_level

            # 检查止损条件
            exit_short_stop = False
            ignore_stop_loss = False
            if hasattr(self, 'strong_downtrend') and self.strong_downtrend:
                ignore_stop_loss = True  # 强趋势不触发止损

            if not ignore_stop_loss and bar.high_price > short_stop_level:
                exit_short_stop = True

            # 检查止盈条件
            exit_short_take_profit = False
            ignore_take_profit = False
            if hasattr(self, 'strong_downtrend') and self.strong_downtrend and hasattr(self, 'adx') and self.adx > 30:
                ignore_take_profit = True  # 强趋势且ADX高不触发止盈

            if not ignore_take_profit and hasattr(self,
                                                  'take_profit_enabled') and self.take_profit_enabled and bar.low_price < self.short_take_profit_level:
                exit_short_take_profit = True

            # 检查趋势反转条件
            exit_short_trend_reversal = False
            if hasattr(self, 'use_trend_exit') and self.use_trend_exit:
                if (hasattr(self, 'crossover_short_long') and self.crossover_short_long and
                        not (hasattr(self, 'strong_downtrend') and self.strong_downtrend)):
                    exit_short_trend_reversal = True
                elif hasattr(self, 'trend_score') and hasattr(self, 'min_trend_score'):
                    if self.trend_score > self.min_trend_score:
                        exit_short_trend_reversal = True

            # 检查指标反转条件
            exit_short_indicator_reversal = False
            if hasattr(self, 'use_indicator_exit') and self.use_indicator_exit:
                if ((hasattr(self, 'macd_cross_up') and hasattr(self, 'macd_below_zero') and
                     self.macd_cross_up and self.macd_below_zero) or
                        (hasattr(self, 'rsi') and hasattr(self, 'rsi_oversold') and hasattr(self, 'rsi_prev') and
                         self.rsi < self.rsi_oversold and self.rsi > self.rsi_prev)):
                    exit_short_indicator_reversal = True

            # 执行平仓
            exit_reason = None
            if exit_short_stop:
                exit_reason = f"止损触发: 价格={bar.close_price}, 止损线={short_stop_level:.4f}"
            elif exit_short_take_profit:
                exit_reason = f"止盈触发: 价格={bar.close_price}, 止盈线={self.short_take_profit_level:.4f}"
            elif exit_short_trend_reversal:
                exit_reason = "趋势反转平仓"
            elif exit_short_indicator_reversal:
                exit_reason = "指标反转平仓"

            if exit_reason:
                position_size = abs(self.target_pos)
                self.cover(bar.close_price, position_size)
                profit_percent = (1 - bar.close_price / self.entry_price) * 100
                self.write_log(f"空头平仓: {bar.close_price}, 获利/亏损: {profit_percent:.2f}%, 原因: {exit_reason}")
                logger.info(f"空头平仓: {bar.close_price}, 获利/亏损: {profit_percent:.2f}%, 原因: {exit_reason}")

                # 记录交易与平仓事件
                if hasattr(self, 'log_trade'):
                    exit_type = "STOP_LOSS" if exit_short_stop else (
                        "TAKE_PROFIT" if exit_short_take_profit else "TREND_REVERSAL")
                    self.log_trade(
                        "SHORT",
                        self.entry_price,
                        bar.close_price,
                        position_size,
                        exit_type,
                        trade_duration=1  # 可以添加实际持仓时间计算
                    )

                if hasattr(self, 'log_position_event'):
                    self.log_position_event(
                        "SHORT",
                        bar.close_price,
                        "EXIT",
                        f"{exit_reason}, 获利/亏损: {profit_percent:.2f}%"
                    )

                # 重置仓位状态
                self.in_short_position = False
                self.target_pos = 0
                self.partial_tp_triggered = False

                # 更新交易统计（如果有此功能）
                if hasattr(self, 'update_trade_stats'):
                    self.update_trade_stats(not exit_short_stop)  # 非止损为成功交易

                # 设置冷静期（如果有此功能）
                if hasattr(self, 'use_cooldown_period') and self.use_cooldown_period:
                    self.in_cooldown = True
                    self.cooldown_bars_remaining = self.cooldown_bars
                    self.write_log(f"进入冷静期: {self.cooldown_bars}个周期")
                    logger.info(f"进入冷静期: {self.cooldown_bars}个周期")

                return

            # 分批止盈逻辑
            if ((hasattr(self, 'use_partial_tp') and self.use_partial_tp) or
                (hasattr(self, 'first_tp_percent') and hasattr(self,
                                                               'first_tp_size'))) and not self.partial_tp_triggered:

                first_tp_percent = self.partial_tp_percent if hasattr(self,
                                                                      'partial_tp_percent') else self.first_tp_percent
                first_tp_size = 0.5 if not hasattr(self, 'first_tp_size') else self.first_tp_size

                if current_price < self.entry_price * (1 - first_tp_percent / 100):
                    partial_size = round(abs(self.target_pos) * first_tp_size, 2)
                    if partial_size > 0:
                        self.cover(bar.close_price, partial_size)
                        self.write_log(f"空头部分止盈: {bar.close_price}, 数量: {partial_size}")
                        logger.info(f"空头部分止盈: {bar.close_price}, 数量: {partial_size}")

                        # 更新剩余持仓量
                        self.target_pos += partial_size  # 因为空头时target_pos是负数
                        self.partial_tp_triggered = True

                        # 可选: 更新止损到成本位
                        old_stop = self.fixed_short_stop_level
                        self.fixed_short_stop_level = min(self.fixed_short_stop_level, self.entry_price)

                        # 记录部分止盈事件
                        if hasattr(self, 'log_position_event'):
                            self.log_position_event(
                                "SHORT",
                                bar.close_price,
                                "PARTIAL_EXIT" if not hasattr(self, 'log_trade') else "PARTIAL_TAKE_PROFIT",
                                f"部分止盈: {(1 - bar.close_price / self.entry_price) * 100:.2f}%, 止损更新: {old_stop:.4f} -> {self.fixed_short_stop_level:.4f}"
                            )

                        # 记录部分止盈交易
                        if hasattr(self, 'log_trade'):
                            self.log_trade(
                                "SHORT",
                                self.entry_price,
                                bar.close_price,
                                partial_size,
                                "PARTIAL_TAKE_PROFIT",
                                trade_duration=1
                            )

        # 处理冷静期倒计时
        if hasattr(self, 'in_cooldown') and self.in_cooldown:
            self.cooldown_bars_remaining -= 1
            if self.cooldown_bars_remaining <= 0:
                self.in_cooldown = False
                self.write_log("冷静期结束，可以重新生成信号")
                logger.info("冷静期结束，可以重新生成信号")

    def on_order(self, order: OrderData):
        """
        订单更新回调
        """
        self.order_data = order  # 回测时，order_data用于计算trade时的offset。实盘时，不需要。
        logging.info(f"策略推送过来的 order: {order}")
        self.put_event()

    def on_trade(self, trade: TradeData):
        """
        成交更新回调
        """
        self.trade = trade
        if self.pos != 0:
            if trade.direction == Direction.LONG:
                self.long_entry_price = trade.price
                # print(f"多單成交價：{self.long_entry_price}")
            elif trade.direction == Direction.SHORT:
                self.short_entry_price = trade.price

        # 回测时，根据成交数据计算进出场价格以及单笔盈亏。实盘时不需要。
        new_order_trade_data = data_calculation.order_trade_data(order=self.order_data,
                                                                 trade=trade,
                                                                 minute_interval=self.minute_interval)
        # 将成交数据写入数据库
        data_calculation.save_order_trader_data_to_sql(self.db_name, 'order_trade_data', new_order_trade_data)

        logging.info(f"最新的成交: {trade}")
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        停止单更新回调
        """
        pass

    def init_database(self):
        """
        初始化SQLite数据库
        """
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.cursor = self.conn.cursor()

            # 创建指标表
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                bar_time TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                short_ema1 REAL,
                short_ema2 REAL,
                short_ema3 REAL,
                short_ema4 REAL,
                short_ema5 REAL,
                short_ema6 REAL,
                long_ema1 REAL,
                long_ema2 REAL,
                long_ema3 REAL,
                long_ema4 REAL,
                long_ema5 REAL,
                long_ema6 REAL,
                short_avg REAL,
                long_avg REAL,
                short_emas_aligned INTEGER,
                short_emas_aligned_down INTEGER,
                long_emas_aligned INTEGER,
                long_emas_aligned_down INTEGER,
                rsi REAL,
                macd_line REAL,
                macd_signal REAL,
                macd_hist REAL,
                adx REAL,
                di_plus REAL,
                di_minus REAL,
                trend_score REAL,
                is_choppy_market INTEGER,
                strong_uptrend INTEGER,
                strong_downtrend INTEGER,
                long_score INTEGER,
                short_score INTEGER
            )
            ''')

            # 创建交易表
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                profit_loss REAL,
                profit_loss_percent REAL,
                exit_reason TEXT,
                trade_duration INTEGER,
                trade_bars INTEGER
            )
            ''')

            # 创建仓位管理表
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_management (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                direction TEXT,
                price REAL,
                stop_loss_level REAL,
                take_profit_level REAL,
                trailing_stop_level REAL,
                position_size REAL,
                event_type TEXT,
                description TEXT
            )
            ''')

            self.conn.commit()
            logger.info(f"数据库 {self.db_name} 初始化成功")

        except sqlite3.Error as e:
            logger.error(f"数据库初始化错误: {e}")

    def log_indicators(self, bar: BarData):
        """
        记录指标数据到数据库
        """
        try:
            # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_time = self.datetime.strftime("%Y-%m-%d %H:%M:%S")

            # 准备一些需要记录的指标，如果指标不存在则设为None
            rsi_value = getattr(self, 'rsi', None)
            macd_line = getattr(self, 'macd_line', None)
            macd_signal = getattr(self, 'macd_signal', None)
            macd_hist = getattr(self, 'macd_hist', None)

            # 确保这些变量都存在，即使策略配置中没有启用相关功能
            short_ema1 = getattr(self, 'short_ema1', False)
            short_ema2 = getattr(self, 'short_ema2', False)
            short_ema3 = getattr(self, 'short_ema3', False)
            short_ema4 = getattr(self, 'short_ema4', False)
            short_ema5 = getattr(self, 'short_ema5', False)
            long_ema1 = getattr(self, 'long_ema1', False)
            long_ema2 = getattr(self, 'long_ema2', False)
            long_ema3 = getattr(self, 'long_ema3', False)
            long_ema4 = getattr(self, 'long_ema4', False)
            long_ema5 = getattr(self, 'long_ema5', False)
            long_ema6 = getattr(self, 'long_ema6', False)
            short_emas_aligned = int(getattr(self, 'short_emas_aligned', False))
            short_emas_aligned_down = int(getattr(self, 'short_emas_aligned_down', False))
            long_emas_aligned = int(getattr(self, 'long_emas_aligned', False))
            long_emas_aligned_down = int(getattr(self, 'long_emas_aligned_down', False))
            is_choppy_market = int(getattr(self, 'is_choppy_market', False))
            strong_uptrend = int(getattr(self, 'strong_uptrend', False))
            strong_downtrend = int(getattr(self, 'strong_downtrend', False))

            # 插入数据
            self.cursor.execute('''
            INSERT INTO indicators (
                timestamp, bar_time, open, high, low, close, short_ema1, short_ema2, short_ema3,
                short_ema4, short_ema5, short_ema6, long_ema1, long_ema2, long_ema3, long_ema4,
                long_ema5, long_ema6, short_avg, long_avg, 
                short_emas_aligned, short_emas_aligned_down, long_emas_aligned, long_emas_aligned_down,
                rsi, macd_line, macd_signal, macd_hist, adx, di_plus, di_minus,
                trend_score, is_choppy_market, strong_uptrend, strong_downtrend,
                long_score, short_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
             ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                current_time,
                bar.datetime.strftime("%Y-%m-%d %H:%M:%S"),
                bar.open_price,
                bar.high_price,
                bar.low_price,
                bar.close_price,
                self.short_ema1,
                self.short_ema2,
                self.short_ema3,
                self.short_ema4,
                self.short_ema5,
                self.short_ema6,
                self.long_ema1,
                self.long_ema2,
                self.long_ema3,
                self.long_ema4,
                self.long_ema5,
                self.long_ema6,
                self.short_avg,
                self.long_avg,
                short_emas_aligned,
                short_emas_aligned_down,
                long_emas_aligned,
                long_emas_aligned_down,
                rsi_value,
                macd_line,
                macd_signal,
                macd_hist,
                getattr(self, 'adx', None),
                getattr(self, 'di_plus', None),
                getattr(self, 'di_minus', None),
                getattr(self, 'trend_score', None),
                is_choppy_market,
                strong_uptrend,
                strong_downtrend,
                getattr(self, 'long_score', 0),  # 这些变量在generate_signals中计算
                getattr(self, 'short_score', 0)
            ))
            self.conn.commit()

            logger.info(
                f"记录指标: 价格={bar.close_price}, 短均={self.short_avg:.2f}, 长均={self.long_avg:.2f}, 趋势分数={getattr(self, 'trend_score', 'N/A')}")

        except Exception as e:
            logger.error(f"记录指标数据失败: {e}")

    def log_position_event(self, direction: str, price: float, event_type: str, description: str = ""):
        """
        记录仓位管理事件到数据库
        """
        try:
            # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_time = self.datetime.strftime("%Y-%m-%d %H:%M:%S")
            # 确定止损止盈水平
            if direction == "LONG":
                stop_loss_level = self.fixed_long_stop_level
                take_profit_level = getattr(self, 'long_take_profit_level', None)
                trailing_stop_level = self.trailing_long_stop_level
                position_size = abs(self.target_pos) if self.in_long_position else 0
            else:  # SHORT
                stop_loss_level = self.fixed_short_stop_level
                take_profit_level = getattr(self, 'short_take_profit_level', None)
                trailing_stop_level = self.trailing_short_stop_level
                position_size = abs(self.target_pos) if self.in_short_position else 0

            # 插入数据
            self.cursor.execute('''
            INSERT INTO position_management (
                timestamp, direction, price, stop_loss_level, take_profit_level,
                trailing_stop_level, position_size, event_type, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                current_time,
                direction,
                price,
                stop_loss_level,
                take_profit_level,
                trailing_stop_level,
                position_size,
                event_type,
                description
            ))
            self.conn.commit()

            logger.debug(f"仓位事件: {direction} {event_type} @ {price} - {description}")

        except Exception as e:
            logger.error(f"记录仓位事件失败: {e}")

    def log_trade(self, direction: str, entry_price: float, exit_price: float,
                  quantity: float, exit_reason: str, trade_duration: int = 0,):
        """
        记录交易到数据库
        """
        try:
            # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_time = self.datetime.strftime("%Y-%m-%d %H:%M:%S")
            # 计算盈亏
            if direction == "LONG":
                profit_loss = (exit_price - entry_price) * quantity
                profit_loss_percent = (exit_price - entry_price) / entry_price * 100
            else:  # SHORT
                profit_loss = (entry_price - exit_price) * quantity
                profit_loss_percent = (entry_price - exit_price) / entry_price * 100

            # 插入数据
            self.cursor.execute('''
            INSERT INTO trades (
                timestamp, direction, entry_price, exit_price, quantity,
                profit_loss, profit_loss_percent, exit_reason, trade_duration, trade_bars
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                current_time,
                direction,
                entry_price,
                exit_price,
                quantity,
                profit_loss,
                profit_loss_percent,
                exit_reason,
                trade_duration,
                trade_duration  # 交易持续的K线数量，简单地设置为与持续时间相同
            ))
            self.conn.commit()

            logger.info(
                f"交易记录: {direction} 入场={entry_price} 出场={exit_price} PnL={profit_loss:.2f}({profit_loss_percent:.2f}%) - {exit_reason}")

        except Exception as e:
            logger.error(f"记录交易失败: {e}")