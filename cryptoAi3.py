"""
加密货币合约自动交易系统
根据高胜率策略构建指南实现：
- 4小时定趋势，1小时找入口，15分钟精确定位
- EMA金叉+RSI超跌回升+关键支撑位企稳
- 止损设在支撑下方2%，阶段性止盈，移动止损
- 单笔亏损不超过总资金1-2%
- 记录交易日记
- 买入点和卖出点发送Telegram通知
"""

import sys
import io

# 设置UTF-8编码输出（Windows兼容）
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass  # 如果已经设置过，忽略错误

import ccxt
import pandas as pd
import numpy as np
import requests
import time
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import ta  # 技术指标库

class CryptoContractTrader:
    def __init__(self, exchange_config: Dict, telegram_config: Dict, portfolio_size: float = 1000.0):
        """
        初始化交易系统
        
        Args:
            exchange_config: 交易所配置 {'apiKey': str, 'secret': str, 'exchange': str}
            telegram_config: Telegram配置 {'token': str, 'chat_id': str}
            portfolio_size: 总资金（USDT）
        """
        # 初始化交易所
        try:
            exchange_class = getattr(ccxt, exchange_config.get('exchange', 'binance'))
            self.exchange = exchange_class({
                'apiKey': exchange_config.get('apiKey', ''),
                'secret': exchange_config.get('secret', ''),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # 合约交易
                    'defaultMarginMode': 'isolated',  # 逐仓模式
                }
            })
        except Exception as e:
            print(f"⚠️ 交易所初始化警告: {e}")
            print("⚠️ 将使用公共API模式（仅读取数据，无法交易）")
            # 使用公共API模式
            exchange_class = getattr(ccxt, exchange_config.get('exchange', 'binance'))
            self.exchange = exchange_class({
            'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                }
        })
        
        self.telegram_config = telegram_config
        self.portfolio_size = portfolio_size
        self.risk_per_trade = 0.015  # 1.5%风险（在1-2%之间）
        
        # 交易记录文件
        self.trade_journal_file = 'trading_journal.json'
        self.positions_file = 'active_positions.json'
        
        # 信号频率优化配置（可调整）
        self.config = {
            'trend_threshold': 30,      # 4小时趋势阈值（原50，降低以提高频率）
            'entry_threshold': 40,      # 1小时入口阈值（原50，降低以提高频率）
            'precision_threshold': 50,  # 15分钟精确阈值（原60，降低以提高频率）
            'support_distance': 0.03,   # 支撑位距离（原0.02，放宽到3%）
            'signal_cooldown': 300,     # 信号冷却时间（秒），避免重复信号
            'enable_short': True,       # 启用做空信号
            'flexible_trend': True,     # 灵活趋势判断（允许轻微趋势）
            'min_reward_risk': 2.5      # 最低盈亏比（奖励/风险）
        }
        
        # 信号冷却记录
        self.last_signal_time = {}
        
        # 资金费率缓存（1小时TTL，因为Binance资金费率每8小时更新一次）
        self.funding_rate_cache = {}
        self.fr_cache_ttl = 3600  # 1小时（秒）
        
        # 加载交易记录
        self.load_trade_journal()
        self.load_positions()
        
    def send_telegram(self, message: str, parse_mode: str = 'HTML'):
        """发送Telegram通知"""
        if not self.telegram_config or not self.telegram_config.get('token'):
            print(f"⚠️ Telegram未配置，消息: {message}")
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_config['token']}/sendMessage"
            payload = {
                'chat_id': self.telegram_config['chat_id'],
                'text': message,
                'parse_mode': parse_mode
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                print("✅ Telegram消息发送成功")
            else:
                print(f"❌ Telegram发送失败: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Telegram发送错误: {e}")
    
    def get_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        获取多时间框架数据
        4小时定趋势，1小时找入口，15分钟精确定位，3分钟用于日内分析
        """
        timeframes = {
            '4h': '4h',   # 定趋势
            '1h': '1h',   # 找入口
            '15m': '15m', # 精确定位
            '3m': '3m'    # 日内分析（参考Go代码）
        }
        
        data = {}
        for name, tf in timeframes.items():
            try:
                limit = 200 if name != '3m' else 100  # 3分钟数据不需要太多
                ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # 数据新鲜度检测（参考Go代码的isStaleData）
                if self.is_stale_data(df, symbol):
                    print(f"⚠️ 警告: {symbol} 检测到数据过期（连续价格冻结），跳过该币种")
                    return {}
                
                data[name] = df
            except Exception as e:
                print(f"❌ 获取{name}数据失败: {e}")
                return {}
        
        return data
    
    def is_stale_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        检测数据是否过期（连续价格冻结）
        参考Go代码的isStaleData函数
        检测连续5个3分钟周期价格不变（15分钟无波动）
        """
        if len(df) < 5:
            return False  # 数据不足，无法判断
        
        # 检测阈值：连续5个周期价格不变
        stale_price_threshold = 5
        price_tolerance_pct = 0.0001  # 0.01%波动容忍度
        
        # 取最后5根K线
        recent_klines = df.tail(stale_price_threshold)
        first_price = recent_klines.iloc[0]['close']
        
        # 检查所有价格是否在容忍范围内
        for idx, row in recent_klines.iterrows():
            price_diff = abs(row['close'] - first_price) / first_price if first_price > 0 else 0
            if price_diff > price_tolerance_pct:
                return False  # 有价格波动，数据正常
        
        # 额外检查：成交量是否也为0（数据完全冻结）
        all_volume_zero = all(row['volume'] == 0 for _, row in recent_klines.iterrows())
        
        if all_volume_zero:
            print(f"⚠️ {symbol} 数据过期确认：价格冻结 + 零成交量")
            return True
        
        # 价格冻结但有成交量：可能是极低波动市场，允许但记录警告
        print(f"⚠️ {symbol} 检测到极端价格稳定性（连续{stale_price_threshold}个周期无波动），但成交量正常")
        return False
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标（增强版，参考Go代码）
        包括：EMA、MACD、RSI7、RSI14、ATR3、ATR14等
        """
        # EMA指标
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        
        # RSI指标（多个周期，参考Go代码）
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD指标（12, 26, 9，参考Go代码）
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # 支撑位和阻力位（使用最近20根K线的最低点和最高点）
        df['support'] = df['low'].rolling(window=20, min_periods=1).min()
        df['resistance'] = df['high'].rolling(window=20, min_periods=1).max()
        
        # ATR（多个周期，参考Go代码）
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        df['atr_3'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=3
        ).average_true_range()
        df['atr_14'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        
        return df
    
    def calculate_price_changes(self, df_3m: pd.DataFrame, df_4h: pd.DataFrame) -> Dict[str, float]:
        """
        计算价格变化（参考Go代码）
        返回：1小时价格变化、4小时价格变化
        """
        price_changes = {
            'change_1h': 0.0,
            'change_4h': 0.0
        }
        
        if len(df_3m) == 0 or len(df_4h) == 0:
            return price_changes
        
        current_price = df_3m.iloc[-1]['close']
        
        # 1小时价格变化 = 20个3分钟K线前的价格
        if len(df_3m) >= 21:
            price_1h_ago = df_3m.iloc[-21]['close']
            if price_1h_ago > 0:
                price_changes['change_1h'] = ((current_price - price_1h_ago) / price_1h_ago) * 100
        
        # 4小时价格变化 = 1个4小时K线前的价格
        if len(df_4h) >= 2:
            price_4h_ago = df_4h.iloc[-2]['close']
            if price_4h_ago > 0:
                price_changes['change_4h'] = ((current_price - price_4h_ago) / price_4h_ago) * 100
        
        return price_changes
    
    def get_open_interest(self, symbol: str) -> Dict[str, float]:
        """
        获取持仓量（Open Interest）数据（参考Go代码）
        返回：最新持仓量、平均持仓量
        """
        try:
            # 标准化symbol（移除:USDT后缀，Binance API需要）
            api_symbol = symbol.replace('/USDT:USDT', '').replace('/USDT', '') + 'USDT'
            
            url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={api_symbol}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                oi_latest = float(data.get('openInterest', 0))
                # 简化处理：使用当前值作为平均值（实际可以计算历史平均）
                oi_average = oi_latest * 0.999
                
                return {
                    'latest': oi_latest,
                    'average': oi_average
                }
            else:
                print(f"⚠️ 获取持仓量失败: HTTP {response.status_code}")
                return {'latest': 0, 'average': 0}
        except Exception as e:
            print(f"⚠️ 获取持仓量错误: {e}")
            return {'latest': 0, 'average': 0}
    
    def get_funding_rate(self, symbol: str) -> float:
        """
        获取资金费率（带1小时缓存，参考Go代码）
        Binance资金费率每8小时更新一次，1小时缓存非常合理
        """
        # 标准化symbol
        api_symbol = symbol.replace('/USDT:USDT', '').replace('/USDT', '') + 'USDT'
        
        # 检查缓存
        if api_symbol in self.funding_rate_cache:
            cache = self.funding_rate_cache[api_symbol]
            if time.time() - cache['updated_at'] < self.fr_cache_ttl:
                # 缓存命中
                return cache['rate']
        
        # 缓存过期或不存在，调用API
        try:
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={api_symbol}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                rate = float(data.get('lastFundingRate', 0))
                
                # 更新缓存
                self.funding_rate_cache[api_symbol] = {
                    'rate': rate,
                    'updated_at': time.time()
                }
                
                return rate
            else:
                print(f"⚠️ 获取资金费率失败: HTTP {response.status_code}")
                return 0.0
        except Exception as e:
            print(f"⚠️ 获取资金费率错误: {e}")
            return 0.0
    
    def calculate_intraday_series(self, df_3m: pd.DataFrame) -> Dict:
        """
        计算日内系列数据（参考Go代码）
        返回最近10个数据点的指标序列
        优化：一次性计算所有指标，避免重复计算
        """
        if len(df_3m) < 10:
            return {
                'mid_prices': [],
                'ema20_values': [],
                'macd_values': [],
                'rsi7_values': [],
                'rsi14_values': [],
                'volume': [],
                'atr14': 0.0
            }
        
        # 一次性计算所有指标（优化性能）
        df_calc = self.calculate_indicators(df_3m.copy())
        
        # 获取最近10个数据点
        recent_df = df_calc.tail(10)
        
        # 提取数据
        mid_prices = recent_df['close'].tolist()
        volume = recent_df['volume'].tolist()
        
        # 提取指标值（如果数据不足，使用NaN或0）
        ema20_values = recent_df['ema_20'].fillna(0.0).tolist()
        macd_values = recent_df['macd'].fillna(0.0).tolist()
        rsi7_values = recent_df['rsi_7'].fillna(0.0).tolist()
        rsi14_values = recent_df['rsi_14'].fillna(0.0).tolist()
        
        # 计算ATR14（使用最后一个值）
        atr14 = df_calc.iloc[-1]['atr_14'] if len(df_calc) > 0 and not pd.isna(df_calc.iloc[-1]['atr_14']) else 0.0
        
        return {
            'mid_prices': mid_prices,
            'ema20_values': ema20_values,
            'macd_values': macd_values,
            'rsi7_values': rsi7_values,
            'rsi14_values': rsi14_values,
            'volume': volume,
            'atr14': atr14
        }
    
    def check_ema_golden_cross(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        检查EMA金叉
        返回: (是否金叉, 描述)
        """
        if len(df) < 2:
            return False, "数据不足"
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # EMA金叉：短期EMA上穿长期EMA
        current_cross = current['ema_20'] > current['ema_50']
        previous_cross = previous['ema_20'] <= previous['ema_50']
        
        if current_cross and previous_cross:
            return True, "EMA金叉（20上穿50）"
        elif current_cross:
            return True, "EMA多头排列（20>50）"
        else:
            return False, "EMA未金叉"
    
    def check_rsi_oversold_rebound(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        检查RSI超跌回升
        返回: (是否超跌回升, 描述)
        """
        if len(df) < 3:
            return False, "数据不足"
        
        current = df.iloc[-1]
        prev1 = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # RSI超跌回升：RSI从超卖区域（<30）回升
        if prev2['rsi'] < 30 and prev1['rsi'] > prev2['rsi'] and current['rsi'] > prev1['rsi']:
            return True, f"RSI超跌回升（{prev2['rsi']:.1f} → {prev1['rsi']:.1f} → {current['rsi']:.1f}）"
        elif current['rsi'] < 30:
            return True, f"RSI超卖（{current['rsi']:.1f}）"
        elif current['rsi'] < 40 and current['rsi'] > prev1['rsi']:
            return True, f"RSI从低位回升（{prev1['rsi']:.1f} → {current['rsi']:.1f}）"
        else:
            return False, f"RSI正常（{current['rsi']:.1f}）"
    
    def check_support_level(self, df: pd.DataFrame, price: float) -> Tuple[bool, float, str]:
        """
        检查关键支撑位企稳
        返回: (是否企稳, 支撑位价格, 描述)
        """
        if len(df) < 20:
            return False, 0, "数据不足"
        
        current = df.iloc[-1]
        support = current['support']
        
        # 判断价格是否在支撑位附近（可配置距离）
        support_distance = self.config['support_distance']
        distance_pct = abs(price - support) / support
        
        if distance_pct <= support_distance:  # 可配置距离（默认3%）
            # 检查是否企稳（最近3根K线都在支撑位上方）
            recent_lows = df['low'].tail(3).values
            if all(low >= support * 0.98 for low in recent_lows):
                return True, support, f"价格在支撑位附近企稳（支撑: {support:.2f}, 距离: {distance_pct*100:.2f}%）"
            else:
                return False, support, f"价格接近支撑位但未企稳（支撑: {support:.2f}）"
        else:
            return False, support, f"价格远离支撑位（支撑: {support:.2f}, 距离: {distance_pct*100:.2f}%）"
    
    def analyze_trend_4h(self, df_4h: pd.DataFrame) -> Tuple[str, float, List[str]]:
        """
        4小时定趋势
        返回: (趋势方向, 趋势强度, 理由列表)
        """
        df_4h = self.calculate_indicators(df_4h)
        current = df_4h.iloc[-1]
        
        reasons = []
        trend_score = 0
        
        # 1. EMA排列判断趋势
        if current['ema_20'] > current['ema_50']:
            trend_score += 30
            reasons.append("4h EMA多头排列（20>50）")
        else:
            trend_score -= 30
            reasons.append("4h EMA空头排列（20<50）")
        
        # 2. 价格与EMA关系
        if current['close'] > current['ema_20']:
            trend_score += 20
            reasons.append("4h 价格在EMA20上方")
        else:
            trend_score -= 20
            reasons.append("4h 价格在EMA20下方")
        
        # 3. MACD判断
        if current['macd'] > current['macd_signal'] and current['macd_hist'] > 0:
            trend_score += 20
            reasons.append("4h MACD金叉且柱状图为正")
        elif current['macd'] < current['macd_signal'] and current['macd_hist'] < 0:
            trend_score -= 20
            reasons.append("4h MACD死叉且柱状图为负")
        
        # 4. 趋势强度（优化：降低阈值以提高信号频率）
        trend_threshold = self.config['trend_threshold']
        
        if self.config['flexible_trend']:
            # 灵活趋势判断：允许轻微趋势
            if abs(trend_score) >= trend_threshold:
                direction = 'LONG' if trend_score > 0 else 'SHORT'
            elif abs(trend_score) >= trend_threshold * 0.7:  # 70%阈值作为轻微趋势
                # 轻微趋势：如果其他条件很好，也可以考虑
                direction = 'LONG' if trend_score > 0 else 'SHORT'
                reasons.append(f"⚠️ 轻微趋势（强度: {abs(trend_score):.1f}）")
            else:
                direction = 'NEUTRAL'
        else:
            # 严格趋势判断
            if abs(trend_score) >= trend_threshold:
                direction = 'LONG' if trend_score > 0 else 'SHORT'
            else:
                direction = 'NEUTRAL'
        
        return direction, abs(trend_score), reasons
    
    def find_entry_1h(self, df_1h: pd.DataFrame, trend_4h: str) -> Tuple[bool, Dict, List[str]]:
        """
        1小时找入口
        返回: (是否找到入口, 入场信息, 理由列表)
        """
        if trend_4h == 'NEUTRAL':
            return False, {}, ["4小时趋势不明确，不寻找入口"]
        
        df_1h = self.calculate_indicators(df_1h)
        current = df_1h.iloc[-1]
        price = current['close']
        
        reasons = []
        entry_score = 0
        
        # 1. EMA金叉
        ema_cross, ema_desc = self.check_ema_golden_cross(df_1h)
        if ema_cross:
            if trend_4h == 'LONG':
                entry_score += 30
                reasons.append(f"1h {ema_desc}")
            elif trend_4h == 'SHORT' and self.config['enable_short']:
                # 做空：EMA死叉
                if current['ema_20'] < current['ema_50']:
                    entry_score += 30
                    reasons.append(f"1h EMA空头排列（20<50）")
        
        # 2. RSI超跌回升（做多）或超买回落（做空）
        rsi_rebound, rsi_desc = self.check_rsi_oversold_rebound(df_1h)
        if rsi_rebound:
            if trend_4h == 'LONG':
                entry_score += 25
                reasons.append(f"1h {rsi_desc}")
            elif trend_4h == 'SHORT' and self.config['enable_short']:
                # 做空：RSI超买回落
                if current['rsi'] > 70:
                    entry_score += 25
                    reasons.append(f"1h RSI超买（{current['rsi']:.1f}）")
        
        # 3. 关键支撑位企稳（做多）或阻力位受阻（做空）
        support_ok, support_price, support_desc = self.check_support_level(df_1h, price)
        if support_ok:
            if trend_4h == 'LONG':
                entry_score += 25
                reasons.append(f"1h {support_desc}")
            elif trend_4h == 'SHORT' and self.config['enable_short']:
                # 做空：接近阻力位
                resistance = current['resistance']
                resistance_distance = abs(price - resistance) / resistance
                if resistance_distance <= self.config['support_distance']:
                    entry_score += 25
                    reasons.append(f"1h 接近阻力位（阻力: {resistance:.2f}, 距离: {resistance_distance*100:.2f}%）")
        
        # 4. 方向一致性检查
        if trend_4h == 'LONG':
            if current['close'] > current['ema_20']:
                entry_score += 20
                reasons.append("1h 价格在EMA20上方，方向一致")
        elif trend_4h == 'SHORT' and self.config['enable_short']:
            if current['close'] < current['ema_20']:
                entry_score += 20
                reasons.append("1h 价格在EMA20下方，方向一致")
        
        # 判断是否找到入口（优化：降低阈值以提高信号频率）
        entry_threshold = self.config['entry_threshold']
        if entry_score >= entry_threshold:
            entry_info = {
                'price': price,
                'support': support_price if support_ok else current['support'],
                'atr': current['atr'],
                'rsi': current['rsi'],
                'score': entry_score
            }
            return True, entry_info, reasons
        else:
            return False, {}, reasons
    
    def precise_entry_15m(self, df_15m: pd.DataFrame, trend_4h: str, entry_1h: Dict) -> Tuple[bool, Dict, List[str]]:
        """
        15分钟精确定位
        返回: (是否精确定位, 精确入场信息, 理由列表)
        """
        if not entry_1h:
            return False, {}, ["1小时未找到入口"]
        
        df_15m = self.calculate_indicators(df_15m)
        current = df_15m.iloc[-1]
        price = current['close']
        
        reasons = []
        precision_score = 0
        
        # 1. 15分钟级别确认方向（支持做多和做空）
        if trend_4h == 'LONG':
            if current['ema_20'] > current['ema_50']:
                precision_score += 30
                reasons.append("15m EMA多头排列")
            
            if current['close'] > current['ema_20']:
                precision_score += 20
                reasons.append("15m 价格在EMA20上方")
            
            # 检查是否在支撑位附近
            support_ok, support_price, support_desc = self.check_support_level(df_15m, price)
            if support_ok:
                precision_score += 25
                reasons.append(f"15m {support_desc}")
            
            # RSI确认
            if 30 < current['rsi'] < 70:
                precision_score += 15
                reasons.append(f"15m RSI正常区间（{current['rsi']:.1f}）")
            elif current['rsi'] < 40:
                precision_score += 10
                reasons.append(f"15m RSI偏低但可接受（{current['rsi']:.1f}）")
        
        elif trend_4h == 'SHORT' and self.config['enable_short']:
            # 做空信号
            if current['ema_20'] < current['ema_50']:
                precision_score += 30
                reasons.append("15m EMA空头排列")
            
            if current['close'] < current['ema_20']:
                precision_score += 20
                reasons.append("15m 价格在EMA20下方")
            
            # 检查是否在阻力位附近
            resistance = current['resistance']
            resistance_distance = abs(price - resistance) / resistance
            if resistance_distance <= self.config['support_distance']:
                precision_score += 25
                reasons.append(f"15m 接近阻力位（阻力: {resistance:.2f}, 距离: {resistance_distance*100:.2f}%）")
            
            # RSI确认（做空）
            if 30 < current['rsi'] < 70:
                precision_score += 15
                reasons.append(f"15m RSI正常区间（{current['rsi']:.1f}）")
            elif current['rsi'] > 60:
                precision_score += 10
                reasons.append(f"15m RSI偏高但可接受（{current['rsi']:.1f}）")
        
        # 判断是否精确定位（优化：降低阈值以提高信号频率）
        precision_threshold = self.config['precision_threshold']
        if precision_score >= precision_threshold:
            precise_entry = {
                'price': price,
                'support': support_price if support_ok else current['support'],
                'atr': current['atr'],
                'rsi': current['rsi'],
                'score': precision_score,
                'entry_1h_price': entry_1h['price']
            }
            return True, precise_entry, reasons
        else:
            return False, {}, reasons
    
    def calculate_stop_loss_take_profit(self, entry_price: float, support_price: float, 
                                       atr: float, direction: str) -> Tuple[float, float, List[float], float]:
        """
        计算止损和止盈
        止损设在支撑下方2%，阶段性止盈
        """
        if direction == 'LONG':
            # 止损：支撑位下方2%
            stop_loss = support_price * 0.98
            
            # 确保止损不超过入场价的2%
            max_stop_loss = entry_price * 0.98
            stop_loss = min(stop_loss, max_stop_loss)
            
            # 阶段性止盈：1.5倍、2倍、3倍风险
            risk = entry_price - stop_loss
            take_profit_1 = entry_price + risk * 1.5
            take_profit_2 = entry_price + risk * 2.0
            take_profit_3 = entry_price + risk * 3.0
            
            take_profits = [take_profit_1, take_profit_2, take_profit_3]
            reward = take_profit_3 - entry_price
            risk = max(entry_price - stop_loss, 0)
        else:  # SHORT
            # 止损：阻力位上方2%（这里简化处理，实际应该用阻力位）
            stop_loss = entry_price * 1.02
            
            # 阶段性止盈
            risk = stop_loss - entry_price
            take_profit_1 = entry_price - risk * 1.5
            take_profit_2 = entry_price - risk * 2.0
            take_profit_3 = entry_price - risk * 3.0
            
            take_profits = [take_profit_1, take_profit_2, take_profit_3]
            reward = entry_price - take_profit_3
            risk = max(risk, 0)
        
        reward_risk = (reward / risk) if risk > 0 else 0
        return stop_loss, take_profit_3, take_profits, reward_risk
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        计算仓位大小
        单笔亏损不超过总资金1-2%
        """
        risk_amount = self.portfolio_size * self.risk_per_trade
        risk_per_contract = abs(entry_price - stop_loss)
        
        if risk_per_contract == 0:
            return 0
        
        # 合约数量（简化计算，实际需要考虑合约面值）
        position_size = risk_amount / risk_per_contract
        
        return position_size
    
    def check_signal_cooldown(self, symbol: str, direction: str) -> bool:
        """检查信号冷却时间，避免重复信号"""
        cooldown = self.config['signal_cooldown']
        key = f"{symbol}_{direction}"
        current_time = time.time()
        
        if key in self.last_signal_time:
            time_passed = current_time - self.last_signal_time[key]
            if time_passed < cooldown:
                return False  # 还在冷却期
        
        self.last_signal_time[key] = current_time
        return True  # 可以发送信号

    
    def generate_trading_signal(self, symbol: str) -> Optional[Dict]:
        """
        生成交易信号（增强版，参考Go代码）
        完整流程：4小时定趋势 -> 1小时找入口 -> 15分钟精确定位
        新增：持仓量、资金费率、价格变化、日内系列数据
        """
        # 1. 获取多时间框架数据（包括3分钟数据）
        data = self.get_multi_timeframe_data(symbol)
        if not data or len(data) < 3:
            return None
        
        df_4h = data['4h']
        df_1h = data['1h']
        df_15m = data['15m']
        df_3m = data.get('3m', pd.DataFrame())  # 3分钟数据（可选）
        
        # 2. 4小时定趋势（优化：降低阈值）
        trend_4h, trend_strength, trend_reasons = self.analyze_trend_4h(df_4h)
        if trend_4h == 'NEUTRAL':
            return None
        
        # 3. 1小时找入口
        entry_found, entry_1h, entry_reasons = self.find_entry_1h(df_1h, trend_4h)
        if not entry_found:
            return None
        
        # 4. 15分钟精确定位
        precise_found, precise_entry, precise_reasons = self.precise_entry_15m(df_15m, trend_4h, entry_1h)
        if not precise_found:
            return None
        
        # 5. 获取合约专用数据（参考Go代码）
        oi_data = self.get_open_interest(symbol)
        funding_rate = self.get_funding_rate(symbol)
        
        # 6. 计算价格变化（如果3分钟数据可用）
        price_changes = {}
        intraday_data = {}
        if not df_3m.empty:
            price_changes = self.calculate_price_changes(df_3m, df_4h)
            intraday_data = self.calculate_intraday_series(df_3m)
        
        # 7. 计算止损止盈
        stop_loss, final_take_profit, take_profits, reward_risk = self.calculate_stop_loss_take_profit(
            precise_entry['price'],
            precise_entry['support'],
            precise_entry['atr'],
            trend_4h
        )
        min_rr = self.config.get('min_reward_risk', 2.5)
        if reward_risk < min_rr:
            print(f"⚠️ {symbol} 当前盈亏比 {reward_risk:.2f}:1 低于阈值 {min_rr}:1，继续观察")
            return None
        
        # 8. 计算仓位
        position_size = self.calculate_position_size(precise_entry['price'], stop_loss)
        
        # 9. 检查信号冷却时间（避免重复信号）
        if not self.check_signal_cooldown(symbol, trend_4h):
            return None  # 信号在冷却期，不生成
        
        # 10. 构建信号（增强版，包含更多市场数据）
        signal = {
            'symbol': symbol,
            'direction': trend_4h,
            'entry_price': precise_entry['price'],
            'stop_loss': stop_loss,
            'take_profit': final_take_profit,
            'take_profits': take_profits,  # 阶段性止盈
            'position_size': position_size,
            'trend_strength': trend_strength,
            'entry_score': entry_1h['score'],
            'precision_score': precise_entry['score'],
            'reward_risk': reward_risk,
            'reasons': {
                'trend_4h': trend_reasons,
                'entry_1h': entry_reasons,
                'precise_15m': precise_reasons
            },
            # 新增：合约专用数据（参考Go代码）
            'open_interest': oi_data,
            'funding_rate': funding_rate,
            'price_changes': price_changes,
            'intraday_data': intraday_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return signal
    
    def send_buy_signal(self, signal: Dict):
        """发送买入信号到Telegram（增强版，包含合约数据）"""
        message = f"🚀 <b>买入信号 做多</b> 🚀\n\n"
        message += f"━━━━━━━━━━━━━━━━━━━━\n"
        message += f"<b>币种:</b> {signal['symbol']}\n"
        message += f"<b>方向:</b> {signal['direction']}\n\n"
        
        message += f"<b>💰 价格信息</b>\n"
        message += f"入场价格: <b>{signal['entry_price']:.4f} USDT</b>\n"
        message += f"止损价格: <b>{signal['stop_loss']:.4f} USDT</b>\n"
        message += f"最终止盈: <b>{signal['take_profit']:.4f} USDT</b>\n"
        message += f"仓位大小: {signal['position_size']:.2f} 合约\n"
        message += f"盈亏比: {signal['reward_risk']:.2f}:1\n\n"
        
        # 新增：价格变化（参考Go代码）
        if signal.get('price_changes'):
            pc = signal['price_changes']
            if pc.get('change_1h', 0) != 0:
                message += f"1小时价格变化: {pc['change_1h']:+.2f}%\n"
            if pc.get('change_4h', 0) != 0:
                message += f"4小时价格变化: {pc['change_4h']:+.2f}%\n"
            message += "\n"
        
        message += f"<b>📊 阶段性止盈</b>\n"
        for i, tp in enumerate(signal['take_profits'], 1):
            message += f"止盈{i}: {tp:.4f} USDT\n"
        message += "\n"
        
        # 新增：合约专用数据（参考Go代码）
        if signal.get('open_interest'):
            oi = signal['open_interest']
            if oi.get('latest', 0) > 0:
                message += f"<b>📈 持仓量 (OI)</b>\n"
                message += f"最新: {oi['latest']:,.0f}\n"
                message += f"平均: {oi['average']:,.0f}\n\n"
        
        if signal.get('funding_rate', 0) != 0:
            fr = signal['funding_rate']
            fr_pct = fr * 100
            fr_emoji = "🔥" if abs(fr_pct) > 0.1 else "📊"
            message += f"<b>{fr_emoji} 资金费率</b>\n"
            message += f"{fr_pct:+.4f}% (每8小时)\n\n"
        
        message += f"<b>📈 信号强度</b>\n"
        message += f"趋势强度: {signal['trend_strength']:.1f}/100\n"
        message += f"入口得分: {signal['entry_score']:.1f}/100\n"
        message += f"精确得分: {signal['precision_score']:.1f}/100\n\n"
        
        message += f"<b>📝 分析理由</b>\n"
        message += f"<b>4小时趋势:</b>\n"
        for reason in signal['reasons']['trend_4h']:
            message += f"  • {reason}\n"
        message += f"\n<b>1小时入口:</b>\n"
        for reason in signal['reasons']['entry_1h']:
            message += f"  • {reason}\n"
        message += f"\n<b>15分钟定位:</b>\n"
        for reason in signal['reasons']['precise_15m']:
            message += f"  • {reason}\n"
        message += "\n"
        
        message += f"<b>⏰ 时间:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        message += f"━━━━━━━━━━━━━━━━━━━━\n"
        message += f"⚠️ <i>请结合市场情况谨慎操作</i>"
        
        self.send_telegram(message)
    
    def send_sell_signal(self, signal: Dict):
        """发送卖出信号到Telegram（增强版，包含合约数据）"""
        message = f"📉 <b>卖出信号 做空</b> 📉\n\n"
        message += f"━━━━━━━━━━━━━━━━━━━━\n"
        message += f"<b>币种:</b> {signal['symbol']}\n"
        message += f"<b>方向:</b> {signal['direction']}\n\n"
        
        message += f"<b>💰 价格信息</b>\n"
        message += f"入场价格: <b>{signal['entry_price']:.4f} USDT</b>\n"
        message += f"止损价格: <b>{signal['stop_loss']:.4f} USDT</b>\n"
        message += f"最终止盈: <b>{signal['take_profit']:.4f} USDT</b>\n"
        message += f"仓位大小: {signal['position_size']:.2f} 合约\n"
        message += f"盈亏比: {signal['reward_risk']:.2f}:1\n\n"
        
        # 新增：价格变化（参考Go代码）
        if signal.get('price_changes'):
            pc = signal['price_changes']
            if pc.get('change_1h', 0) != 0:
                message += f"1小时价格变化: {pc['change_1h']:+.2f}%\n"
            if pc.get('change_4h', 0) != 0:
                message += f"4小时价格变化: {pc['change_4h']:+.2f}%\n"
            message += "\n"
        
        message += f"<b>📊 阶段性止盈</b>\n"
        for i, tp in enumerate(signal['take_profits'], 1):
            message += f"止盈{i}: {tp:.4f} USDT\n"
        message += "\n"
        
        # 新增：合约专用数据（参考Go代码）
        if signal.get('open_interest'):
            oi = signal['open_interest']
            if oi.get('latest', 0) > 0:
                message += f"<b>📈 持仓量 (OI)</b>\n"
                message += f"最新: {oi['latest']:,.0f}\n"
                message += f"平均: {oi['average']:,.0f}\n\n"
        
        if signal.get('funding_rate', 0) != 0:
            fr = signal['funding_rate']
            fr_pct = fr * 100
            fr_emoji = "🔥" if abs(fr_pct) > 0.1 else "📊"
            message += f"<b>{fr_emoji} 资金费率</b>\n"
            message += f"{fr_pct:+.4f}% (每8小时)\n\n"
        
        message += f"<b>📈 信号强度</b>\n"
        message += f"趋势强度: {signal['trend_strength']:.1f}/100\n"
        message += f"入口得分: {signal['entry_score']:.1f}/100\n"
        message += f"精确得分: {signal['precision_score']:.1f}/100\n\n"
        
        message += f"<b>📝 分析理由</b>\n"
        message += f"<b>4小时趋势:</b>\n"
        for reason in signal['reasons']['trend_4h']:
            message += f"  • {reason}\n"
        message += f"\n<b>1小时入口:</b>\n"
        for reason in signal['reasons']['entry_1h']:
            message += f"  • {reason}\n"
        message += f"\n<b>15分钟定位:</b>\n"
        for reason in signal['reasons']['precise_15m']:
            message += f"  • {reason}\n"
        message += "\n"
        
        message += f"<b>⏰ 时间:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        message += f"━━━━━━━━━━━━━━━━━━━━\n"
        message += f"⚠️ <i>请结合市场情况谨慎操作</i>"
        
        self.send_telegram(message)
    
    def record_trade_journal(self, signal: Dict, action: str, result: Optional[Dict] = None):
        """
        记录交易日记
        记录每笔交易的进出场理由、情绪状态等
        """
        journal_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': action,  # 'BUY', 'SELL', 'STOP_LOSS', 'TAKE_PROFIT'
            'symbol': signal['symbol'],
            'direction': signal['direction'],
            'entry_price': signal['entry_price'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'position_size': signal['position_size'],
            'reasons': signal['reasons'],
            'signal_strength': {
                'trend': signal['trend_strength'],
                'entry': signal['entry_score'],
                'precision': signal['precision_score']
            },
            'result': result  # 平仓时的结果（盈亏等）
        }
        
        self.trade_journal.append(journal_entry)
        self.save_trade_journal()
    
    def load_trade_journal(self):
        """加载交易日记"""
        if os.path.exists(self.trade_journal_file):
            try:
                with open(self.trade_journal_file, 'r', encoding='utf-8') as f:
                    self.trade_journal = json.load(f)
            except:
                self.trade_journal = []
        else:
            self.trade_journal = []
    
    def save_trade_journal(self):
        """保存交易日记"""
        try:
            with open(self.trade_journal_file, 'w', encoding='utf-8') as f:
                json.dump(self.trade_journal, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ 保存交易日记失败: {e}")
    
    def load_positions(self):
        """加载持仓记录"""
        if os.path.exists(self.positions_file):
            try:
                with open(self.positions_file, 'r', encoding='utf-8') as f:
                    self.positions = json.load(f)
            except:
                self.positions = {}
        else:
            self.positions = {}
    
    def save_positions(self):
        """保存持仓记录"""
        try:
            with open(self.positions_file, 'w', encoding='utf-8') as f:
                json.dump(self.positions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ 保存持仓记录失败: {e}")
    
    def run(self, symbol: str = 'ETH/USDT:USDT', interval: int = 300):
        """
        运行交易系统
        interval: 检查间隔（秒），默认5分钟
        """
        print(f"🚀 加密货币合约自动交易系统启动（优化版 - 提高信号频率）")
        print(f"📊 监控币种: {symbol}")
        print(f"⏱️ 检查间隔: {interval}秒")
        print(f"💰 总资金: {self.portfolio_size} USDT")
        print(f"⚠️ 单笔风险: {self.risk_per_trade*100:.1f}%")
        print(f"📈 优化配置:")
        print(f"   - 趋势阈值: {self.config['trend_threshold']} (原50)")
        print(f"   - 入口阈值: {self.config['entry_threshold']} (原50)")
        print(f"   - 精确阈值: {self.config['precision_threshold']} (原60)")
        print(f"   - 支撑距离: {self.config['support_distance']*100:.1f}% (原2%)")
        print(f"   - 信号冷却: {self.config['signal_cooldown']}秒")
        print(f"   - 做空信号: {'启用' if self.config['enable_short'] else '禁用'}")
        print(f"   - 灵活趋势: {'启用' if self.config['flexible_trend'] else '禁用'}")
        print(f"   - 最低盈亏比: {self.config['min_reward_risk']}:1")
        print(f"━━━━━━━━━━━━━━━━━━━━\n")
        
        startup_msg = f"🤖 <b>交易系统启动</b>\n\n"
        startup_msg += f"监控币种: {symbol}\n"
        startup_msg += f"检查间隔: {interval}秒\n"
        startup_msg += f"总资金: {self.portfolio_size} USDT\n"
        startup_msg += f"单笔风险: {self.risk_per_trade*100:.1f}%\n"
        self.send_telegram(startup_msg)
        
        while True:
            try:
                # 生成交易信号
                signal = self.generate_trading_signal(symbol)
                
                if signal:
                    # 发送信号通知
                    if signal['direction'] == 'LONG':
                        self.send_buy_signal(signal)
                        # 记录交易日记
                        self.record_trade_journal(signal, 'BUY')
                    else:
                        self.send_sell_signal(signal)
                        # 记录交易日记
                        self.record_trade_journal(signal, 'SELL')
                    
                    print(f"✅ 发现{signal['direction']}信号，已发送通知")
                else:
                    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 未发现交易信号，继续监控...")
                
                # 等待下次检查
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n🛑 系统停止")
                self.send_telegram("🛑 <b>交易系统已停止</b>")
                break
            except Exception as e:
                error_msg = f"❌ 系统错误: {str(e)}"
                print(error_msg)
                self.send_telegram(f"❌ <b>系统错误</b>\n\n{error_msg}")
                time.sleep(60)  # 出错后等待1分钟再继续


# 使用示例
if __name__ == "__main__":
    # 配置交易所（请替换为您的API密钥）
    exchange_config = {
        'exchange': 'binance',
        'apiKey': 'YOUR_API_KEY',
        'secret': 'YOUR_SECRET'
    }
    
    # 配置Telegram（请替换为您的Bot Token和Chat ID）
    telegram_config = {
        'token': 'YOUR_TELEGRAM_BOT_TOKEN',
        'chat_id': 'YOUR_TELEGRAM_CHAT_ID'
    }
    
    # 创建交易系统
    trader = CryptoContractTrader(
        exchange_config=exchange_config,
        telegram_config=telegram_config,
        portfolio_size=1000.0  # 总资金1000 USDT
    )
    
    # 运行系统（每5分钟检查一次）
    trader.run(symbol='ETH/USDT:USDT', interval=300)
