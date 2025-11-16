import ccxt
import pandas as pd
import talib
import requests
import time
import sqlite3
from datetime import datetime


class CryptoAutoTrader:
    def __init__(self, exchange_id='binance', telegram_config=None):
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': 'YOUR_API_KEY',
            'secret': 'YOUR_SECRET',
            'enableRateLimit': True,
        })

        self.telegram_config = telegram_config
        self.db_conn = sqlite3.connect('trading_log.db')

    def send_telegram_alert(self, message):
        """发送Telegram通知"""
        if self.telegram_config:
            url = f"https://api.telegram.org/bot{self.telegram_config['token']}/sendMessage"
            data = {
                "chat_id": self.telegram_config['chat_id'],
                "text": message,
                "parse_mode": "Markdown"
            }
            try:
                requests.post(url, data=data, timeout=10)
            except Exception as e:
                print(f"Telegram发送失败: {e}")

    def get_multi_timeframe_data(self, symbol, timeframes=['1h', '4h', '1d']):
        """获取多时间框架数据"""
        data = {}
        for tf in timeframes:
            ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            data[tf] = df
        return data

    def calculate_indicators(self, df):
        """计算技术指标"""
        # 趋势指标
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)

        # 动量指标
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])

        # 波动率指标
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)

        # 支撑阻力
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()

        return df

    def generate_signal(self, symbol):
        """生成交易信号"""
        data = self.get_multi_timeframe_data(symbol)
        signals = []

        for tf, df in data.items():
            df = self.calculate_indicators(df)
            current = df.iloc[-1]
            previous = df.iloc[-2]

            signal = {
                'timeframe': tf,
                'timestamp': datetime.now(),
                'price': current['close'],
                'action': 'HOLD',
                'confidence': 0,
                'reason': []
            }

            # 趋势判断 (权重: 40%)
            trend_score = 0
            if current['ema_20'] > current['ema_50'] > current['ema_200']:
                trend_score += 1
                signal['reason'].append(f"{tf} EMA多头排列")
            if current['close'] > current['ema_20']:
                trend_score += 0.5
            if current['close'] > current['bb_middle']:
                trend_score += 0.5

            # 动量判断 (权重: 30%)
            momentum_score = 0
            if 30 < current['rsi'] < 70:
                momentum_score += 1
                signal['reason'].append(f"{tf} RSI正常区间")
            elif current['rsi'] < 30:
                momentum_score += 1.5
                signal['reason'].append(f"{tf} RSI超卖")
            if current['macd'] > current['macd_signal']:
                momentum_score += 1
                signal['reason'].append(f"{tf} MACD金叉")

            # 位置判断 (权重: 30%)
            position_score = 0
            support_distance = abs(current['close'] - current['support']) / current['close']
            resistance_distance = abs(current['close'] - current['resistance']) / current['close']

            if support_distance < 0.02:  # 接近支撑
                position_score += 1.5
                signal['reason'].append(f"{tf} 接近支撑位")
            elif resistance_distance < 0.02:  # 接近阻力
                position_score -= 1

            # 综合评分
            total_score = (trend_score * 0.4 + momentum_score * 0.3 + position_score * 0.3)
            signal['confidence'] = total_score

            if total_score >= 1.5:
                signal['action'] = 'BUY'
            elif total_score <= -1:
                signal['action'] = 'SELL'

            signals.append(signal)

        return signals

    def execute_trading_decision(self, symbol, signals):
        """执行交易决策"""
        # 综合所有时间框架信号
        buy_signals = [s for s in signals if s['action'] == 'BUY']
        sell_signals = [s for s in signals if s['action'] == 'SELL']

        # 4小时和1小时信号权重更高
        timeframe_weights = {'1h': 1.0, '4h': 1.2, '1d': 0.8}

        total_buy_score = sum(s['confidence'] * timeframe_weights[s['timeframe']] for s in buy_signals)
        total_sell_score = sum(s['confidence'] * timeframe_weights[s['timeframe']] for s in sell_signals)

        current_price = signals[0]['price']

        # 决策逻辑
        if total_buy_score >= 2.5 and len(buy_signals) >= 2:
            # 买入逻辑
            position_size = self.calculate_position_size(current_price)
            stop_loss = current_price * 0.98  # 2% 止损
            take_profit = current_price * 1.06  # 6% 止盈

            message = f"🚀 *买入信号* 🚀\n"
            message += f"币种: {symbol}\n"
            message += f"价格: ${current_price:.4f}\n"
            message += f"仓位: {position_size} USDT\n"
            message += f"止损: ${stop_loss:.4f}\n"
            message += f"止盈: ${take_profit:.4f}\n"
            message += f"信号强度: {total_buy_score:.2f}\n"
            message += "理由:\n" + "\n".join([f"- {r}" for s in buy_signals for r in s['reason']][:5])

            self.send_telegram_alert(message)
            # self.place_buy_order(symbol, position_size, stop_loss, take_profit)

        elif total_sell_score >= 2.0:
            # 卖出逻辑
            message = f"📉 *卖出信号* 📉\n"
            message += f"币种: {symbol}\n"
            message += f"价格: ${current_price:.4f}\n"
            message += f"信号强度: {total_sell_score:.2f}\n"
            message += "理由:\n" + "\n".join([f"- {r}" for s in sell_signals for r in s['reason']][:5])

            self.send_telegram_alert(message)
            # self.place_sell_order(symbol)

    def calculate_position_size(self, price, risk_per_trade=0.02, portfolio_size=1000):
        """计算仓位大小"""
        return portfolio_size * risk_per_trade

    def run_strategy(self, symbols=['BTC/USDT', 'ETH/USDT']):
        """运行策略主循环"""
        while True:
            try:
                for symbol in symbols:
                    signals = self.generate_signal(symbol)
                    self.execute_trading_decision(symbol, signals)

                # 每小时运行一次
                time.sleep(3600)

            except Exception as e:
                error_msg = f"❌ 策略执行错误: {str(e)}"
                self.send_telegram_alert(error_msg)
                time.sleep(300)  # 5分钟后重试
