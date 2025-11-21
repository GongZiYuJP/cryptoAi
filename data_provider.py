import logging
import time
from datetime import datetime
from functools import wraps
import requests
import pandas as pd
from config_loader import CONFIG

def retry_on_exception(func):
    """装饰器，用于在函数抛出异常时自动重试。"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tries = CONFIG['api_retries']['max_retries']
        delay = CONFIG['api_retries']['retry_delay']
        for i in range(tries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if i < tries - 1:
                    logging.warning(f"函数 {func.__name__} 失败: {e}. {delay}s后重试({i+1}/{tries})...")
                    time.sleep(delay)
                else:
                    logging.error(f"函数 {func.__name__} 在 {tries} 次重试后彻底失败: {e}")
                    return None
    return wrapper

@retry_on_exception
def get_fear_and_greed_index():
    """获取加密市场的恐惧与贪婪指数。"""
    response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
    response.raise_for_status()
    data = response.json()['data'][0]
    return {'value': data['value'], 'classification': data['value_classification']}

@retry_on_exception
def get_ohlcv_with_indicators(exchange_client, symbol):
    """获取K线数据并计算所有需要的技术指标。"""
    timeframe = CONFIG['timeframe']
    periods = CONFIG['analysis_periods']
    limit = periods['long_term'] + 50
    ohlcv = exchange_client.fetch_ohlcv(symbol, timeframe, limit=limit)

    if not ohlcv or len(ohlcv) < periods['long_term']:
        logging.warning(f"[{symbol}] K线数据不足，无法计算指标。")
        return None

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # --- 指标计算 ---
    # --- 1. 趋势指标 (SMA, EMA, MACD) ---
    df['sma_short'] = df['close'].rolling(window=periods['short_term']).mean()
    df['sma_medium'] = df['close'].rolling(window=periods['medium_term']).mean()
    df['ema_long'] = df['close'].ewm(span=periods['long_term'], adjust=False).mean()

    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # --- 2. 动量指标 (RSI) ---
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1)  # 避免除零
    df['rsi'] = 100 - (100 / (1 + rs))

    # --- 3. 波动率指标 (Bollinger Bands, ATR) ---
    bb_window = periods['bollinger_window']
    df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
    bb_std = df['close'].rolling(window=bb_window).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

    # 计算价格在布林带中的相对位置 (非常有用)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 1)

    df['tr'] = pd.DataFrame({'h-l': df['high'] - df['low'],
                             'h-pc': (df['high'] - df['close'].shift()).abs(),
                             'l-pc': (df['low'] - df['close'].shift()).abs()}).max(axis=1)
    df['atr'] = df['tr'].ewm(span=CONFIG['atr_period'], adjust=False).mean()
    
    # --- 4. 成交量指标 ---
    df['volume_ma'] = df['volume'].rolling(window=periods['short_term']).mean()
    # 计算成交量比率，大于1表示放量
    df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)

    # --- 5. 趋势强度 (ADX) ---
    adx_period = CONFIG['adx_period']

    # 计算方向移动
    df['move_up'] = df['high'].diff()
    df['move_down'] = -df['low'].diff()
    df['plus_dm'] = ((df['move_up'] > df['move_down']) & (df['move_up'] > 0)) * df['move_up']
    df['minus_dm'] = ((df['move_down'] > df['move_up']) & (df['move_down'] > 0)) * df['move_down']

    # 平滑TR和方向移动
    df['tr_smooth'] = df['tr'].ewm(span=adx_period, adjust=False).mean()
    df['plus_di'] = 100 * (df['plus_dm'].ewm(span=adx_period, adjust=False).mean() / df['tr_smooth'])
    df['minus_di'] = 100 * (df['minus_dm'].ewm(span=adx_period, adjust=False).mean() / df['tr_smooth'])

    # 计算DX
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, 1)

    # 使用EMA
    df['adx'] = df['dx'].ewm(span=adx_period, adjust=False).mean()

    # --- 6. 支撑与阻力位 ---
    lookback = periods['short_term']
    df['support'] = df['low'].rolling(window=lookback).min()
    df['resistance'] = df['high'].rolling(window=lookback).max()

    # 信号生成基于最后一根【已收盘】的K线 (df.iloc[-2])
    signal_candle = df.iloc[-2]
    # 当前实时价格来自【未收盘】的K线 (df.iloc[-1])
    current_candle = df.iloc[-1]
    # 在返回前检查关键指标的有效性
    if pd.isna(signal_candle['sma_short']) or pd.isna(signal_candle['ema_long']):
        logging.warning("关键指标包含NaN值，数据可能不足")
        return None

    return {
        'price': current_candle['close'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'indicators': {
            'sma_short': signal_candle['sma_short'],
            'sma_medium': signal_candle['sma_medium'],
            'ema_long': signal_candle['ema_long'],
            'rsi': signal_candle['rsi'],
            'macd': signal_candle['macd'],
            'macd_signal': signal_candle['macd_signal'],
            'adx': signal_candle['adx'],
            'atr': signal_candle['atr'],
            'bb_upper': signal_candle['bb_upper'],
            'bb_lower': signal_candle['bb_lower'],
            'bb_position': signal_candle['bb_position'],
            'volume_ratio': signal_candle['volume_ratio'],
            'support': signal_candle['support'],
            'resistance': signal_candle['resistance']
        },
        'kline_data': df.tail(5).to_dict('records')
    }

@retry_on_exception
def calculate_atr_risk_parameters(signal_data, price_data, price_precision, config):
    """
    完全基于代码逻辑，根据ATR和动态波动率规则计算止损和止盈。
    此函数不再依赖AI提供的任何价格水平。
    """
    signal = signal_data.get('signal')
    if not signal or signal not in ['BUY', 'SELL']:
        signal_data['stop_loss'] = 0.0
        signal_data['take_profit'] = 0.0
        return signal_data

    current_price = price_data['price']
    atr = price_data['indicators']['atr']

    # 【动态倍数规则】
    volatility_ratio = atr / current_price
    if volatility_ratio > config.get('volatility_high_threshold', 0.005):  # 高波动 (> 0.5%)
        # 高波动市场需要更宽的止损空间
        atr_multiplier = config.get('atr_multiplier_high', 2.8) 
    elif config.get('volatility_low_threshold', 0.002) <= volatility_ratio <= config.get('volatility_high_threshold', 0.005):  # 正常波动
        atr_multiplier = config.get('atr_multiplier_normal', 2.2)
    else:  # 低波动 (< 0.2%)
        # 低波动市场可以使用较紧的止损
        atr_multiplier = config.get('atr_multiplier_low', 1.8)

    # 【风控计算 - 纯ATR版】
    # 1. 根据ATR计算止损距离，并在原基础上再额外添加2%缓冲，减少被插针扫损
    atr_distance = atr * atr_multiplier
    extra_buffer_pct = config.get('extra_stop_loss_pct', 0.02)  # 额外2%
    extra_buffer = current_price * extra_buffer_pct
    stop_loss_distance = atr_distance + extra_buffer

    if signal == 'BUY':
        final_stop_loss = current_price - stop_loss_distance
    else:  # SELL
        final_stop_loss = current_price + stop_loss_distance

    # 2. 根据止损距离计算止盈位
    take_profit_ratio = config.get('take_profit_ratio', 1.5)
    
    if signal == 'BUY':
        take_profit = current_price + (stop_loss_distance * take_profit_ratio)
    else: # SELL
        take_profit = current_price - (stop_loss_distance * take_profit_ratio)
        
    # 将最终计算结果更新回signal_data
    signal_data['stop_loss'] = round(final_stop_loss, price_precision)
    signal_data['take_profit'] = round(take_profit, price_precision)
    
    logging.info(f"纯ATR风控计算完成: Signal={signal}, SL={signal_data['stop_loss']}, TP={signal_data['take_profit']}")
    
    return signal_data