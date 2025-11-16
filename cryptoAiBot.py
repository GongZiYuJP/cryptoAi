import ccxt
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import time
from datetime import datetime, timedelta
import schedule
import sys
import ta  # 技术指标库
import json
import os

# 深度学习库（可选，如果未安装会自动降级）
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow未安装，深度学习功能将使用简化版本")
    print("   安装命令: pip install tensorflow")

# 配置（替换为您的key）
TESTNET_API_KEY = "ylc7VTuA7zSuWLhEezYYYec6mMZWbH06t7RLTriuvb4ufj4VDZJWEiaRsl7xY0qM"
TESTNET_API_SECRET = "7WjmJJCAp0rY9jq1sD7pnobAFJY087nSVr7BbtsS9x2JX2JLO1JPXbx7SeKIrpaj"
# Telegram 配置
TELEGRAM_BOT_TOKEN = "8534033934:AAEZ1AY6K3llNT3viVoYkdRGJSUik_xSrUQ"
TELEGRAM_CHAT_ID = "1450400854"
# 监控币种
COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
# ETH专用配置（合约交易）
ETH_SYMBOL = 'ETH/USDT:USDT'  # 永续合约格式
TIMEFRAME = '1h'  # 主时间周期
SMALL_TIMEFRAMES = ['5m', '15m']  # 小级别K线用于精确入场
LEVERAGE = {'LONG': 3, 'SHORT': 3}  # 杠杆倍数
RISK_PER_TRADE = 0.01  # 1%风险
STOP_LOSS_PCT = 0.02  # 2%止损
TAKE_PROFIT_PCT = 0.06  # 6%止盈，盈亏比3:1
SIGNAL_THRESHOLD = 70  # 信号强度阈值（0-100）
MODEL_PATH = "ai_model.pkl"
LOG_FILE = "trading_log.txt"
# 实时监控间隔（秒）
MONITOR_INTERVAL = 300  # 5分钟检查一次
# FVG配置
FVG_MIN_SIZE = 0.001  # FVG最小大小（0.1%）
# 自动交易开关
AUTO_TRADE_ENABLED = False  # 设置为True启用自动交易，False仅监控（当前：仅监控模式，不执行实际交易）
# 最大持仓数量
MAX_POSITIONS = 1  # 最多同时持有1个仓位
# 交易记录文件
TRADE_RECORD_FILE = "trade_records.json"
# 信号历史记录文件（用于深度学习）
SIGNAL_HISTORY_FILE = "signal_history.json"
# 深度学习模型路径
DL_MODEL_PATH = "dl_lstm_model.h5"
DL_SCALER_PATH = "dl_scaler.pkl"
# 深度学习配置
DL_SEQUENCE_LENGTH = 60  # 使用60根K线作为输入序列
DL_PREDICTION_HORIZON = 24  # 预测未来24根K线（24小时）
DL_TRAIN_INTERVAL = 100  # 每100个新信号后重新训练模型
DL_MIN_SIGNALS_FOR_TRAIN = 50  # 至少需要50个信号才开始训练

# Binance 交易所配置（币安测试网 - 合约交易）
try:
    # 配置币安测试网（永续合约交易）
    exchange = ccxt.binance({
        'apiKey': TESTNET_API_KEY,
        'secret': TESTNET_API_SECRET,
        'sandbox': True,  # 启用测试网模式
        'options': {
            'defaultType': 'future',  # 使用永续合约市场
            'defaultMarginMode': 'isolated',  # 逐仓模式（isolated）或全仓模式（cross）
        },
        'enableRateLimit': True,
        'timeout': 30000,
    })
    print("✅ 币安测试网（合约）连接成功")
except Exception as e:
    print(f"❌ 币安测试网连接失败: {e}")
    # 如果配置有问题，使用公共API（仅读取数据，无法交易）
    exchange = ccxt.binance({
        'options': {'defaultType': 'future'},
        'enableRateLimit': True,
        'timeout': 30000,
    })
    print("⚠️ 使用公共API模式（仅读取数据，无法交易）")
    print("⚠️ 自动交易已禁用，当前为仅监控模式")

# 获取链上数据分数
def get_onchain_score(coin):
    # 暂时注释掉 Glassnode API 调用，等待配置 GLASSNODE_KEY
    # score = 0
    # try:
    #     # 交易所净流入 (示例：净流出加分)
    #     inflow_data = requests.get(
    #         f"https://api.glassnode.com/v1/metrics/exchanges/netflow_total",
    #         params={'a': coin.lower(), 'api_key': GLASSNODE_KEY}
    #     ).json()
    #     if inflow_data:
    #         inflow = inflow_data[-1]['v']
    #         if inflow > 500000000: score -= 20  # 大流入（抛售）
    #         if inflow < -500000000: score += 20  # 大流出（囤积）
    #
    #     # MVRV Z-Score
    #     mvrv_data = requests.get(
    #         f"https://api.glassnode.com/v1/metrics/market/mvrv_z_score",
    #         params={'a': coin.lower(), 'api_key': GLASSNODE_KEY}
    #     ).json()
    #     if mvrv_data:
    #         mvrv = mvrv_data[-1]['v']
    #         if mvrv > 7: score -= 10
    #         if mvrv < -1: score += 10
    #
    #     # 其他链上：NVT, 活跃地址等（类似添加）
    # except Exception as e:
    #     log(f"链上数据错误: {e}")
    # return score
    return 0  # 暂时返回 0，不影响其他逻辑

# 加载或训练AI模型（GradientBoosting，持续学习）
def load_or_train_model(df_features, labels):
    try:
        model = joblib.load(MODEL_PATH)
    except:
        # 初始训练
        scaler = StandardScaler()
        X = scaler.fit_transform(df_features)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
        model = GradientBoostingClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)

    # 持续学习：partial_fit (XGBoost支持增量)
    if len(df_features) > 0:
        model.partial_fit(df_features, labels)  # 假设有新数据
        joblib.dump(model, MODEL_PATH)
    return model

# ==================== 深度学习功能 ====================

# 记录信号历史
def record_signal_history(signal):
    """记录交易信号，用于后续学习和评估"""
    try:
        if not signal:
            return
        
        # 读取现有记录
        if os.path.exists(SIGNAL_HISTORY_FILE):
            with open(SIGNAL_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        
        # 获取当前K线数据作为特征
        df = get_historical_data(ETH_SYMBOL, timeframe=TIMEFRAME, limit=100)
        if df.empty:
            return
        
        current = df.iloc[-1]
        
        # 记录信号
        signal_record = {
            'timestamp': datetime.now().isoformat(),
            'signal_id': len(history) + 1,
            'direction': signal.get('direction'),
            'signal_strength': signal.get('signal_strength', 0),
            'entry_price': signal.get('entry_price', 0),
            'current_price': signal.get('current_price', 0),
            'stop_loss': signal.get('stop_loss', 0),
            'take_profit': signal.get('take_profit', 0),
            'risk_reward_ratio': signal.get('risk_reward_ratio', 0),
            # 技术指标特征
            'rsi': float(current.get('rsi', 0)),
            'macd': float(current.get('macd', 0)),
            'macd_signal': float(current.get('macd_signal', 0)),
            'macd_hist': float(current.get('macd_hist', 0)),
            'ema7': float(current.get('ema7', 0)),
            'ema14': float(current.get('ema14', 0)),
            'ema21': float(current.get('ema21', 0)),
            'ema50': float(current.get('ema50', 0)),
            'vol_ratio': float(current.get('vol_ratio', 0)),
            'atr_pct': float(current.get('atr_pct', 0)),
            # 后续价格走势（待填充）
            'future_prices': [],
            'actual_result': None,  # 'WIN', 'LOSS', 'PENDING'
            'max_profit_pct': 0,
            'max_loss_pct': 0,
            'final_pnl_pct': 0,
            'evaluated': False
        }
        
        history.append(signal_record)
        
        # 保存记录
        with open(SIGNAL_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"记录信号历史错误: {e}")

# 更新信号历史（评估之前的信号质量）
def evaluate_signal_history():
    """评估历史信号的质量，更新实际结果"""
    try:
        if not os.path.exists(SIGNAL_HISTORY_FILE):
            return
        
        with open(SIGNAL_HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        if not history:
            return
        
        # 获取最新价格数据
        df = get_historical_data(ETH_SYMBOL, timeframe=TIMEFRAME, limit=DL_PREDICTION_HORIZON + 10)
        if df.empty:
            return
        
        current_price = df.iloc[-1]['close']
        current_time = datetime.now()
        
        updated = False
        for record in history:
            if record.get('evaluated', False):
                continue
            
            signal_time = datetime.fromisoformat(record['timestamp'])
            hours_passed = (current_time - signal_time).total_seconds() / 3600
            
            # 如果信号产生超过24小时，进行评估
            if hours_passed >= DL_PREDICTION_HORIZON:
                direction = record.get('direction')
                entry_price = record.get('entry_price', 0)
                stop_loss = record.get('stop_loss', 0)
                take_profit = record.get('take_profit', 0)
                
                if entry_price == 0:
                    continue
                
                # 获取信号产生后的价格走势
                signal_idx = None
                for i, row in df.iterrows():
                    if abs((pd.to_datetime(row['timestamp']) - signal_time).total_seconds()) < 3600:
                        signal_idx = i
                        break
                
                if signal_idx is None:
                    continue
                
                # 分析后续价格走势
                future_prices = []
                max_profit = 0
                max_loss = 0
                hit_stop_loss = False
                hit_take_profit = False
                
                for i in range(signal_idx, min(signal_idx + DL_PREDICTION_HORIZON, len(df))):
                    price = df.iloc[i]['close']
                    future_prices.append(float(price))
                    
                    if direction == 'LONG':
                        profit_pct = ((price - entry_price) / entry_price) * 100
                        if price <= stop_loss:
                            hit_stop_loss = True
                        if price >= take_profit:
                            hit_take_profit = True
                    else:  # SHORT
                        profit_pct = ((entry_price - price) / entry_price) * 100
                        if price >= stop_loss:
                            hit_stop_loss = True
                        if price <= take_profit:
                            hit_take_profit = True
                    
                    max_profit = max(max_profit, profit_pct)
                    max_loss = min(max_loss, profit_pct)
                
                # 最终结果
                final_price = future_prices[-1] if future_prices else current_price
                if direction == 'LONG':
                    final_pnl = ((final_price - entry_price) / entry_price) * 100
                else:
                    final_pnl = ((entry_price - final_price) / entry_price) * 100
                
                # 判断结果
                if hit_stop_loss:
                    actual_result = 'LOSS'
                elif hit_take_profit:
                    actual_result = 'WIN'
                elif final_pnl > 0:
                    actual_result = 'WIN'
                else:
                    actual_result = 'LOSS'
                
                # 更新记录
                record['future_prices'] = future_prices
                record['actual_result'] = actual_result
                record['max_profit_pct'] = max_profit
                record['max_loss_pct'] = max_loss
                record['final_pnl_pct'] = final_pnl
                record['evaluated'] = True
                updated = True
        
        if updated:
            with open(SIGNAL_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            print(f"✅ 已评估 {sum(1 for r in history if r.get('evaluated'))} 个历史信号")
            
    except Exception as e:
        print(f"评估信号历史错误: {e}")
        import traceback
        traceback.print_exc()

# 构建LSTM深度学习模型
def build_lstm_model(input_shape):
    """构建LSTM深度学习模型"""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    try:
        model = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True, dropout=0.2),
            LSTM(64, return_sequences=False, dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')  # 3个输出：LONG, SHORT, NEUTRAL的概率
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        print(f"构建LSTM模型错误: {e}")
        return None

# 准备训练数据
def prepare_training_data():
    """从信号历史中准备训练数据"""
    try:
        if not os.path.exists(SIGNAL_HISTORY_FILE):
            return None, None
        
        with open(SIGNAL_HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # 只使用已评估的信号
        evaluated_signals = [r for r in history if r.get('evaluated', False)]
        
        if len(evaluated_signals) < DL_MIN_SIGNALS_FOR_TRAIN:
            return None, None
        
        # 获取历史K线数据
        df = get_historical_data(ETH_SYMBOL, timeframe=TIMEFRAME, limit=500)
        if df.empty:
            return None, None
        
        X_sequences = []
        y_labels = []
        
        for signal in evaluated_signals:
            try:
                signal_time = datetime.fromisoformat(signal['timestamp'])
                
                # 找到信号产生时的K线索引
                signal_idx = None
                for i, row in df.iterrows():
                    if abs((pd.to_datetime(row['timestamp']) - signal_time).total_seconds()) < 3600:
                        signal_idx = i
                        break
                
                if signal_idx is None or signal_idx < DL_SEQUENCE_LENGTH:
                    continue
                
                # 提取序列特征（使用信号产生前60根K线）
                sequence = []
                for j in range(signal_idx - DL_SEQUENCE_LENGTH, signal_idx):
                    row = df.iloc[j]
                    features = [
                        float(row.get('close', 0)),
                        float(row.get('rsi', 0)),
                        float(row.get('macd', 0)),
                        float(row.get('macd_hist', 0)),
                        float(row.get('ema7', 0)),
                        float(row.get('ema14', 0)),
                        float(row.get('ema21', 0)),
                        float(row.get('ema50', 0)),
                        float(row.get('vol_ratio', 0)),
                        float(row.get('atr_pct', 0)),
                    ]
                    sequence.append(features)
                
                if len(sequence) == DL_SEQUENCE_LENGTH:
                    X_sequences.append(sequence)
                    
                    # 标签：根据实际结果
                    result = signal.get('actual_result', 'LOSS')
                    direction = signal.get('direction', 'NEUTRAL')
                    
                    # 如果信号正确，使用原方向；如果错误，使用相反方向或NEUTRAL
                    if result == 'WIN':
                        if direction == 'LONG':
                            y_labels.append([1, 0, 0])  # LONG
                        else:
                            y_labels.append([0, 1, 0])  # SHORT
                    else:  # LOSS
                        # 错误信号，标记为NEUTRAL或相反方向
                        y_labels.append([0, 0, 1])  # NEUTRAL
                        
            except Exception as e:
                continue
        
        if len(X_sequences) < 10:
            return None, None
        
        X = np.array(X_sequences)
        y = np.array(y_labels)
        
        return X, y
        
    except Exception as e:
        print(f"准备训练数据错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# 训练深度学习模型
def train_deep_learning_model():
    """训练LSTM深度学习模型"""
    if not TENSORFLOW_AVAILABLE:
        print("⚠️ TensorFlow未安装，跳过深度学习模型训练")
        return None
    
    try:
        print("🔄 开始训练深度学习模型...")
        
        # 准备数据
        X, y = prepare_training_data()
        if X is None or y is None:
            print("⚠️ 训练数据不足，跳过训练")
            return None
        
        print(f"📊 训练数据: {len(X)} 个样本")
        
        # 数据标准化
        scaler = MinMaxScaler()
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # 保存scaler
        joblib.dump(scaler, DL_SCALER_PATH)
        
        # 构建模型
        input_shape = (n_timesteps, n_features)
        model = build_lstm_model(input_shape)
        
        if model is None:
            return None
        
        # 训练模型
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(DL_MODEL_PATH, save_best_only=True, monitor='loss')
        
        history = model.fit(
            X_scaled, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1,
            callbacks=[early_stopping, model_checkpoint]
        )
        
        print(f"✅ 深度学习模型训练完成，准确率: {max(history.history['accuracy']):.2%}")
        return model
        
    except Exception as e:
        print(f"训练深度学习模型错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# 使用深度学习模型预测
def predict_with_dl_model(df):
    """使用深度学习模型预测交易方向"""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    try:
        # 加载模型
        if not os.path.exists(DL_MODEL_PATH):
            return None
        
        model = load_model(DL_MODEL_PATH)
        
        # 加载scaler
        if not os.path.exists(DL_SCALER_PATH):
            return None
        
        scaler = joblib.load(DL_SCALER_PATH)
        
        # 准备输入数据（最近60根K线）
        if len(df) < DL_SEQUENCE_LENGTH:
            return None
        
        sequence = []
        for i in range(len(df) - DL_SEQUENCE_LENGTH, len(df)):
            row = df.iloc[i]
            features = [
                float(row.get('close', 0)),
                float(row.get('rsi', 0)),
                float(row.get('macd', 0)),
                float(row.get('macd_hist', 0)),
                float(row.get('ema7', 0)),
                float(row.get('ema14', 0)),
                float(row.get('ema21', 0)),
                float(row.get('ema50', 0)),
                float(row.get('vol_ratio', 0)),
                float(row.get('atr_pct', 0)),
            ]
            sequence.append(features)
        
        X = np.array([sequence])
        
        # 标准化
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # 预测
        predictions = model.predict(X_scaled, verbose=0)
        probs = predictions[0]  # [LONG概率, SHORT概率, NEUTRAL概率]
        
        return {
            'long_prob': float(probs[0]),
            'short_prob': float(probs[1]),
            'neutral_prob': float(probs[2]),
            'predicted_direction': 'LONG' if probs[0] > probs[1] and probs[0] > 0.5 else 
                                  'SHORT' if probs[1] > probs[0] and probs[1] > 0.5 else 'NEUTRAL',
            'confidence': float(max(probs))
        }
        
    except Exception as e:
        print(f"深度学习预测错误: {e}")
        return None

# 自我修正交易算法
def self_correct_trading_algorithm():
    """根据历史信号表现，自我修正交易算法参数"""
    try:
        if not os.path.exists(SIGNAL_HISTORY_FILE):
            return
        
        with open(SIGNAL_HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        evaluated_signals = [r for r in history if r.get('evaluated', False)]
        
        if len(evaluated_signals) < 20:
            return
        
        # 分析信号表现
        win_count = sum(1 for s in evaluated_signals if s.get('actual_result') == 'WIN')
        loss_count = sum(1 for s in evaluated_signals if s.get('actual_result') == 'LOSS')
        total = len(evaluated_signals)
        win_rate = win_count / total if total > 0 else 0
        
        # 分析不同信号强度的表现
        strength_performance = {}
        for signal in evaluated_signals:
            strength = int(signal.get('signal_strength', 0) // 10) * 10  # 按10分区间分组
            if strength not in strength_performance:
                strength_performance[strength] = {'win': 0, 'loss': 0}
            
            if signal.get('actual_result') == 'WIN':
                strength_performance[strength]['win'] += 1
            else:
                strength_performance[strength]['loss'] += 1
        
        # 找出表现最好的信号强度区间
        best_threshold = SIGNAL_THRESHOLD
        best_win_rate = 0
        
        for strength, perf in strength_performance.items():
            total_strength = perf['win'] + perf['loss']
            if total_strength >= 5:  # 至少5个样本
                strength_win_rate = perf['win'] / total_strength
                if strength_win_rate > best_win_rate:
                    best_win_rate = strength_win_rate
                    best_threshold = strength
        
        # 如果发现更好的阈值，建议调整
        global SIGNAL_THRESHOLD
        if best_threshold != SIGNAL_THRESHOLD and best_win_rate > win_rate + 0.1:
            old_threshold = SIGNAL_THRESHOLD
            SIGNAL_THRESHOLD = max(70, min(90, best_threshold))  # 限制在70-90之间
            
            correction_msg = f"🧠 <b>算法自我修正</b>\n\n" \
                           f"历史信号分析:\n" \
                           f"总信号数: {total}\n" \
                           f"胜率: {win_rate:.1%}\n" \
                           f"最佳信号强度阈值: {best_threshold} (胜率: {best_win_rate:.1%})\n\n" \
                           f"建议调整:\n" \
                           f"信号阈值: {old_threshold} → {SIGNAL_THRESHOLD}"
            
            log(correction_msg, send_to_telegram=True)
            print(f"🧠 算法自我修正: 信号阈值 {old_threshold} → {SIGNAL_THRESHOLD}")
        
    except Exception as e:
        print(f"自我修正算法错误: {e}")

# 获取历史数据并计算所有技术指标
def get_historical_data(symbol, timeframe=None, limit=500):
    """获取K线数据并计算完整的技术指标"""
    try:
        if timeframe is None:
            timeframe = TIMEFRAME
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # MA (移动平均线) - 多个周期
        df['ma7'] = df['close'].rolling(7).mean()
        df['ma14'] = df['close'].rolling(14).mean()
        df['ma21'] = df['close'].rolling(21).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        df['ma100'] = df['close'].rolling(100).mean()
        df['ma200'] = df['close'].rolling(200).mean()
        
        # EMA (指数移动平均线) - 多个周期
        df['ema7'] = df['close'].ewm(span=7, adjust=False).mean()
        df['ema14'] = df['close'].ewm(span=14, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # RSI (相对强弱指标)
        rsi_indicator = ta.momentum.RSIIndicator(df['close'], window=14)
        df['rsi'] = rsi_indicator.rsi()
        
        # MACD (指数平滑异同移动平均线)
        macd_indicator = ta.trend.MACD(df['close'])
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()
        
        # VOL (成交量分析)
        df['vol_ma20'] = df['volume'].rolling(20).mean()  # 成交量20日均线
        df['vol_ratio'] = df['volume'] / df['vol_ma20']  # 成交量比率
        df['vol_change'] = df['volume'].pct_change()  # 成交量变化率
        
        # ATR (平均真实波幅) - 用于止损计算
        atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr_indicator.average_true_range()
        df['atr_pct'] = (df['atr'] / df['close']) * 100  # ATR百分比
        
        # 布林带
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']  # 布林带宽度
        
        # 价格变化率
        df['price_change'] = df['close'].pct_change()
        df['price_change_3'] = df['close'].pct_change(3)
        df['price_change_7'] = df['close'].pct_change(7)
        
        return df.dropna()
    except Exception as e:
        print(f"获取数据错误: {e}")
        return pd.DataFrame()

# 检测FVG（Fair Value Gap - 公平价值缺口）
def detect_fvg(df, min_size_pct=0.001):
    """
    检测FVG（公平价值缺口）
    FVG是价格快速移动时留下的不平衡区域，通常会被回填
    返回: [{type: 'bullish'/'bearish', top: float, bottom: float, strength: float}, ...]
    """
    fvgs = []
    if len(df) < 3:
        return fvgs
    
    for i in range(1, len(df) - 1):
        prev_candle = df.iloc[i-1]
        current_candle = df.iloc[i]
        next_candle = df.iloc[i+1]
        
        # 看涨FVG：前一根K线的高点 < 后一根K线的低点（中间K线形成向上缺口）
        # 三根K线：前一根、中间（缺口）、后一根
        if prev_candle['high'] < next_candle['low']:
            # 看涨FVG：价格跳空上涨
            fvg_bottom = prev_candle['high']  # FVG底部是前一根K线的高点
            fvg_top = next_candle['low']      # FVG顶部是后一根K线的低点
            fvg_size = (fvg_top - fvg_bottom) / fvg_bottom
            
            if fvg_size >= min_size_pct:
                strength = min(fvg_size * 1000, 100)  # 强度0-100
                fvgs.append({
                    'type': 'bullish',
                    'top': fvg_top,
                    'bottom': fvg_bottom,
                    'mid': (fvg_top + fvg_bottom) / 2,
                    'size_pct': fvg_size * 100,
                    'strength': strength,
                    'index': i,
                    'timestamp': current_candle['timestamp']
                })
        
        # 看跌FVG：前一根K线的低点 > 后一根K线的高点（中间K线形成向下缺口）
        elif prev_candle['low'] > next_candle['high']:
            # 看跌FVG：价格跳空下跌
            fvg_top = prev_candle['low']      # FVG顶部是前一根K线的低点
            fvg_bottom = next_candle['high']  # FVG底部是后一根K线的高点
            fvg_size = (fvg_top - fvg_bottom) / fvg_bottom
            
            if fvg_size >= min_size_pct:
                strength = min(fvg_size * 1000, 100)
                fvgs.append({
                    'type': 'bearish',
                    'top': fvg_top,
                    'bottom': fvg_bottom,
                    'mid': (fvg_top + fvg_bottom) / 2,
                    'size_pct': fvg_size * 100,
                    'strength': strength,
                    'index': i,
                    'timestamp': current_candle['timestamp']
                })
    
    return fvgs

# K线形态识别
def detect_candlestick_patterns(df):
    """
    识别K线形态
    返回: {pattern_name: bool, ...}
    """
    patterns = {}
    if len(df) < 3:
        return patterns
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) >= 3 else None
    
    body = abs(current['close'] - current['open'])
    upper_shadow = current['high'] - max(current['open'], current['close'])
    lower_shadow = min(current['open'], current['close']) - current['low']
    total_range = current['high'] - current['low']
    
    # 避免除零
    if total_range == 0:
        return patterns
    
    body_ratio = body / total_range
    upper_ratio = upper_shadow / total_range
    lower_ratio = lower_shadow / total_range
    
    # 1. 锤子线（Hammer）- 看涨反转
    patterns['hammer'] = (
        lower_ratio > 0.6 and 
        upper_ratio < 0.1 and 
        body_ratio < 0.3 and
        current['close'] > current['open']
    )
    
    # 2. 上吊线（Hanging Man）- 看跌反转
    patterns['hanging_man'] = (
        lower_ratio > 0.6 and 
        upper_ratio < 0.1 and 
        body_ratio < 0.3 and
        current['close'] < current['open']
    )
    
    # 3. 吞没形态（Engulfing）
    if prev2 is not None:
        # 看涨吞没
        patterns['bullish_engulfing'] = (
            prev['close'] < prev['open'] and  # 前一根是阴线
            current['close'] > current['open'] and  # 当前是阳线
            current['open'] < prev['close'] and  # 当前开盘低于前一根收盘
            current['close'] > prev['open']  # 当前收盘高于前一根开盘
        )
        
        # 看跌吞没
        patterns['bearish_engulfing'] = (
            prev['close'] > prev['open'] and  # 前一根是阳线
            current['close'] < current['open'] and  # 当前是阴线
            current['open'] > prev['close'] and  # 当前开盘高于前一根收盘
            current['close'] < prev['open']  # 当前收盘低于前一根开盘
        )
    
    # 4. 十字星（Doji）
    patterns['doji'] = body_ratio < 0.1
    
    # 5. 流星线（Shooting Star）- 看跌
    patterns['shooting_star'] = (
        upper_ratio > 0.6 and 
        lower_ratio < 0.1 and 
        body_ratio < 0.3 and
        current['close'] < current['open']
    )
    
    # 6. 三只乌鸦（Three Black Crows）- 看跌
    if len(df) >= 3:
        patterns['three_black_crows'] = (
            df.iloc[-3]['close'] < df.iloc[-3]['open'] and
            df.iloc[-2]['close'] < df.iloc[-2]['open'] and
            current['close'] < current['open'] and
            df.iloc[-2]['close'] < df.iloc[-3]['close'] and
            current['close'] < df.iloc[-2]['close']
        )
        
        # 三只白兵（Three White Soldiers）- 看涨
        patterns['three_white_soldiers'] = (
            df.iloc[-3]['close'] > df.iloc[-3]['open'] and
            df.iloc[-2]['close'] > df.iloc[-2]['open'] and
            current['close'] > current['open'] and
            df.iloc[-2]['close'] > df.iloc[-3]['close'] and
            current['close'] > df.iloc[-2]['close']
        )
    
    return patterns

# 均线策略判断
def analyze_ma_strategy(df):
    """
    使用均线法判断做多/做空
    返回: {direction: 'LONG'/'SHORT'/'NEUTRAL', score: float, details: str}
    """
    if len(df) < 200:
        return {'direction': 'NEUTRAL', 'score': 0, 'details': '数据不足'}
    
    current = df.iloc[-1]
    price = current['close']
    
    # 均线排列
    ema7 = current['ema7']
    ema14 = current['ema14']
    ema21 = current['ema21']
    ema50 = current['ema50']
    ema100 = current['ema100']
    ema200 = current['ema200']
    
    long_score = 0
    short_score = 0
    details = []
    
    # 1. 均线多头排列（短期>长期）
    if ema7 > ema14 > ema21 > ema50:
        long_score += 30
        details.append("✅ 均线多头排列")
    elif ema7 < ema14 < ema21 < ema50:
        short_score += 30
        details.append("❌ 均线空头排列")
    
    # 2. 价格与均线关系
    if price > ema7 > ema14 > ema21:
        long_score += 20
        details.append("✅ 价格在均线上方")
    elif price < ema7 < ema14 < ema21:
        short_score += 20
        details.append("❌ 价格在均线下方")
    
    # 3. EMA50作为关键支撑/阻力
    if price > ema50 and ema7 > ema50:
        long_score += 15
        details.append("✅ 价格在EMA50上方")
    elif price < ema50 and ema7 < ema50:
        short_score += 15
        details.append("❌ 价格在EMA50下方")
    
    # 4. EMA200长期趋势
    if price > ema200:
        long_score += 10
        details.append("✅ 价格在EMA200上方（长期看涨）")
    else:
        short_score += 10
        details.append("❌ 价格在EMA200下方（长期看跌）")
    
    # 5. 均线斜率（趋势强度）
    if len(df) >= 5:
        ema7_slope = (ema7 - df.iloc[-5]['ema7']) / df.iloc[-5]['ema7']
        if ema7_slope > 0.001:  # 上升趋势
            long_score += 10
            details.append(f"✅ EMA7上升趋势 ({ema7_slope*100:.2f}%)")
        elif ema7_slope < -0.001:  # 下降趋势
            short_score += 10
            details.append(f"❌ EMA7下降趋势 ({ema7_slope*100:.2f}%)")
    
    total_score = long_score - short_score
    
    if total_score > 20:
        direction = 'LONG'
    elif total_score < -20:
        direction = 'SHORT'
    else:
        direction = 'NEUTRAL'
    
    return {
        'direction': direction,
        'score': abs(total_score),
        'long_score': long_score,
        'short_score': short_score,
        'details': ' | '.join(details)
    }

# 综合分析ETH：均线+形态+FVG+小级别K线
def analyze_eth_advanced():
    """
    综合分析ETH走势：
    1. 使用均线法判断大方向（做多/做空）
    2. 识别K线形态（赛福形态等）
    3. 使用小级别K线（5m, 15m）和FVG找最佳入场点
    返回: 完整的交易信号
    """
    try:
        # 1. 主时间周期分析（1h）- 判断大方向
        df_1h = get_historical_data(ETH_SYMBOL, timeframe='1h', limit=300)
        if df_1h.empty or len(df_1h) < 200:
            return None
        
        current_price = df_1h.iloc[-1]['close']
        
        # 均线策略判断
        ma_analysis = analyze_ma_strategy(df_1h)
        main_direction = ma_analysis['direction']
        
        if main_direction == 'NEUTRAL':
            return None  # 方向不明确，不交易
        
        # K线形态识别
        patterns = detect_candlestick_patterns(df_1h)
        
        # 形态评分
        pattern_score = 0
        pattern_signals = []
        
        # 看涨形态
        if patterns.get('hammer') or patterns.get('bullish_engulfing') or patterns.get('three_white_soldiers'):
            if main_direction == 'LONG':
                pattern_score += 20
                if patterns.get('hammer'):
                    pattern_signals.append("🔨 锤子线（看涨反转）")
                if patterns.get('bullish_engulfing'):
                    pattern_signals.append("📈 看涨吞没")
                if patterns.get('three_white_soldiers'):
                    pattern_signals.append("⚪ 三只白兵（强烈看涨）")
        
        # 看跌形态
        if patterns.get('hanging_man') or patterns.get('bearish_engulfing') or patterns.get('three_black_crows'):
            if main_direction == 'SHORT':
                pattern_score += 20
                if patterns.get('hanging_man'):
                    pattern_signals.append("🔻 上吊线（看跌反转）")
                if patterns.get('bearish_engulfing'):
                    pattern_signals.append("📉 看跌吞没")
                if patterns.get('three_black_crows'):
                    pattern_signals.append("⚫ 三只乌鸦（强烈看跌）")
        
        # 2. 小级别K线分析 - 找精确入场点
        best_entry_points = []
        
        for small_tf in SMALL_TIMEFRAMES:
            try:
                df_small = get_historical_data(ETH_SYMBOL, timeframe=small_tf, limit=200)
                if df_small.empty or len(df_small) < 50:
                    continue
                
                # 检测FVG
                fvgs = detect_fvg(df_small, min_size_pct=FVG_MIN_SIZE)
                
                # 筛选有效的FVG（与主方向一致）
                valid_fvgs = []
                for fvg in fvgs:
                    # 只保留最近20根K线内的FVG
                    if len(df_small) - fvg['index'] <= 20:
                        if (main_direction == 'LONG' and fvg['type'] == 'bullish') or \
                           (main_direction == 'SHORT' and fvg['type'] == 'bearish'):
                            valid_fvgs.append(fvg)
                
                # 找到最佳入场点
                for fvg in valid_fvgs:
                    # 检查价格是否接近FVG
                    fvg_mid = fvg['mid']
                    price_distance = abs(current_price - fvg_mid) / current_price
                    
                    # 如果价格在FVG附近（1%以内），这是一个好的入场点
                    if price_distance < 0.01:
                        entry_price = fvg_mid
                        
                        # 计算止损止盈
                        if main_direction == 'LONG':
                            # 做多：止损在FVG底部下方，止盈在FVG顶部上方
                            stop_loss = fvg['bottom'] * 0.998  # FVG底部下方0.2%
                            take_profit = fvg['top'] * 1.002 + (fvg['top'] - fvg['bottom']) * 2  # FVG顶部上方+2倍FVG高度
                        else:
                            # 做空：止损在FVG顶部上方，止盈在FVG底部下方
                            stop_loss = fvg['top'] * 1.002  # FVG顶部上方0.2%
                            take_profit = fvg['bottom'] * 0.998 - (fvg['top'] - fvg['bottom']) * 2  # FVG底部下方-2倍FVG高度
                        
                        # 计算盈亏比
                        risk = abs(entry_price - stop_loss)
                        reward = abs(take_profit - entry_price)
                        risk_reward = reward / risk if risk > 0 else 0
                        
                        if risk_reward >= 2.0:  # 盈亏比至少2:1
                            best_entry_points.append({
                                'timeframe': small_tf,
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'risk_reward': risk_reward,
                                'fvg': fvg,
                                'price_distance_pct': price_distance * 100
                            })
                
            except Exception as e:
                print(f"分析{small_tf}时间周期错误: {e}")
                continue
        
        # 如果没有找到FVG入场点，使用ATR计算止损止盈
        if not best_entry_points:
            atr = df_1h.iloc[-1]['atr']
            if main_direction == 'LONG':
                entry_price = current_price
                stop_loss = current_price - atr * 2
                take_profit = current_price + atr * 3
            else:
                entry_price = current_price
                stop_loss = current_price + atr * 2
                take_profit = current_price - atr * 3
            
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            best_entry_points.append({
                'timeframe': '1h',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': risk_reward,
                'fvg': None,
                'price_distance_pct': 0
            })
        
        # 选择最佳入场点（风险回报比最高的）
        best_entry = max(best_entry_points, key=lambda x: x['risk_reward']) if best_entry_points else None
        
        if not best_entry:
            return None
        
        # 3. 深度学习模型预测（如果可用）
        dl_prediction = None
        dl_adjustment = 0
        if TENSORFLOW_AVAILABLE:
            dl_prediction = predict_with_dl_model(df_1h)
            if dl_prediction:
                # 如果深度学习预测与主方向一致，增加信号强度
                if dl_prediction['predicted_direction'] == main_direction:
                    dl_adjustment = dl_prediction['confidence'] * 20  # 最高增加20分
                    pattern_signals.append(f"🤖 深度学习确认: {dl_prediction['predicted_direction']} (置信度: {dl_prediction['confidence']:.1%})")
                elif dl_prediction['predicted_direction'] == 'NEUTRAL':
                    dl_adjustment = -10  # 深度学习建议观望，降低信号强度
                    pattern_signals.append(f"🤖 深度学习建议: 观望 (置信度: {dl_prediction['neutral_prob']:.1%})")
                else:
                    # 深度学习预测相反方向，大幅降低信号强度
                    dl_adjustment = -30
                    pattern_signals.append(f"⚠️ 深度学习警告: 预测方向相反 ({dl_prediction['predicted_direction']})")
        
        # 计算综合信号强度（结合深度学习）
        signal_strength = ma_analysis['score'] + pattern_score + dl_adjustment
        signal_strength = max(0, min(signal_strength, 100))  # 限制在0-100之间
        
        if signal_strength < SIGNAL_THRESHOLD:
            return None
        
        # 构建完整信号
        signal = {
            'direction': main_direction,
            'signal_strength': signal_strength,
            'current_price': current_price,
            'entry_price': best_entry['entry_price'],
            'stop_loss': best_entry['stop_loss'],
            'take_profit': best_entry['take_profit'],
            'risk_reward_ratio': best_entry['risk_reward'],
            'ma_analysis': ma_analysis,
            'patterns': pattern_signals,
            'best_entry': best_entry,
            'all_fvg_entries': best_entry_points,
            'rsi': df_1h.iloc[-1]['rsi'],
            'macd_hist': df_1h.iloc[-1]['macd_hist'],
            'dl_prediction': dl_prediction,  # 添加深度学习预测结果
        }
        
        return signal
        
    except Exception as e:
        print(f"综合分析ETH错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# 分析ETH走势并生成交易信号（保留原函数作为备用）
def analyze_eth_signal():
    """专门分析ETH的实时走势，返回详细的交易信号"""
    try:
        df = get_historical_data(ETH_SYMBOL, limit=300)
        if df.empty or len(df) < 200:
            return None
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 初始化信号分数
        long_score = 0
        short_score = 0
        signal_details = {
            'current_price': current['close'],
            'rsi': current['rsi'],
            'macd': current['macd'],
            'macd_signal': current['macd_signal'],
            'macd_hist': current['macd_hist'],
            'vol_ratio': current['vol_ratio'],
            'atr_pct': current['atr_pct'],
        }
        
        # 1. RSI 分析 (权重: 20分)
        rsi = current['rsi']
        if rsi < 30:  # 超卖区域
            long_score += 20
            signal_details['rsi_signal'] = '超卖，看多'
        elif rsi < 40:
            long_score += 10
            signal_details['rsi_signal'] = '偏弱，轻微看多'
        elif rsi > 70:  # 超买区域
            short_score += 20
            signal_details['rsi_signal'] = '超买，看空'
        elif rsi > 60:
            short_score += 10
            signal_details['rsi_signal'] = '偏强，轻微看空'
        else:
            signal_details['rsi_signal'] = '中性'
        
        # 2. MA/EMA 均线分析 (权重: 25分)
        # 多头排列：短期均线在长期均线之上
        ma_bullish = (current['ema7'] > current['ema14'] > current['ema21'] > current['ema50'])
        ma_bearish = (current['ema7'] < current['ema14'] < current['ema21'] < current['ema50'])
        
        if ma_bullish and current['close'] > current['ema7']:
            long_score += 25
            signal_details['ma_signal'] = '多头排列，价格在均线上方'
        elif ma_bearish and current['close'] < current['ema7']:
            short_score += 25
            signal_details['ma_signal'] = '空头排列，价格在均线下方'
        else:
            if current['close'] > current['ema50']:
                long_score += 10
                signal_details['ma_signal'] = '价格在EMA50上方'
            else:
                short_score += 10
                signal_details['ma_signal'] = '价格在EMA50下方'
        
        # 3. MACD 分析 (权重: 25分)
        macd_hist = current['macd_hist']
        macd_cross_up = (current['macd'] > current['macd_signal'] and 
                        prev['macd'] <= prev['macd_signal'])
        macd_cross_down = (current['macd'] < current['macd_signal'] and 
                          prev['macd'] >= prev['macd_signal'])
        
        if macd_cross_up and macd_hist > 0:
            long_score += 25
            signal_details['macd_signal'] = 'MACD金叉，看多'
        elif macd_cross_down and macd_hist < 0:
            short_score += 25
            signal_details['macd_signal'] = 'MACD死叉，看空'
        elif macd_hist > 0 and current['macd'] > 0:
            long_score += 15
            signal_details['macd_signal'] = 'MACD在零轴上方，偏多'
        elif macd_hist < 0 and current['macd'] < 0:
            short_score += 15
            signal_details['macd_signal'] = 'MACD在零轴下方，偏空'
        else:
            signal_details['macd_signal'] = 'MACD中性'
        
        # 4. 成交量分析 (权重: 15分)
        vol_ratio = current['vol_ratio']
        price_change = current['price_change']
        
        # 放量上涨看多，放量下跌看空
        if vol_ratio > 1.5 and price_change > 0:
            long_score += 15
            signal_details['vol_signal'] = f'放量上涨 (成交量比率: {vol_ratio:.2f})'
        elif vol_ratio > 1.5 and price_change < 0:
            short_score += 15
            signal_details['vol_signal'] = f'放量下跌 (成交量比率: {vol_ratio:.2f})'
        elif vol_ratio < 0.7:
            signal_details['vol_signal'] = f'缩量 (成交量比率: {vol_ratio:.2f})'
        else:
            signal_details['vol_signal'] = f'正常成交量 (比率: {vol_ratio:.2f})'
        
        # 5. 布林带分析 (权重: 15分)
        bb_position = (current['close'] - current['bb_lower']) / (current['bb_upper'] - current['bb_lower'])
        if bb_position < 0.2:  # 接近下轨，可能反弹
            long_score += 15
            signal_details['bb_signal'] = '价格接近布林带下轨，可能反弹'
        elif bb_position > 0.8:  # 接近上轨，可能回调
            short_score += 15
            signal_details['bb_signal'] = '价格接近布林带上轨，可能回调'
        else:
            signal_details['bb_signal'] = '价格在布林带中轨附近'
        
        # 计算最终信号
        total_score = long_score - short_score
        signal_strength = abs(total_score)
        direction = 'LONG' if total_score > 0 else 'SHORT'
        
        signal_details['long_score'] = long_score
        signal_details['short_score'] = short_score
        signal_details['total_score'] = total_score
        signal_details['signal_strength'] = signal_strength
        signal_details['direction'] = direction
        
        return signal_details if signal_strength >= SIGNAL_THRESHOLD else None
        
    except Exception as e:
        print(f"分析ETH信号错误: {e}")
        return None

# 生成交易信号通知（支持高级分析）
def send_trading_signal(signal_details):
    """发送详细的交易信号到Telegram（支持FVG和形态分析）"""
    try:
        # 检查是否是高级分析信号
        is_advanced = 'best_entry' in signal_details
        
        if is_advanced:
            # 高级分析信号（包含FVG和形态）
            direction = signal_details['direction']
            signal_strength = signal_details['signal_strength']
            entry_price = signal_details['entry_price']
            stop_loss = signal_details['stop_loss']
            take_profit = signal_details['take_profit']
            risk_reward_ratio = signal_details['risk_reward_ratio']
            current_price = signal_details['current_price']
            best_entry = signal_details['best_entry']
            ma_analysis = signal_details['ma_analysis']
            patterns = signal_details.get('patterns', [])
            
            direction_emoji = "📈" if direction == 'LONG' else "📉"
            direction_text = "做多 (LONG)" if direction == 'LONG' else "做空 (SHORT)"
            
            # 构建详细消息
            message = f"🎯 <b>ETH 高级交易信号</b> {direction_emoji}\n\n"
            message += f"━━━━━━━━━━━━━━━━━━━━\n"
            message += f"<b>交易方向:</b> {direction_text}\n"
            message += f"<b>信号强度:</b> {signal_strength:.1f}/100\n\n"
            
            message += f"<b>💰 价格信息</b>\n"
            message += f"当前价格: {current_price:.2f} USDT\n"
            message += f"<b>最佳入场: {entry_price:.2f} USDT</b>\n"
            message += f"止损价格: <b>{stop_loss:.2f} USDT</b>\n"
            message += f"止盈价格: <b>{take_profit:.2f} USDT</b>\n"
            message += f"盈亏比: <b>{risk_reward_ratio:.2f}:1</b>\n\n"
            
            # FVG信息
            if best_entry.get('fvg'):
                fvg = best_entry['fvg']
                message += f"<b>🎯 FVG入场点</b>\n"
                message += f"时间周期: {best_entry['timeframe']}\n"
                message += f"FVG类型: {'看涨' if fvg['type'] == 'bullish' else '看跌'}\n"
                message += f"FVG区间: {fvg['bottom']:.2f} - {fvg['top']:.2f} USDT\n"
                message += f"FVG大小: {fvg['size_pct']:.2f}%\n"
                message += f"入场点距离: {best_entry['price_distance_pct']:.2f}%\n\n"
            else:
                message += f"<b>🎯 入场方式</b>\n"
                message += f"时间周期: {best_entry['timeframe']}\n"
                message += f"使用ATR计算止损止盈\n\n"
            
            # 均线分析
            message += f"<b>📊 均线策略分析</b>\n"
            message += f"{ma_analysis['details']}\n"
            message += f"均线得分: {ma_analysis['score']}/100\n\n"
            
            # K线形态
            if patterns:
                message += f"<b>🕯️ K线形态识别</b>\n"
                for pattern in patterns:
                    message += f"{pattern}\n"
                message += "\n"
            
            # 技术指标
            message += f"<b>📈 技术指标</b>\n"
            message += f"RSI: {signal_details.get('rsi', 0):.1f}\n"
            message += f"MACD柱: {signal_details.get('macd_hist', 0):.4f}\n\n"
            
            # 其他FVG入场点（如果有）
            all_entries = signal_details.get('all_fvg_entries', [])
            if len(all_entries) > 1:
                message += f"<b>📍 其他可选入场点</b>\n"
                for i, entry in enumerate(all_entries[:3], 1):  # 最多显示3个
                    if entry != best_entry:
                        message += f"{i}. {entry['timeframe']}: {entry['entry_price']:.2f} USDT "
                        message += f"(盈亏比: {entry['risk_reward']:.2f}:1)\n"
                message += "\n"
            
            message += f"<b>⏰ 时间:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            message += f"━━━━━━━━━━━━━━━━━━━━\n"
            message += f"⚠️ <i>此为分析信号，请结合市场情况谨慎操作</i>"
            
        else:
            # 旧版信号格式（兼容）
            price = signal_details['current_price']
            direction = signal_details['direction']
            signal_strength = signal_details['signal_strength']
            
            if direction == 'LONG':
                entry_price = price
                stop_loss = price * (1 - STOP_LOSS_PCT)
                take_profit = price * (1 + TAKE_PROFIT_PCT)
                direction_emoji = "📈"
                direction_text = "做多 (LONG)"
            else:
                entry_price = price
                stop_loss = price * (1 + STOP_LOSS_PCT)
                take_profit = price * (1 - TAKE_PROFIT_PCT)
                direction_emoji = "📉"
                direction_text = "做空 (SHORT)"
            
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            message = f"🎯 <b>ETH 交易信号</b> {direction_emoji}\n\n"
            message += f"━━━━━━━━━━━━━━━━━━━━\n"
            message += f"<b>交易方向:</b> {direction_text}\n"
            message += f"<b>信号强度:</b> {signal_strength:.1f}/100\n\n"
            message += f"<b>💰 价格信息</b>\n"
            message += f"入场价格: <b>{entry_price:.2f} USDT</b>\n"
            message += f"止损价格: <b>{stop_loss:.2f} USDT</b>\n"
            message += f"止盈价格: <b>{take_profit:.2f} USDT</b>\n"
            message += f"盈亏比: <b>{risk_reward_ratio:.2f}:1</b>\n\n"
            message += f"<b>⏰ 时间:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            message += f"━━━━━━━━━━━━━━━━━━━━\n"
        
        log(message)
        return True
        
    except Exception as e:
        print(f"发送交易信号错误: {e}")
        import traceback
        traceback.print_exc()
        return False

# 发送 Telegram 通知
def send_telegram(message):
    """发送消息到 Telegram"""
    # 检查是否已配置
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or \
       TELEGRAM_BOT_TOKEN == "your_telegram_bot_token" or \
       TELEGRAM_CHAT_ID == "your_telegram_chat_id":
        return  # 如果未配置，跳过发送
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code != 200:
            error_detail = response.json() if response.text else {}
            print(f"Telegram 发送失败: {response.status_code} - {error_detail.get('description', '未知错误')}")
            print(f"响应内容: {response.text}")
        else:
            print("Telegram 消息发送成功")
    except Exception as e:
        print(f"Telegram 通知错误: {e}")

# 日志
def log(message, send_to_telegram=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_message + "\n")
    print(log_message)
    # 发送到 Telegram
    if send_to_telegram:
        send_telegram(log_message)

# 设置杠杆
def set_leverage(symbol, leverage):
    """设置合约杠杆"""
    try:
        exchange.set_leverage(leverage, symbol)
        print(f"✅ 设置{symbol}杠杆为{leverage}x")
        return True
    except Exception as e:
        print(f"⚠️ 设置杠杆失败: {e}（可能已设置或测试网不支持）")
        return False

# 查询当前情况
def check_status():
    try:
        # 合约账户余额
        balance = exchange.fetch_balance({'type': 'future'})
        usdt_balance = balance.get('USDT', {}).get('total', 0)
        usdt_free = balance.get('USDT', {}).get('free', 0)
        
        status_message = f"📈 <b>合约账户状态</b>\n\n" \
                        f"USDT总余额: {usdt_balance:.2f} USDT\n" \
                        f"USDT可用余额: {usdt_free:.2f} USDT\n"
        
        # 获取当前持仓（合约）
        positions = exchange.fetch_positions([ETH_SYMBOL])
        active_positions = [pos for pos in positions if float(pos.get('contracts', 0)) != 0]
        
        if active_positions:
            status_message += f"\n<b>当前持仓:</b>\n"
            for pos in active_positions:
                side = pos.get('side', 'unknown')
                contracts = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                mark_price = float(pos.get('markPrice', 0))
                unrealized_pnl = float(pos.get('unrealizedPnl', 0))
                percentage = float(pos.get('percentage', 0))
                leverage = pos.get('leverage', 1)
                
                status_message += f"  {side.upper()}: {abs(contracts)} 张\n"
                status_message += f"  开仓价: {entry_price:.2f} USDT\n"
                status_message += f"  标记价: {mark_price:.2f} USDT\n"
                status_message += f"  杠杆: {leverage}x\n"
                status_message += f"  未实现盈亏: {unrealized_pnl:+.2f} USDT ({percentage:+.2f}%)\n"
        else:
            status_message += "\n无持仓\n"
            
        log(status_message)
        # 读取最后10行日志
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                logs = f.readlines()[-10:]
                print("最近日志:\n" + ''.join(logs))
        except:
            pass
    except Exception as e:
        print(f"查询状态错误: {e}")
        import traceback
        traceback.print_exc()

# 获取当前持仓
def get_current_position():
    """获取当前ETH合约持仓，并从交易记录中读取开仓信息"""
    try:
        import json
        import os
        
        # 获取合约持仓
        positions = exchange.fetch_positions([ETH_SYMBOL])
        active_positions = [pos for pos in positions if float(pos.get('contracts', 0)) != 0]
        
        current_price = exchange.fetch_ticker(ETH_SYMBOL)['last']
        
        if active_positions:
            # 有持仓
            pos = active_positions[0]  # 取第一个持仓
            side = pos.get('side', 'long').upper()
            contracts = float(pos.get('contracts', 0))
            entry_price = float(pos.get('entryPrice', current_price))
            mark_price = float(pos.get('markPrice', current_price))
            
            # 从交易记录中查找止损止盈
            stop_loss = entry_price * (1 - STOP_LOSS_PCT) if side == 'LONG' else entry_price * (1 + STOP_LOSS_PCT)
            take_profit = entry_price * (1 + TAKE_PROFIT_PCT) if side == 'LONG' else entry_price * (1 - TAKE_PROFIT_PCT)
            
            if os.path.exists(TRADE_RECORD_FILE):
                try:
                    with open(TRADE_RECORD_FILE, 'r', encoding='utf-8') as f:
                        records = json.load(f)
                    # 查找最近的开仓记录
                    action_prefix = 'OPEN_LONG' if side == 'LONG' else 'OPEN_SHORT'
                    open_records = [r for r in records if r.get('action') == action_prefix]
                    close_action = 'CLOSE_LONG' if side == 'LONG' else 'CLOSE_SHORT'
                    close_records = [r for r in records if r.get('action') == close_action]
                    # 如果开仓记录数大于平仓记录数，说明有持仓
                    if len(open_records) > len(close_records):
                        last_open = open_records[len(close_records)]
                        entry_price = last_open.get('entry_price', entry_price)
                        stop_loss = last_open.get('stop_loss', stop_loss)
                        take_profit = last_open.get('take_profit', take_profit)
                except:
                    pass
            
            return {
                'side': side,
                'contracts': abs(contracts),
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'current_price': mark_price,
                'unrealized_pnl': float(pos.get('unrealizedPnl', 0)),
                'percentage': float(pos.get('percentage', 0)),
                'leverage': pos.get('leverage', 1)
            }
        else:
            # 无持仓
            return {
                'side': 'NONE',
                'contracts': 0,
                'entry_price': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'current_price': current_price,
                'unrealized_pnl': 0,
                'percentage': 0,
                'leverage': 0
            }
    except Exception as e:
        print(f"获取持仓错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# 执行交易
def execute_trade(signal):
    """根据信号执行合约交易"""
    if not AUTO_TRADE_ENABLED:
        print("⚠️ 自动交易已禁用，仅监控模式")
        return False
    
    try:
        direction = signal['direction']
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        current_price = signal['current_price']
        
        # 设置杠杆
        leverage = LEVERAGE.get(direction, 3)
        set_leverage(ETH_SYMBOL, leverage)
        
        # 检查当前持仓
        position = get_current_position()
        
        # 如果已有同向持仓，不重复开仓
        if position and position['side'] == direction:
            print(f"⚠️ 已有{direction}持仓，跳过开仓")
            return False
        
        # 如果有反向持仓，先平仓
        if position and position['side'] != 'NONE' and position['side'] != direction:
            print(f"🔄 检测到反向持仓，先平仓...")
            close_position(position)
            time.sleep(2)  # 等待订单完成
        
        # 获取合约账户余额
        balance = exchange.fetch_balance({'type': 'future'})
        usdt_balance = balance.get('USDT', {}).get('free', 0)
        
        if usdt_balance < 10:  # 最少需要10 USDT
            print(f"❌ USDT余额不足: {usdt_balance:.2f} USDT")
            return False
        
        # 计算开仓金额（使用80%的可用余额）
        trade_amount_usdt = usdt_balance * 0.8
        
        # 获取交易对信息
        market = exchange.market(ETH_SYMBOL)
        contract_size = float(market.get('contractSize', 1))  # 合约面值
        amount_precision = market['precision']['amount']
        
        # 计算合约数量（考虑杠杆）
        # 合约数量 = (开仓金额 * 杠杆) / (当前价格 * 合约面值)
        contracts = (trade_amount_usdt * leverage) / (current_price * contract_size)
        contracts = round(contracts, amount_precision)
        
        if contracts < market['limits']['amount']['min']:
            print(f"❌ 合约数量太小: {contracts} 张")
            return False
        
        # 执行开仓
        side = 'buy' if direction == 'LONG' else 'sell'
        print(f"🔄 执行开仓: {direction} {contracts} 张 @ {current_price:.2f} USDT (杠杆{leverage}x)")
        
        order = exchange.create_market_order(
            ETH_SYMBOL,
            side,
            contracts,
            None,  # 市价单不需要价格
            None,  # 默认参数
            {
                'leverage': leverage,
                'positionSide': 'BOTH'  # 单向持仓模式
            }
        )
        
        # 记录交易
        record_trade({
            'action': f'OPEN_{direction}',
            'symbol': ETH_SYMBOL,
            'contracts': contracts,
            'price': current_price,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'leverage': leverage,
            'signal_strength': signal['signal_strength'],
            'risk_reward_ratio': signal['risk_reward_ratio'],
            'order_id': order.get('id'),
            'timestamp': datetime.now().isoformat()
        })
        
        trade_msg = f"✅ <b>开仓成功 - {direction} ETH合约</b>\n\n" \
                   f"方向: {direction}\n" \
                   f"合约数量: {contracts} 张\n" \
                   f"开仓价格: {current_price:.2f} USDT\n" \
                   f"杠杆: {leverage}x\n" \
                   f"止损: {stop_loss:.2f} USDT\n" \
                   f"止盈: {take_profit:.2f} USDT\n" \
                   f"信号强度: {signal['signal_strength']:.1f}/100\n" \
                   f"盈亏比: {signal['risk_reward_ratio']:.2f}:1"
        log(trade_msg)
        return True
            
    except Exception as e:
        error_msg = f"❌ 执行交易错误: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        log(error_msg, send_to_telegram=False)
        return False

# 平仓
def close_position(position):
    """平仓当前合约持仓"""
    try:
        if not position or position['side'] == 'NONE':
            return False
        
        contracts = position['contracts']
        side = position['side']
        entry_price = position['entry_price']
        current_price = position['current_price']
        unrealized_pnl = position.get('unrealized_pnl', 0)
        percentage = position.get('percentage', 0)
        
        # 获取交易对信息
        market = exchange.market(ETH_SYMBOL)
        amount_precision = market['precision']['amount']
        contracts = round(contracts, amount_precision)
        
        # 平仓方向：做多平仓用sell，做空平仓用buy
        close_side = 'sell' if side == 'LONG' else 'buy'
        
        print(f"🔄 平仓: {side} {contracts} 张 @ {current_price:.2f} USDT")
        
        # 执行平仓（使用reduceOnly确保只平仓不开新仓）
        order = exchange.create_market_order(
            ETH_SYMBOL,
            close_side,
            contracts,
            None,  # 市价单
            None,
            {
                'reduceOnly': True,  # 只减仓标志
                'positionSide': 'BOTH'
            }
        )
        
        # 计算盈亏
        if side == 'LONG':
            pnl = (current_price - entry_price) * contracts * market.get('contractSize', 1)
        else:
            pnl = (entry_price - current_price) * contracts * market.get('contractSize', 1)
        
        record_trade({
            'action': f'CLOSE_{side}',
            'symbol': ETH_SYMBOL,
            'contracts': contracts,
            'price': current_price,
            'entry_price': entry_price,
            'pnl': unrealized_pnl,  # 使用实际的未实现盈亏
            'pnl_pct': percentage,
            'order_id': order.get('id'),
            'timestamp': datetime.now().isoformat()
        })
        
        close_msg = f"✅ <b>平仓成功</b>\n\n" \
                   f"方向: {side}\n" \
                   f"合约数量: {contracts} 张\n" \
                   f"平仓价格: {current_price:.2f} USDT\n" \
                   f"开仓价格: {entry_price:.2f} USDT\n" \
                   f"盈亏: {unrealized_pnl:+.2f} USDT ({percentage:+.2f}%)"
        log(close_msg)
        return True
            
    except Exception as e:
        error_msg = f"❌ 平仓错误: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        log(error_msg, send_to_telegram=False)
        return False

# 检查止损止盈
def check_stop_loss_take_profit():
    """检查当前持仓是否触发止损或止盈"""
    try:
        position = get_current_position()
        if not position or position['side'] == 'NONE':
            return
        
        current_price = position['current_price']
        stop_loss_price = position.get('stop_loss', 0)
        take_profit_price = position.get('take_profit', 0)
        side = position['side']
        
        if side == 'LONG':
            # 做多：价格下跌触发止损，价格上涨触发止盈
            if stop_loss_price > 0 and current_price <= stop_loss_price:
                print(f"🛑 触发止损: {current_price:.2f} <= {stop_loss_price:.2f}")
                close_position(position)
                log(f"🛑 <b>止损触发，已平仓</b>\n\n"
                    f"方向: {side}\n"
                    f"平仓价格: {current_price:.2f} USDT\n"
                    f"止损价格: {stop_loss_price:.2f} USDT\n"
                    f"开仓价格: {position['entry_price']:.2f} USDT\n"
                    f"盈亏: {position.get('unrealized_pnl', 0):+.2f} USDT", send_to_telegram=True)
            elif take_profit_price > 0 and current_price >= take_profit_price:
                print(f"🎯 触发止盈: {current_price:.2f} >= {take_profit_price:.2f}")
                close_position(position)
                log(f"🎯 <b>止盈触发，已平仓</b>\n\n"
                    f"方向: {side}\n"
                    f"平仓价格: {current_price:.2f} USDT\n"
                    f"止盈价格: {take_profit_price:.2f} USDT\n"
                    f"开仓价格: {position['entry_price']:.2f} USDT\n"
                    f"盈亏: {position.get('unrealized_pnl', 0):+.2f} USDT", send_to_telegram=True)
        elif side == 'SHORT':
            # 做空：价格上涨触发止损，价格下跌触发止盈
            if stop_loss_price > 0 and current_price >= stop_loss_price:
                print(f"🛑 触发止损: {current_price:.2f} >= {stop_loss_price:.2f}")
                close_position(position)
                log(f"🛑 <b>止损触发，已平仓</b>\n\n"
                    f"方向: {side}\n"
                    f"平仓价格: {current_price:.2f} USDT\n"
                    f"止损价格: {stop_loss_price:.2f} USDT\n"
                    f"开仓价格: {position['entry_price']:.2f} USDT\n"
                    f"盈亏: {position.get('unrealized_pnl', 0):+.2f} USDT", send_to_telegram=True)
            elif take_profit_price > 0 and current_price <= take_profit_price:
                print(f"🎯 触发止盈: {current_price:.2f} <= {take_profit_price:.2f}")
                close_position(position)
                log(f"🎯 <b>止盈触发，已平仓</b>\n\n"
                    f"方向: {side}\n"
                    f"平仓价格: {current_price:.2f} USDT\n"
                    f"止盈价格: {take_profit_price:.2f} USDT\n"
                    f"开仓价格: {position['entry_price']:.2f} USDT\n"
                    f"盈亏: {position.get('unrealized_pnl', 0):+.2f} USDT", send_to_telegram=True)
                
    except Exception as e:
        print(f"检查止损止盈错误: {e}")

# 记录交易
def record_trade(trade_data):
    """记录交易到文件"""
    try:
        import json
        import os
        
        # 读取现有记录
        if os.path.exists(TRADE_RECORD_FILE):
            with open(TRADE_RECORD_FILE, 'r', encoding='utf-8') as f:
                records = json.load(f)
        else:
            records = []
        
        # 添加新记录
        records.append(trade_data)
        
        # 保存记录
        with open(TRADE_RECORD_FILE, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"记录交易错误: {e}")

# 获取交易统计
def get_trade_statistics():
    """获取交易统计信息"""
    try:
        import json
        import os
        
        if not os.path.exists(TRADE_RECORD_FILE):
            return None
        
        with open(TRADE_RECORD_FILE, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        if not records:
            return None
        
        # 统计
        total_trades = len(records)
        closed_trades = [r for r in records if r.get('action', '').startswith('CLOSE')]
        total_pnl = sum([r.get('pnl', 0) for r in closed_trades])
        winning_trades = len([r for r in closed_trades if r.get('pnl', 0) > 0])
        losing_trades = len([r for r in closed_trades if r.get('pnl', 0) < 0])
        win_rate = (winning_trades / len(closed_trades) * 100) if closed_trades else 0
        
        return {
            'total_trades': total_trades,
            'closed_trades': len(closed_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl
        }
    except Exception as e:
        print(f"获取交易统计错误: {e}")
        return None

# ETH实时监控函数（使用高级分析）
def monitor_eth():
    """实时监控ETH走势，使用均线+形态+FVG+深度学习分析，发现交易机会时自动交易"""
    try:
        # 1. 先检查当前持仓的止损止盈
        check_stop_loss_take_profit()
        
        # 2. 评估历史信号（更新信号质量）
        evaluate_signal_history()
        
        # 3. 使用高级分析（均线+形态+FVG+深度学习）
        signal = analyze_eth_advanced()
        
        if signal:
            # 发现强信号
            print(f"✅ 发现ETH交易信号: {signal['direction']}, 强度: {signal['signal_strength']:.1f}, "
                  f"入场: {signal['entry_price']:.2f}, 盈亏比: {signal['risk_reward_ratio']:.2f}:1")
            
            # 记录信号历史（用于深度学习）
            record_signal_history(signal)
            
            # 检查是否需要训练模型（每100个新信号后）
            if os.path.exists(SIGNAL_HISTORY_FILE):
                with open(SIGNAL_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                if len(history) % DL_TRAIN_INTERVAL == 0 and len(history) >= DL_MIN_SIGNALS_FOR_TRAIN:
                    print(f"🔄 检测到{len(history)}个信号，开始训练深度学习模型...")
                    train_deep_learning_model()
            
            # 发送详细通知
            send_trading_signal(signal)
            
            # 如果启用自动交易，执行交易
            if AUTO_TRADE_ENABLED:
                print("🤖 自动交易已启用，准备执行交易...")
                execute_trade(signal)
            else:
                print("⚠️ 自动交易已禁用，仅发送信号通知")
        else:
            # 无强信号，仅记录日志（不发送Telegram）
            try:
                current_price = exchange.fetch_ticker(ETH_SYMBOL)['last']
                position = get_current_position()
                position_info = ""
                if position and position['side'] != 'NONE':
                    pnl_pct = position.get('percentage', 0)
                    unrealized_pnl = position.get('unrealized_pnl', 0)
                    contracts = position.get('contracts', 0)
                    position_info = f" | 持仓: {position['side']} {contracts}张 | 盈亏: {unrealized_pnl:+.2f} USDT ({pnl_pct:+.2f}%)"
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_message = f"[{timestamp}] ETH监控中... 当前价格: {current_price:.2f} USDT (无强信号){position_info}"
                print(log_message)
                with open(LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(log_message + "\n")
            except:
                pass
                
    except Exception as e:
        error_msg = f"监控ETH错误: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        log(error_msg, send_to_telegram=False)

# 分析所有币种的走势（用于参考）
def analyze_all_coins():
    """分析所有监控币种的走势，用于参考"""
    try:
        summary = "📊 <b>市场走势分析</b>\n\n"
        summary += f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for coin in COINS:
            symbol = f"{coin}/USDT"
            try:
                df = get_historical_data(symbol, limit=200)
                if df.empty:
                    continue
                    
                current = df.iloc[-1]
                price = current['close']
                rsi = current['rsi']
                ma_trend = "📈" if current['close'] > current['ema50'] else "📉"
                macd_trend = "📈" if current['macd_hist'] > 0 else "📉"
                
                summary += f"<b>{coin}</b> {ma_trend}\n"
                summary += f"  价格: {price:.2f} USDT\n"
                summary += f"  RSI: {rsi:.1f} | MACD: {macd_trend}\n\n"
                
            except Exception as e:
                summary += f"<b>{coin}</b>: 分析失败\n\n"
        
        # 只在有多个币种数据时发送
        if len(COINS) > 1:
            send_telegram(summary)
            
    except Exception as e:
        print(f"分析所有币种错误: {e}")

# 测试 Telegram 连接
def test_telegram():
    """测试 Telegram 配置是否正确"""
    print("正在测试 Telegram 连接...")
    test_message = "🧪 <b>Telegram 连接测试</b>\n\n如果您收到这条消息，说明配置成功！"
    send_telegram(test_message)
    print("\n提示：如果收到 404 错误，请检查：")
    print("1. Bot Token 是否正确（从 @BotFather 获取）")
    print("2. Chat ID 是否正确（从 @userinfobot 获取）")
    print("3. 是否已向 Bot 发送过至少一条消息（Bot 需要先收到您的消息才能向您发送）")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check":
            check_status()
            # 显示交易统计
            stats = get_trade_statistics()
            if stats:
                print(f"\n📊 交易统计:")
                print(f"总交易次数: {stats['total_trades']}")
                print(f"已平仓次数: {stats['closed_trades']}")
                print(f"盈利次数: {stats['winning_trades']}")
                print(f"亏损次数: {stats['losing_trades']}")
                print(f"胜率: {stats['win_rate']:.2f}%")
                print(f"总盈亏: {stats['total_pnl']:+.2f} USDT")
        elif sys.argv[1] == "--test-telegram":
            test_telegram()
        elif sys.argv[1] == "--analyze-eth":
            # 立即分析一次ETH（使用高级分析）
            print("正在分析ETH走势（均线+形态+FVG）...")
            signal = analyze_eth_advanced()
            if signal:
                send_trading_signal(signal)
                print(f"✅ 发现信号: {signal['direction']}, 强度: {signal['signal_strength']:.1f}")
                if AUTO_TRADE_ENABLED:
                    print("🤖 自动交易已启用，准备执行...")
                    execute_trade(signal)
            else:
                print("当前无强交易信号")
        elif sys.argv[1] == "--analyze-all":
            # 分析所有币种
            analyze_all_coins()
        elif sys.argv[1] == "--stats":
            # 显示交易统计
            stats = get_trade_statistics()
            if stats:
                stats_msg = f"📊 <b>交易统计</b>\n\n" \
                           f"总交易次数: {stats['total_trades']}\n" \
                           f"已平仓次数: {stats['closed_trades']}\n" \
                           f"盈利次数: {stats['winning_trades']}\n" \
                           f"亏损次数: {stats['losing_trades']}\n" \
                           f"胜率: {stats['win_rate']:.2f}%\n" \
                           f"总盈亏: {stats['total_pnl']:+.2f} USDT"
                log(stats_msg)
            else:
                print("暂无交易记录")
    else:
        # 启动实时监控
        trade_mode = "🤖 自动交易模式" if AUTO_TRADE_ENABLED else "👁️ 仅监控模式（不执行实际交易）"
        startup_message = f"🤖 <b>ETH AI合约交易机器人启动</b>\n\n" \
                         f"{trade_mode}\n" \
                         f"📈 交易类型: 永续合约\n" \
                         f"🎯 专注币种: ETH/USDT:USDT\n" \
                         f"📊 监控币种: {', '.join(COINS)}\n" \
                         f"⏱️ 时间周期: {TIMEFRAME}\n" \
                         f"⚡ 杠杆倍数: {LEVERAGE['LONG']}x (做多/做空)\n" \
                         f"📈 信号阈值: {SIGNAL_THRESHOLD}/100\n" \
                         f"💰 止损: {STOP_LOSS_PCT*100:.1f}% | 止盈: {TAKE_PROFIT_PCT*100:.1f}%\n" \
                         f"🔄 监控间隔: {MONITOR_INTERVAL//60}分钟\n" \
                         f"━━━━━━━━━━━━━━━━━━━━\n" \
                         f"⚠️ <b>当前为仅监控模式，不会执行实际交易</b>\n" \
                         f"✅ 机器人已启动，开始实时监控ETH合约走势并发送信号..."
        log(startup_message)
        
        # 显示当前账户状态
        check_status()
        
        # 显示交易统计（如果有）
        stats = get_trade_statistics()
        if stats:
            print(f"\n📊 历史交易统计:")
            print(f"总交易次数: {stats['total_trades']} | 已平仓: {stats['closed_trades']}")
            print(f"胜率: {stats['win_rate']:.2f}% | 总盈亏: {stats['total_pnl']:+.2f} USDT\n")
        
        # 立即执行一次分析
        monitor_eth()
        
        # 定时任务：每5分钟监控一次ETH
        schedule.every(MONITOR_INTERVAL // 60).minutes.do(monitor_eth)
        
        # 每1分钟检查一次止损止盈（更频繁检查）
        schedule.every(1).minutes.do(check_stop_loss_take_profit)
        
        # 每小时显示一次账户状态
        schedule.every().hour.do(check_status)
        
        # 每天分析一次所有币种（可选）
        schedule.every().day.at("09:00").do(analyze_all_coins)
        
        # 每天显示交易统计
        schedule.every().day.at("20:00").do(lambda: (
            stats := get_trade_statistics(),
            stats and log(f"📊 <b>每日交易统计</b>\n\n"
                         f"总交易: {stats['total_trades']} | 已平仓: {stats['closed_trades']}\n"
                         f"胜率: {stats['win_rate']:.2f}% | 总盈亏: {stats['total_pnl']:+.2f} USDT")
        ))
        
        # 每天凌晨2点训练深度学习模型（如果数据足够）
        schedule.every().day.at("02:00").do(lambda: (
            print("🔄 开始定期训练深度学习模型..."),
            train_deep_learning_model()
        ))
        
        # 每天凌晨3点执行自我修正
        schedule.every().day.at("03:00").do(lambda: (
            print("🧠 开始算法自我修正..."),
            self_correct_trading_algorithm()
        ))
        
        # 检查是否需要训练模型（每100个新信号后）
        def check_and_train_model():
            if os.path.exists(SIGNAL_HISTORY_FILE):
                with open(SIGNAL_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                if len(history) % DL_TRAIN_INTERVAL == 0 and len(history) >= DL_MIN_SIGNALS_FOR_TRAIN:
                    print(f"🔄 检测到{len(history)}个信号，开始训练深度学习模型...")
                    train_deep_learning_model()
        
        print(f"\n✅ 机器人运行中...")
        print(f"   - 每{MONITOR_INTERVAL//60}分钟检查一次ETH信号")
        print(f"   - 每1分钟检查一次止损止盈")
        print(f"   - 自动交易: {'已启用' if AUTO_TRADE_ENABLED else '已禁用（仅监控模式）'}")
        if TENSORFLOW_AVAILABLE:
            print(f"   - 🤖 深度学习: 已启用（LSTM模型）")
            print(f"   - 🧠 自我修正: 每天03:00自动执行")
        else:
            print(f"   - ⚠️  深度学习: 未安装TensorFlow（pip install tensorflow）")
        if not AUTO_TRADE_ENABLED:
            print(f"   ⚠️  注意：当前为仅监控模式，不会执行实际交易")
            print(f"   ⚠️  如需启用自动交易，请将 AUTO_TRADE_ENABLED 设置为 True")
        print(f"按 Ctrl+C 停止\n")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            log("🛑 机器人已停止", send_to_telegram=True)
            print("\n机器人已停止")
            # 显示最终统计
            stats = get_trade_statistics()
            if stats:
                print(f"\n📊 最终交易统计:")
                print(f"总交易次数: {stats['total_trades']}")
                print(f"已平仓次数: {stats['closed_trades']}")
                print(f"胜率: {stats['win_rate']:.2f}%")
                print(f"总盈亏: {stats['total_pnl']:+.2f} USDT")