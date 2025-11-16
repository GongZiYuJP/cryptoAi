# 加密货币合约自动交易系统

根据高胜率策略构建指南实现的自动交易系统。

## 功能特点

### 1. 多时间框架分析
- **4小时定趋势**：判断大趋势方向
- **1小时找入口**：寻找合适的入场点
- **15分钟精确定位**：精确确定入场价格

### 2. 入场条件（严格按照指南）
- ✅ **EMA金叉**：短期EMA上穿长期EMA
- ✅ **RSI超跌回升**：RSI从超卖区域（<30）回升
- ✅ **关键支撑位企稳**：价格在支撑位附近（2%以内）企稳

### 3. 出场策略
- ✅ **止损**：设在支撑下方2%
- ✅ **阶段性止盈**：1.5倍、2倍、3倍风险的多级止盈
- ✅ **移动止损**：可根据需要实现（代码中预留接口）

### 4. 仓位管理
- ✅ **风险控制**：单笔亏损不超过总资金1-2%（默认1.5%）
- ✅ **自动计算仓位**：根据止损距离自动计算合适的仓位大小

### 5. 交易日记
- ✅ **完整记录**：记录每笔交易的进出场理由、信号强度等
- ✅ **JSON格式**：便于后续分析和回测

### 6. Telegram通知
- ✅ **买入信号**：详细的买入信号通知
- ✅ **卖出信号**：详细的卖出信号通知
- ✅ **系统状态**：启动、停止、错误等状态通知

## 安装依赖

```bash
pip install ccxt pandas numpy requests ta
```

或者使用 requirements.txt：

```bash
pip install -r requirements.txt
```

## 配置步骤

### 1. 配置交易所API

1. 登录币安（或其他交易所）
2. 创建API密钥（需要合约交易权限）
3. 复制API Key和Secret

### 2. 配置Telegram Bot

1. 在Telegram中搜索 `@BotFather`
2. 发送 `/newbot` 创建新Bot
3. 获取Bot Token
4. 在Telegram中搜索 `@userinfobot` 获取您的Chat ID
5. 向您的Bot发送任意消息（Bot需要先收到您的消息才能向您发送）

### 3. 创建配置文件

```bash
# 复制配置示例文件
cp config.py config.py

# 编辑配置文件，填入您的API密钥和Telegram信息
# Windows: notepad config.py
# Linux/Mac: nano config.py
```

编辑 `config.py`：

```python
EXCHANGE_CONFIG = {
    'exchange': 'binance',
    'apiKey': '您的API_KEY',
    'secret': '您的SECRET'
}

TELEGRAM_CONFIG = {
    'token': '您的BOT_TOKEN',
    'chat_id': '您的CHAT_ID'
}

TRADING_CONFIG = {
    'portfolio_size': 1000.0,  # 总资金
    'risk_per_trade': 0.015,  # 单笔风险1.5%
    'symbol': 'ETH/USDT:USDT',  # 交易对
    'check_interval': 300  # 检查间隔（秒）
}
```

## 使用方法

### 方法1：使用便捷脚本（推荐）

```bash
python run_trader.py
```

### 方法2：直接运行主程序

```bash
python cryptoAi2.py
```

### 方法3：在代码中使用

```python
from cryptoAi2 import CryptoContractTrader

# 配置
exchange_config = {
    'exchange': 'binance',
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET'
}

telegram_config = {
    'token': 'YOUR_BOT_TOKEN',
    'chat_id': 'YOUR_CHAT_ID'
}

# 创建交易系统
trader = CryptoContractTrader(
    exchange_config=exchange_config,
    telegram_config=telegram_config,
    portfolio_size=1000.0
)

# 运行（每5分钟检查一次）
trader.run(symbol='ETH/USDT:USDT', interval=300)
```

## 交易策略详解

### 信号生成流程

1. **4小时级别分析**
   - 判断EMA排列（20>50为多头，20<50为空头）
   - 检查价格与EMA关系
   - MACD确认趋势

2. **1小时级别分析**
   - 检查EMA金叉
   - 检查RSI超跌回升
   - 检查关键支撑位企稳
   - 确保方向与4小时一致

3. **15分钟级别分析**
   - 精确确认入场点
   - 再次验证支撑位
   - 确认RSI状态

4. **信号生成**
   - 只有三个时间框架都满足条件时才生成信号
   - 计算止损止盈
   - 计算仓位大小

### 止损止盈计算

- **止损**：支撑位下方2%（确保不超过入场价的2%）
- **止盈1**：入场价 + 1.5倍风险
- **止盈2**：入场价 + 2.0倍风险
- **止盈3**：入场价 + 3.0倍风险（最终止盈）

### 仓位计算

```
风险金额 = 总资金 × 风险比例（1.5%）
风险每合约 = |入场价 - 止损价|
仓位大小 = 风险金额 / 风险每合约
```

## 文件说明

- `cryptoAi2.py` - 主程序文件
- `config_example.py` - 配置文件示例
- `config.py` - 实际配置文件（需要自己创建）
- `run_trader.py` - 便捷运行脚本
- `trading_journal.json` - 交易日记（自动生成）
- `active_positions.json` - 持仓记录（自动生成）

## 注意事项

⚠️ **重要提示**：

1. **测试环境**：建议先在测试环境或模拟盘测试
2. **API权限**：确保API密钥有合约交易权限，但建议限制为只读+交易，不要给提现权限
3. **资金管理**：不要使用超过您能承受损失的资金
4. **市场风险**：加密货币市场波动大，请谨慎操作
5. **网络稳定**：确保网络连接稳定，避免因网络问题导致交易失败

## 交易日记格式

交易日记保存在 `trading_journal.json`，包含：

- 时间戳
- 操作类型（BUY/SELL）
- 交易对
- 方向（LONG/SHORT）
- 入场价格
- 止损价格
- 止盈价格
- 仓位大小
- 分析理由
- 信号强度
- 交易结果（平仓时）

## 常见问题

### Q: 如何修改检查间隔？

A: 修改 `config.py` 中的 `check_interval` 参数（单位：秒）

### Q: 如何修改风险比例？

A: 修改 `config.py` 中的 `risk_per_trade` 参数（0.01 = 1%, 0.02 = 2%）

### Q: 如何更换交易对？

A: 修改 `config.py` 中的 `symbol` 参数，格式为 `币种/USDT:USDT`（合约格式）

### Q: Telegram通知收不到？

A: 检查：
1. Bot Token是否正确
2. Chat ID是否正确
3. 是否已向Bot发送过至少一条消息
4. 网络连接是否正常

### Q: 系统一直显示"未发现交易信号"？

A: 这是正常的，系统只在满足所有条件时才生成信号，确保信号质量。

## 技术支持

如有问题，请检查：
1. 日志输出
2. 交易日记文件
3. Telegram错误消息

## 免责声明

本系统仅供学习和研究使用。使用本系统进行实际交易的风险由用户自行承担。作者不对任何交易损失负责。

