"""
配置文件示例
请复制此文件为 config.py 并填入您的实际配置信息
"""

# 交易所配置
EXCHANGE_CONFIG = {
    'exchange': 'binance',  # 交易所名称
    'apiKey': '',  # 您的API密钥
    'secret': ''  # 您的API密钥
}

# Telegram配置
TELEGRAM_CONFIG = {
    'token': '',  # Bot Token（从 @BotFather 获取）
    'chat_id': ''  # Chat ID（从 @userinfobot 获取）
}

# 交易配置
TRADING_CONFIG = {
    'portfolio_size': 1000.0,  # 总资金（USDT）
    'risk_per_trade': 0.015,  # 单笔风险（1.5%，在1-2%之间）
    'symbol': 'ETH/USDT:USDT',  # 交易对（合约格式）
    'check_interval': 300  # 检查间隔（秒），默认5分钟
}

