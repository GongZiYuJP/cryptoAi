"""
运行交易系统的便捷脚本
"""

from cryptoAi2 import CryptoContractTrader
import sys
import os

# 尝试导入配置文件
try:
    from config import EXCHANGE_CONFIG, TELEGRAM_CONFIG, TRADING_CONFIG
except ImportError:
    print("❌ 未找到 config.py 文件")
    print("请复制 config.py 为 config.py 并填入您的配置信息")
    sys.exit(1)

def main():
    """主函数"""
    print("=" * 50)
    print("加密货币合约自动交易系统")
    print("=" * 50)
    print()
    
    # 检查配置
    if EXCHANGE_CONFIG['apiKey'] == 'YOUR_API_KEY':
        print("⚠️ 警告: 请先配置交易所API密钥")
        response = input("是否继续使用公共API（仅监控，无法交易）? (y/n): ")
        if response.lower() != 'y':
            print("已取消")
            return
    
    if TELEGRAM_CONFIG['token'] == 'YOUR_TELEGRAM_BOT_TOKEN':
        print("⚠️ 警告: 未配置Telegram，将无法发送通知")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            print("已取消")
            return
    
    # 创建交易系统
    trader = CryptoContractTrader(
        exchange_config=EXCHANGE_CONFIG,
        telegram_config=TELEGRAM_CONFIG,
        portfolio_size=TRADING_CONFIG['portfolio_size']
    )
    
    # 运行系统
    try:
        trader.run(
            symbol=TRADING_CONFIG['symbol'],
            interval=TRADING_CONFIG['check_interval']
        )
    except KeyboardInterrupt:
        print("\n系统已停止")

if __name__ == "__main__":
    main()

