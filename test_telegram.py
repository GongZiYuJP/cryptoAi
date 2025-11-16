"""
测试Telegram连接
"""

import requests
import sys

def test_telegram(token: str, chat_id: str):
    """测试Telegram Bot连接"""
    print("正在测试Telegram连接...")
    print(f"Bot Token: {token[:10]}...")
    print(f"Chat ID: {chat_id}")
    print()
    
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': '🧪 <b>Telegram连接测试</b>\n\n如果您收到这条消息，说明配置成功！',
            'parse_mode': 'HTML'
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            print("✅ Telegram连接成功！")
            print("您应该已经收到测试消息。")
            return True
        else:
            error_data = response.json() if response.text else {}
            error_desc = error_data.get('description', '未知错误')
            print(f"❌ Telegram连接失败: {response.status_code}")
            print(f"错误信息: {error_desc}")
            
            if response.status_code == 401:
                print("\n💡 提示: Bot Token可能不正确，请检查：")
                print("   1. 从 @BotFather 获取正确的Token")
                print("   2. 确保Token没有多余的空格")
            elif response.status_code == 400:
                print("\n💡 提示: Chat ID可能不正确，请检查：")
                print("   1. 从 @userinfobot 获取您的Chat ID")
                print("   2. 确保已向Bot发送过至少一条消息")
            
            return False
            
    except requests.exceptions.Timeout:
        print("❌ 连接超时，请检查网络连接")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 连接错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False

if __name__ == "__main__":
    # 尝试从配置文件读取
    try:
        from config import TELEGRAM_CONFIG
        token = TELEGRAM_CONFIG['token']
        chat_id = TELEGRAM_CONFIG['chat_id']
    except ImportError:
        print("❌ 未找到 config.py 文件")
        print("请先创建配置文件，或手动输入Token和Chat ID")
        token = input("请输入Bot Token: ").strip()
        chat_id = input("请输入Chat ID: ").strip()
    except KeyError:
        print("⚠️ 配置文件中缺少Telegram配置")
        token = input("请输入Bot Token: ").strip()
        chat_id = input("请输入Chat ID: ").strip()
    
    if not token or token == 'YOUR_TELEGRAM_BOT_TOKEN':
        print("❌ Bot Token未配置")
        sys.exit(1)
    
    if not chat_id or chat_id == 'YOUR_TELEGRAM_CHAT_ID':
        print("❌ Chat ID未配置")
        sys.exit(1)
    
    success = test_telegram(token, chat_id)
    sys.exit(0 if success else 1)

