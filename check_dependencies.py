"""
检查依赖包是否已安装
"""

import sys
import io

# 设置UTF-8编码输出（Windows兼容）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} - 已安装")
        return True
    except ImportError:
        print(f"❌ {package_name} - 未安装")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("检查依赖包安装情况")
    print("=" * 50)
    print()
    
    # 需要检查的包
    packages = [
        ('ccxt', 'ccxt'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('requests', 'requests'),
        ('ta', 'ta'),
    ]
    
    # 内置模块（不需要安装）
    builtin_modules = [
        'time',
        'json',
        'os',
        'datetime',
        'typing',
    ]
    
    print("📦 检查第三方包:")
    print("-" * 50)
    missing_packages = []
    
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    print()
    print("📦 检查内置模块:")
    print("-" * 50)
    for module in builtin_modules:
        try:
            __import__(module)
            print(f"✅ {module} - 可用")
        except ImportError:
            print(f"❌ {module} - 不可用（这不应该发生）")
    
    print()
    print("=" * 50)
    
    if missing_packages:
        print(f"❌ 缺少以下包: {', '.join(missing_packages)}")
        print()
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        print()
        print("或者安装所有依赖:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("✅ 所有依赖包已安装！")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

