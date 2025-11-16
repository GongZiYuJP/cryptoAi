# Python 3.11 虚拟环境设置脚本

Write-Host "正在检查 Python 3.11..." -ForegroundColor Cyan

# 检查 Python 3.11 是否已安装
$python311 = py -3.11 --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: 未找到 Python 3.11" -ForegroundColor Red
    Write-Host "请先安装 Python 3.11:" -ForegroundColor Yellow
    Write-Host "1. 访问: https://www.python.org/downloads/release/python-31110/" -ForegroundColor Yellow
    Write-Host "2. 下载并安装 Python 3.11.10" -ForegroundColor Yellow
    Write-Host "3. 安装时勾选 'Add Python 3.11 to PATH'" -ForegroundColor Yellow
    exit 1
}

Write-Host "找到 Python 3.11: $python311" -ForegroundColor Green

# 创建虚拟环境
Write-Host "`n正在创建虚拟环境 (venv311)..." -ForegroundColor Cyan
py -3.11 -m venv venv311

if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: 虚拟环境创建失败" -ForegroundColor Red
    exit 1
}

Write-Host "虚拟环境创建成功!" -ForegroundColor Green

# 激活虚拟环境
Write-Host "`n正在激活虚拟环境..." -ForegroundColor Cyan
& .\venv311\Scripts\Activate.ps1

if ($LASTEXITCODE -ne 0) {
    Write-Host "警告: 无法自动激活虚拟环境" -ForegroundColor Yellow
    Write-Host "请手动运行: .\venv311\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "如果遇到执行策略错误，运行:" -ForegroundColor Yellow
    Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
} else {
    Write-Host "虚拟环境已激活!" -ForegroundColor Green
}

# 升级 pip
Write-Host "`n正在升级 pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# 安装依赖
Write-Host "`n正在安装项目依赖..." -ForegroundColor Cyan
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ 设置完成!" -ForegroundColor Green
    Write-Host "`n使用以下命令激活虚拟环境:" -ForegroundColor Cyan
    Write-Host ".\venv311\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "`n然后运行项目:" -ForegroundColor Cyan
    Write-Host "python cryptoAiBot.py" -ForegroundColor White
} else {
    Write-Host "`n❌ 依赖安装失败" -ForegroundColor Red
    exit 1
}

