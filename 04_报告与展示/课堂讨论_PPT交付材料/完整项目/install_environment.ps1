param([string]$BasePython = '')
$ErrorActionPreference = 'Stop'
$environmentDir = Join-Path $PSScriptRoot '.venv'
$runtimePython = Join-Path $environmentDir 'Scripts\python.exe'
New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot 'cache\tmp') -Force | Out-Null
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot 'cache\pip'
$env:TEMP = Join-Path $PSScriptRoot 'cache\tmp'
$env:TMP = $env:TEMP
if (-not (Test-Path -LiteralPath $runtimePython)) {
    if ($BasePython) {
        & $BasePython -c "import sys; assert sys.version_info[:2] == (3, 9), 'Python 3.9 is required'"
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.9 is required.' }
        & $BasePython -m venv $environmentDir
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.9 -m venv $environmentDir
    } else {
        & python -c "import sys; assert sys.version_info[:2] == (3, 9), 'Python 3.9 is required'"
        if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.9, then rerun this script.' }
        & python -m venv $environmentDir
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the project environment.' }
}
& $runtimePython -m pip install 'https://download.pytorch.org/whl/cpu/torch-2.1.2%2Bcpu-cp39-cp39-win_amd64.whl' --progress-bar off
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $runtimePython -m pip install -r (Join-Path $PSScriptRoot 'requirements-runtime.txt') --index-url https://mirrors.aliyun.com/pypi/simple --progress-bar off --no-compile
if ($LASTEXITCODE -ne 0) { throw 'Runtime installation failed.' }
& $runtimePython -m pip install --no-index 'torch-scatter==2.1.2+pt21cpu' 'torch-sparse==0.6.18+pt21cpu' -f 'https://data.pyg.org/whl/torch-2.1.0+cpu.html' --progress-bar off
if ($LASTEXITCODE -ne 0) { throw 'Graph extension installation failed.' }
& $runtimePython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
