$ErrorActionPreference = 'Stop'
$basePython = 'C:\anacode\envs\econml-py39\python.exe'
$runtimePython = 'D:\Envs\recbole-gnn-py39\Scripts\python.exe'
New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot 'cache\tmp') -Force | Out-Null
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot 'cache\pip'
$env:TEMP = Join-Path $PSScriptRoot 'cache\tmp'
$env:TMP = $env:TEMP
if (-not (Test-Path -LiteralPath $runtimePython)) {
    & $basePython -m venv 'D:\Envs\recbole-gnn-py39'
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated environment.' }
}
# Direct official URL avoids the slow index-selected download-r2 mirror on this machine.
& $runtimePython -m pip install 'https://download.pytorch.org/whl/cpu/torch-2.1.2%2Bcpu-cp39-cp39-win_amd64.whl' --progress-bar off
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $runtimePython -m pip install -r (Join-Path $PSScriptRoot 'requirements-runtime.txt') --index-url https://mirrors.aliyun.com/pypi/simple --progress-bar off --no-compile
if ($LASTEXITCODE -ne 0) { throw 'Runtime installation failed.' }
& $runtimePython -m pip install --no-index 'torch-scatter==2.1.2+pt21cpu' 'torch-sparse==0.6.18+pt21cpu' -f 'https://data.pyg.org/whl/torch-2.1.0+cpu.html' --progress-bar off
if ($LASTEXITCODE -ne 0) { throw 'Graph extension installation failed.' }
& $runtimePython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
