param(
    [ValidateSet('SRGNN', 'GCSAN', 'GCEGNN')]
    [string[]]$Models = @('SRGNN', 'GCSAN', 'GCEGNN'),
    [int]$Epochs = 3,
    [switch]$InferenceOnly,
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$runtimePython = if ($Python) { $Python } else { Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $runtimePython)) { throw 'Run install_environment.ps1 first.' }
$env:PYTHONIOENCODING = 'utf-8'
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot 'cache\pip'
$env:TEMP = Join-Path $PSScriptRoot 'cache\tmp'
$env:TMP = $env:TEMP
$consoleDir = Join-Path $PSScriptRoot 'outputs\console'
New-Item -ItemType Directory -Path $consoleDir -Force | Out-Null
Push-Location $PSScriptRoot
try {
    foreach ($modelName in $Models) {
        $scriptArguments = @('-u', 'scripts\run_model.py', '--model', $modelName, '--epochs', "$Epochs")
        if ($InferenceOnly) { $scriptArguments += '--inference-only' }
        $logSuffix = if ($InferenceOnly) { 'inference' } else { 'train' }
        # Windows PowerShell 5 treats progress output on stderr as NativeCommandError.
        # Capture it as log text and determine success from the actual process exit code.
        $ErrorActionPreference = 'Continue'
        & $runtimePython @scriptArguments 2>&1 | Tee-Object -FilePath (Join-Path $consoleDir "$modelName-$logSuffix.log")
        $modelExitCode = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($modelExitCode -ne 0) { throw "$modelName failed with exit code $modelExitCode; see outputs\console." }
    }
} finally { Pop-Location }
