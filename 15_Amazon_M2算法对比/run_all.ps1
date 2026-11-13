$ErrorActionPreference='Stop'
$env:PYTHONIOENCODING='utf-8'
$python='D:\Envs\recbole-gnn-py39\Scripts\python.exe'
function Run-Python([string]$name,[string[]]$pythonArgs=@()){
 & $python -u (Join-Path $PSScriptRoot $name) @pythonArgs
 if($LASTEXITCODE -ne 0){throw "$name failed"}
}
$preparedFiles=@('run\manifest.json','run\data.json','run\attributes.npz')
if(@($preparedFiles | Where-Object {-not(Test-Path -LiteralPath (Join-Path $PSScriptRoot $_))}).Count -gt 0){Run-Python 'prepare.py'}
foreach($model in @('SRGNN','GCSAN','GCEGNN')){
 if((-not(Test-Path -LiteralPath (Join-Path $PSScriptRoot "run\results\${model}_training.json"))) -or (-not(Test-Path -LiteralPath (Join-Path $PSScriptRoot "run\models\${model}.pt")))){Run-Python 'train.py' @('--model',$model)}
}
if(-not(Test-Path -LiteralPath (Join-Path $PSScriptRoot 'run\results\frozen_policy.json'))){
 & (Join-Path $PSScriptRoot 'start_qwen.ps1')
 try{Run-Python 'policy.py'}finally{
  $localServiceId=[int](Get-Content -LiteralPath (Join-Path $PSScriptRoot 'run\native.pid'))
  $localService=Get-CimInstance Win32_Process -Filter "ProcessId=$localServiceId" -ErrorAction SilentlyContinue
  if($localService -and $localService.Name -eq 'llama-server.exe' -and $localService.CommandLine -like '*11437*'){Stop-Process -Id $localServiceId}
 }
}
$predictionMethods=@('popularity','grey','grey_improved','fixed_policy','qwen_raw','qwen_offline','SRGNN','GCSAN','GCEGNN')
$missingPredictions=@($predictionMethods | Where-Object {-not(Test-Path -LiteralPath (Join-Path $PSScriptRoot "run\predictions\$_.jsonl"))})
if((-not(Test-Path -LiteralPath (Join-Path $PSScriptRoot 'run\results\comparison.json'))) -or $missingPredictions.Count -gt 0){Run-Python 'evaluate.py'}
Run-Python 'verify.py'
Run-Python 'report.py'
