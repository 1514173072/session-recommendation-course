$ErrorActionPreference='Stop'
if(Get-NetTCPConnection -State Listen -LocalPort 11437 -ErrorAction SilentlyContinue){throw 'Port 11437 already occupied'}
$env:GGML_BACKEND_PATH='D:\CodexTools\ollama\lib\ollama\cuda_v13\ggml-cuda.dll'
$env:PATH='D:\CodexTools\ollama\lib\ollama;D:\CodexTools\ollama\lib\ollama\cuda_v13;'+$env:PATH
$env:GGML_CUDA_INIT='1'
$nativeArgs=@('-m','D:\Models\Ollama\blobs\sha256-20cb277f0967ace47b0b5d5658e5e494a88937f7378b74b5a266496400938f4c','--host','127.0.0.1','--port','11437','-c','2048','-np','1','--device','CUDA0','-ngl','all','-t','4','-tb','4','-fa','on','-b','64','-ub','64','--no-mmproj','--no-webui','--offline','--log-verbosity','2')
$nativeProc=Start-Process -FilePath 'D:\CodexTools\ollama\lib\ollama\llama-server.exe' -ArgumentList $nativeArgs -WindowStyle Hidden -WorkingDirectory 'D:\CodexTools\ollama\lib\ollama' -RedirectStandardOutput (Join-Path $PSScriptRoot 'run\logs\native.stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'run\logs\native.stderr.log') -PassThru
$nativeProc.Id | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'run\native.pid')
Write-Output "Started task-local Qwen PID=$($nativeProc.Id) port11437"
