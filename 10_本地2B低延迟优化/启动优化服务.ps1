param([ValidateSet(8,16,32,128,512)][int]$BatchSize=32)
$ErrorActionPreference='Stop'
$taskDir='D:\Projects\算法分析与理论课程\10_本地2B低延迟优化'
$env:GGML_BACKEND_PATH='D:\CodexTools\ollama\lib\ollama\cuda_v13\ggml-cuda.dll'
$env:PATH='D:\CodexTools\ollama\lib\ollama;D:\CodexTools\ollama\lib\ollama\cuda_v13;'+$env:PATH
$env:GGML_CUDA_INIT='1'
if (Get-NetTCPConnection -State Listen -LocalPort 11435 -ErrorAction SilentlyContinue) { Write-Output 'Optimization service already running.'; exit 0 }
$env:OLLAMA_HOST='127.0.0.1:11434'
& 'D:\CodexTools\ollama\ollama.exe' stop 'qwen3.5:2b-q4_K_M'
$nativeArgs=@('-m','D:\Models\Ollama\blobs\sha256-20cb277f0967ace47b0b5d5658e5e494a88937f7378b74b5a266496400938f4c','--host','127.0.0.1','--port','11435','-c','1024','-np','1','--device','CUDA0','-ngl','all','-t','4','-tb','4','-fa','on','-b',"$BatchSize",'-ub',"$BatchSize",'--cache-ram','128','--ctx-checkpoints','16','--checkpoint-min-step','0','--no-mmproj','--no-webui','--offline','--log-verbosity','4')
$nativeProc=Start-Process -FilePath 'D:\CodexTools\ollama\lib\ollama\llama-server.exe' -ArgumentList $nativeArgs -WorkingDirectory 'D:\CodexTools\ollama\lib\ollama' -WindowStyle Hidden -RedirectStandardOutput "$taskDir\native.stdout.log" -RedirectStandardError "$taskDir\native.stderr.log" -PassThru
$nativeProc.Id | Set-Content -LiteralPath "$taskDir\native.pid"
"batch_size=$BatchSize; checkpoints=16; full Qwen3.5-2B Q4; CUDA0; flash attention on" | Set-Content -LiteralPath "$taskDir\runtime_config.txt"
Write-Output "Started native CUDA inference PID $($nativeProc.Id)"
