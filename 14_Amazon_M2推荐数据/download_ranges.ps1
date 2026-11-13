$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$m2Raw=Join-Path $PSScriptRoot 'data\raw'
$m2Part=Join-Path $m2Raw 'amazon-m2-v1.zip.part'
$m2Zip=Join-Path $m2Raw 'amazon-m2-v1.zip'
$m2Uri='https://www.kaggle.com/api/v1/datasets/download/marquis03/amazon-m2?datasetVersionNumber=1'
$expectedBytes=417556865L
if(Test-Path -LiteralPath $m2Zip){throw 'Archive already exists; inspect it before downloading again'}
$planPath=Join-Path $m2Raw 'download_ranges.json'
if(Test-Path -LiteralPath $planPath){
 $plan=Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
}else{
 if(-not (Test-Path -LiteralPath $m2Part)){[System.IO.File]::WriteAllBytes($m2Part,[byte[]]@())}
 $prefix=(Get-Item -LiteralPath $m2Part).Length
 if($prefix -gt $expectedBytes){throw 'Unexpected partial size'}
 $width=[long][math]::Ceiling(($expectedBytes-$prefix)/12.0)
 $chunks=@()
 for($i=0;$i -lt 12;$i++){
  $start=$prefix+$i*$width
  if($start -ge $expectedBytes){break}
  $end=[math]::Min($expectedBytes-1,$start+$width-1)
  $chunks+=@{Index=$i;Start=$start;End=$end;Path=(Join-Path $m2Raw ('range_{0:D2}.part' -f $i))}
 }
 $plan=@{PrefixBytes=$prefix;ExpectedBytes=$expectedBytes;Chunks=$chunks}
 $plan | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $planPath -Encoding utf8
}
if((Get-Item -LiteralPath $m2Part).Length -ne $plan.PrefixBytes){throw 'Partial prefix changed after plan creation'}
$outcomes=@($plan.Chunks | ForEach-Object -ThrottleLimit 12 -Parallel {
 $chunk=$_
 $expected=[long]$chunk.End-[long]$chunk.Start+1
 for($attempt=1;$attempt -le 3;$attempt++){
  $already=if(Test-Path -LiteralPath $chunk.Path){(Get-Item -LiteralPath $chunk.Path).Length}else{0L}
  if($already -eq $expected){[pscustomobject]@{Index=$chunk.Index;Complete=$true;Bytes=$already};return}
  if($already -gt $expected){throw 'Oversized chunk'}
  $http=[System.Net.Http.HttpClient]::new()
  $http.Timeout=[TimeSpan]::FromMinutes(5)
  $response=$null;$file=$null;$cancel=[System.Threading.CancellationTokenSource]::new([TimeSpan]::FromMinutes(5))
  try{
   $request=[System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Get,$using:m2Uri)
   $start=[long]$chunk.Start+$already
   $request.Headers.Range=[System.Net.Http.Headers.RangeHeaderValue]::new($start,[long]$chunk.End)
   $response=$http.SendAsync($request,[System.Net.Http.HttpCompletionOption]::ResponseHeadersRead,$cancel.Token).GetAwaiter().GetResult()
   if([int]$response.StatusCode -ne 206){throw ('Range status '+[int]$response.StatusCode)}
   if($response.Content.Headers.ContentRange.From -ne $start -or $response.Content.Headers.ContentRange.To -ne $chunk.End){throw 'Range mismatch'}
   $file=[System.IO.File]::Open($chunk.Path,[System.IO.FileMode]::Append,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
   [void]$response.Content.CopyToAsync($file,$cancel.Token).GetAwaiter().GetResult()
   $file.Dispose();$file=$null
   if((Get-Item -LiteralPath $chunk.Path).Length -ne $expected){throw 'Truncated chunk'}
   [pscustomobject]@{Index=$chunk.Index;Complete=$true;Bytes=$expected};return
  }catch{
   if($attempt -eq 3){[pscustomobject]@{Index=$chunk.Index;Complete=$false;Error=$_.Exception.Message}}
  }finally{
   if($null -ne $file){$file.Dispose()}
   if($null -ne $response){$response.Dispose()}
   $cancel.Dispose();$http.Dispose()
  }
 }
})
$outcomes | Sort-Object Index | ConvertTo-Json -Compress
if(@($outcomes | Where-Object {-not $_.Complete}).Count -or $outcomes.Count -ne $plan.Chunks.Count){throw 'Some ranges remain incomplete'}
$target=[System.IO.File]::Open($m2Part,[System.IO.FileMode]::Append,[System.IO.FileAccess]::Write)
try{
 foreach($chunk in ($plan.Chunks | Sort-Object Index)){
  $inputStream=[System.IO.File]::OpenRead($chunk.Path)
  try{$inputStream.CopyTo($target)}finally{$inputStream.Dispose()}
 }
}finally{$target.Dispose()}
if((Get-Item -LiteralPath $m2Part).Length -ne $expectedBytes){throw 'Combined archive size mismatch'}
Rename-Item -LiteralPath $m2Part -NewName 'amazon-m2-v1.zip'
Get-Item -LiteralPath $m2Zip | Select-Object Name,Length | ConvertTo-Json -Compress



