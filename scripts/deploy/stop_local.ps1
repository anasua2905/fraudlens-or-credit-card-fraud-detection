# Stop the FraudLens API, dashboard and tunnels started by run_local.ps1.
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$file = Join-Path $root "data\deploy\deployment.json"
if (-not (Test-Path $file)) { Write-Host "Nothing to stop (no data\deploy\deployment.json)."; exit 0 }
$info = Get-Content $file -Raw | ConvertFrom-Json
foreach ($p in $info.pids.PSObject.Properties) {
    # /T stops the whole process tree (cmd.exe and the server it started)
    $null = cmd /c "taskkill /PID $($p.Value) /T /F 2>&1"
    if ($LASTEXITCODE -eq 0) { Write-Host "stopped $($p.Name) (pid $($p.Value))" } else { Write-Host "$($p.Name) (pid $($p.Value)) was not running" }
}
Remove-Item $file
