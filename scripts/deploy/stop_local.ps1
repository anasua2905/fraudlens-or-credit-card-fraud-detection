# Stop the FraudLens API, dashboard and tunnels started by run_local.ps1.
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$file = Join-Path $root "data\deploy\deployment.json"
if (Test-Path $file) {
    $info = Get-Content $file -Raw | ConvertFrom-Json
    foreach ($p in $info.pids.PSObject.Properties) {
        # /T stops the whole process tree (cmd.exe and the server it started)
        $null = cmd /c "taskkill /PID $($p.Value) /T /F 2>&1"
        if ($LASTEXITCODE -eq 0) { Write-Host "stopped $($p.Name) (pid $($p.Value))" } else { Write-Host "$($p.Name) (pid $($p.Value)) was not running" }
    }
    Remove-Item $file
}
# Leftovers whose parent cmd.exe has already gone (e.g. after a sleep or a crashed wrapper):
# match on the command line so nothing else on the machine is touched.
$patterns = @("*fraudlens.api:app*", "*streamlit*dashboard/app.py*", "*data\deploy\cloudflared.exe*")
Get-CimInstance Win32_Process | Where-Object {
    $cl = $_.CommandLine; $cl -and ($patterns | Where-Object { $cl -like $_ })
} | ForEach-Object {
    $null = cmd /c "taskkill /PID $($_.ProcessId) /T /F 2>&1"
    Write-Host "stopped leftover $($_.Name) (pid $($_.ProcessId))"
}
