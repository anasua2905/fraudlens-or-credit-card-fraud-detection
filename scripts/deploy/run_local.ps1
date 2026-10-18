<#
Deploy FraudLens on this Windows machine: the scoring API (FastAPI) and the dashboard (Streamlit),
both from the project's virtual environment. With -Public, each is also published through a
Cloudflare quick tunnel (a temporary https://*.trycloudflare.com address, no account needed).

    powershell -ExecutionPolicy Bypass -File scripts\deploy\run_local.ps1            # local only
    powershell -ExecutionPolicy Bypass -File scripts\deploy\run_local.ps1 -Public    # + public links
    powershell -ExecutionPolicy Bypass -File scripts\deploy\stop_local.ps1           # stop everything

Logs and the running state are written to data\deploy\ (not committed).
#>
param([switch]$Public, [int]$ApiPort = 8000, [int]$DashboardPort = 8501)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No virtual environment at $py. Create it first (see README)." }
$state = Join-Path $root "data\deploy"
New-Item -ItemType Directory -Force $state | Out-Null

function Start-Bg($name, $file, $arguments) {
    # Started through WMI so the process outlives this script (and any job that runs it);
    # cmd.exe redirects the output to data\deploy\<name>.log.
    $log = Join-Path $state "$name.log"
    $cmd = "cmd.exe /c `"set PYTHONUTF8=1&& set MLFLOW_DISABLE_AGENT_HINT=1&& `"$file`" $arguments > `"$log`" 2>&1`""
    $hidden = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
        -Arguments @{ CommandLine = $cmd; CurrentDirectory = $root; ProcessStartupInformation = $hidden }
    if ($r.ReturnValue -ne 0) { throw "could not start $name (WMI code $($r.ReturnValue))" }
    Write-Host "started $name (pid $($r.ProcessId))"
    return $r.ProcessId
}
function Wait-Http($url, $seconds = 180) {
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalSeconds -lt $seconds) {
        try { if ((Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 5).StatusCode -eq 200) { return $true } } catch { }
        Start-Sleep -Seconds 3
    }
    return $false
}

$pids = @{}
$pids.api = Start-Bg "api" $py "-m uvicorn fraudlens.api:app --host 127.0.0.1 --port $ApiPort"
$pids.dashboard = Start-Bg "dashboard" $py "-m streamlit run dashboard/app.py --server.port $DashboardPort --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false"
$apiOk = Wait-Http "http://127.0.0.1:$ApiPort/health"
$dashOk = Wait-Http "http://127.0.0.1:$DashboardPort/_stcore/health"
$info = [ordered]@{ started = (Get-Date -Format o); api_local = "http://127.0.0.1:$ApiPort/docs"; api_healthy = $apiOk
                    dashboard_local = "http://127.0.0.1:$DashboardPort"; dashboard_healthy = $dashOk }

if ($Public) {
    $cf = Join-Path $state "cloudflared.exe"
    if (-not (Test-Path $cf)) {
        Write-Host "downloading cloudflared (Cloudflare's tunnel client)"
        Invoke-WebRequest -UseBasicParsing -OutFile $cf `
            -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    }
    foreach ($svc in @(@{n = "tunnel_dashboard"; port = $DashboardPort}, @{n = "tunnel_api"; port = $ApiPort})) {
        $pids[$svc.n] = Start-Bg $svc.n $cf "tunnel --no-autoupdate --url http://127.0.0.1:$($svc.port)"
    }
    foreach ($n in @("tunnel_dashboard", "tunnel_api")) {
        $url = $null
        for ($i = 0; $i -lt 40 -and -not $url; $i++) {
            Start-Sleep -Seconds 3
            $m = Select-String -Path (Join-Path $state "$n.log") -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($m) { $url = $m.Matches[0].Value }
        }
        $info["${n}_url"] = $url
    }
    if ($info["tunnel_api_url"]) { $info["api_public_docs"] = $info["tunnel_api_url"] + "/docs" }
}
$info.pids = $pids
$info | ConvertTo-Json | Set-Content -Path (Join-Path $state "deployment.json") -Encoding ascii
$info | ConvertTo-Json
