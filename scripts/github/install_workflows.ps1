# Install the FraudLens GitHub Actions workflows into .github/workflows, commit and push.
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File scripts\github\install_workflows.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
New-Item -ItemType Directory -Force .github\workflows | Out-Null
Copy-Item deploy\github-workflows\*.yml .github\workflows\ -Force
git add .github/workflows
git commit -m "Install CI/CD workflows (tests, platform end to end, deploy)"
git push
Write-Host "Workflows pushed. See the Actions tab on GitHub." -ForegroundColor Green
