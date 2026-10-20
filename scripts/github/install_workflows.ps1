# Install the FraudLens GitHub files (Actions workflows, issue and pull-request templates) from
# deploy\github into .github, then commit and push. Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts\github\install_workflows.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
New-Item -ItemType Directory -Force .github | Out-Null
Copy-Item deploy\github\workflows, deploy\github\ISSUE_TEMPLATE -Destination .github -Recurse -Force
Copy-Item deploy\github\pull_request_template.md .github\ -Force
git add .github
git commit -m "Install CI/CD workflows and issue/PR templates"
git push
Write-Host "Pushed. Open the Actions tab on GitHub to watch the first runs." -ForegroundColor Green
