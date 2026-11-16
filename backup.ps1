# One-command backup: stage new raw data, commit, push.
# Usage (from the project folder):  powershell -ExecutionPolicy Bypass -File backup.ps1
# Only touches data/ so half-finished code is never committed by accident.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

git add data
git diff --cached --quiet -- data
if ($LASTEXITCODE -eq 0) {
    Write-Host "Nothing new to back up."
    exit 0
}

$n = (git diff --cached --name-only -- data | Measure-Object -Line).Lines
$stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mmZ")
git commit --quiet -m "data: backup $stamp ($n files)" -- data
if ($LASTEXITCODE -ne 0) { throw "git commit failed" }

git push --quiet
if ($LASTEXITCODE -ne 0) { throw "git push failed - data is committed locally but NOT on GitHub" }

Write-Host "Backed up $n files at $stamp."
