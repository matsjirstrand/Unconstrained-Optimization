$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$localPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if (Test-Path $localPython) {
    & $localPython app.py
} elseif (Test-Path $bundledPython) {
    & $bundledPython app.py
} else {
    python app.py
}
