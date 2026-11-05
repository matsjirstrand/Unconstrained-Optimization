$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$localPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path $localPython)) {
    throw 'Create the project environment first: py -3.15 -m venv .venv'
}
& $localPython -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 15) else 1)"
if ($LASTEXITCODE -ne 0) { throw 'This project requires Python 3.15 in .venv.' }
& $localPython app.py
exit $LASTEXITCODE
