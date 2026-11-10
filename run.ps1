param(
    [int]$Port = 8501,
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtimePath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtimePath)) {
    throw 'Environment belum tersedia. Ikuti instalasi pada README.md.'
}
$env:INDOMEME_DEVICE = $Device
& $runtimePath -m streamlit run app.py --server.port $Port --server.address 127.0.0.1 --server.headless true
exit $LASTEXITCODE
