param([switch]$Cpu)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtimePath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtimePath)) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Gagal membuat environment Python.' }
}
& $runtimePath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Gagal menyiapkan pip.' }
$torchIndex = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
& $runtimePath -m pip install torch==2.10.0 torchvision==0.25.0 --index-url $torchIndex --no-cache-dir
if ($LASTEXITCODE -ne 0) { throw 'Gagal memasang runtime PyTorch.' }
& $runtimePath -m pip install -r requirements.txt --no-cache-dir
if ($LASTEXITCODE -ne 0) { throw 'Gagal memasang dependency aplikasi.' }
& $runtimePath -u scripts/download_encoders.py
if ($LASTEXITCODE -ne 0) { throw 'Gagal mengunduh atau memverifikasi encoder.' }
& $runtimePath -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Pengujian dasar belum lulus.' }
Write-Output 'Instalasi selesai. Jalankan .\run.ps1 untuk membuka aplikasi.'
