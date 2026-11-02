# Minimal installer for Windows. Any pip downloads are explicit when this script is invoked.
param([switch]$AI)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Get-Command py -ErrorAction SilentlyContinue) -and -not (Get-Command python -ErrorAction SilentlyContinue)) {
  throw 'Python 3.11 veya 3.12 gerekli. https://www.python.org/downloads/'
}
if (-not (Test-Path '.venv\Scripts\python.exe')) {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { & py -3.11 -m venv .venv }
  } else { & python -m venv .venv }
  if ($LASTEXITCODE -ne 0) { throw 'Python sanal ortamı oluşturulamadı.' }
}
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if ($AI) {
  Write-Host 'UYARI: PyTorch/AI bağımlılıkları büyük indirmelere yol açabilir.' -ForegroundColor Yellow
  $reply = Read-Host 'Devam etmek istiyor musun? (EVET yaz)'
  if ($reply -cne 'EVET') { Write-Host 'Kurulum iptal edildi.'; exit 0 }
  & $python -m pip install -r requirements-ai.txt
} else {
  Write-Host 'Yalnızca temel panel paketleri kuruluyor. AI için ayrıca -AI anahtarını kullan.'
  & $python -m pip install -r requirements.txt
}
if ($LASTEXITCODE -ne 0) { throw 'Paket kurulumu tamamlanamadı.' }
Write-Host 'Kurulum tamamlandı. Başlatmak için BASLAT_WINDOWS.bat çalıştır.'
