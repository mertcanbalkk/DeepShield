@echo off
chcp 65001 > nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Sanal ortam bulunamadi. Once py -3.12 -m venv .venv calistirin.
  pause
  exit /b 1
)
if not exist "datasets\ai_origin\train\ai" (
  echo Veri seti eksik! README dosyasindaki klasor yapisini hazirlayin.
  pause
  exit /b 1
)
echo Veri seti denetleniyor...
.venv\Scripts\python.exe -m training.dataset_quality --data datasets\ai_origin
if errorlevel 1 (echo Denetim basarisiz. Kopya/bozuk veri kontrolu yapin. & pause & exit /b 1)
echo Egitim basliyor...
.venv\Scripts\python.exe training\train_classifier.py --data datasets\ai_origin --out outputs\deepshield_ai_b0.pt --epochs 8 --batch-size 16
pause
