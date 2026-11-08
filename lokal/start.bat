@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
  echo Kor install.bat forst.
  pause
  exit /b 1
)
call venv\Scripts\activate.bat
echo Startar Filmverkstan... webblasaren oppnas av sig sjalv.
echo Stang det har fonstret for att avsluta programmet.
python app.py
pause
