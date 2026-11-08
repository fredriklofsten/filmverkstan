@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo  ===== Filmverkstan - installation =====
echo  Det här tar 5-15 minuter första gången.
echo.

set PY=
for %%V in (3.12 3.11 3.13 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set PY=py -%%V
  )
)
if not defined PY (
  echo  Python hittades inte.
  echo  Installera Python 3.12 fran https://www.python.org/downloads/
  echo  Bocka i "Add python.exe to PATH" under installationen och kor sedan install.bat igen.
  pause
  exit /b 1
)

echo  Anvander %PY%
%PY% -m venv venv || (echo Kunde inte skapa venv & pause & exit /b 1)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo.
echo  Installerar PyTorch for Nvidia (stor fil, ca 3 GB)...
pip install torch --index-url https://download.pytorch.org/whl/cu128 || (echo PyTorch-installationen misslyckades & pause & exit /b 1)
echo.
echo  Installerar resten...
pip install -r requirements.txt || (echo Installationen misslyckades & pause & exit /b 1)
echo.
python -c "import torch; print('  Grafikkort:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'HITTADES INTE - uppdatera Nvidia-drivrutinen')"
echo.
echo  Klart! Starta programmet med start.bat
pause
