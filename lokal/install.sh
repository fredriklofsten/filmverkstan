#!/usr/bin/env bash
# Filmverkstan – installation för Linux
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
$PY -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print('Grafikkort:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'HITTADES INTE')"
echo "Klart! Starta med ./start.sh"
