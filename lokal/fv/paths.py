"""Var allt sparas. Allt ligger i mappen bredvid programmet."""
from pathlib import Path
import datetime

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "resultat"
MODELS = ROOT / "modeller"
VOICES = MODELS / "roster"

for p in (OUTPUT, MODELS, VOICES):
    p.mkdir(parents=True, exist_ok=True)


def new_file(prefix: str, ext: str) -> Path:
    """Skapar ett unikt filnamn i resultat-mappen, t.ex. video_2026-11-07_19-30-12.mp4"""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = OUTPUT / f"{prefix}_{stamp}.{ext}"
    n = 2
    while path.exists():
        path = OUTPUT / f"{prefix}_{stamp}_{n}.{ext}"
        n += 1
    return path
