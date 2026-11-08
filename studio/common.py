"""Gemensamma saker: sökvägar, körning av arbetare (workers), ffmpeg."""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STUDIO = REPO / "studio"
WORKERS = STUDIO / "workers"

DRIVE_ROOT = Path("/content/drive/MyDrive/Filmverkstan")
DRIVE_MOUNTED = Path("/content/drive/MyDrive").exists()

# Resultat sparas i Drive om den är ansluten, annars lokalt i Colab
OUT = (DRIVE_ROOT / "resultat") if DRIVE_MOUNTED else Path("/content/fv_resultat")
OUT.mkdir(parents=True, exist_ok=True)
TMP = Path("/content/fv_tmp")
TMP.mkdir(parents=True, exist_ok=True)

SADTALKER = Path("/content/SadTalker")
TTS_PY = "/content/ttsenv/bin/python"
PIPER_VOICE = Path("/content/fv_models/sv_SE-nst-medium.onnx")


def env():
    e = dict(os.environ)
    if os.environ.get("FV_DRIVE_MODELS") == "1" and DRIVE_MOUNTED:
        e["HF_HOME"] = str(DRIVE_ROOT / "modeller" / "hf")
    e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return e


def new_path(prefix, ext):
    return str(OUT / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.{ext}")


def ffmpeg():
    return shutil.which("ffmpeg") or "ffmpeg"


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(out or 0)


def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    return bool(out)


class WorkerError(RuntimeError):
    pass


def run_worker(script, args, progress=None, python=None, cwd=None):
    """Kör ett arbetsskript i en egen process (så minnet frigörs efteråt).
    Skriptet skriver '@@P <0-1> <text>' för framsteg och '@@R <sökväg>' för resultat."""
    cmd = [python or sys.executable, "-u", str(WORKERS / script), json.dumps(args)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env(), cwd=cwd)
    results, tail, buf = [], [], b""
    while True:
        chunk = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(1)
        if not chunk:
            break
        buf += chunk
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for raw in parts:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            if line.startswith("@@P "):
                try:
                    _, frac, *desc = line.split(" ")
                    if progress:
                        progress(float(frac), desc=" ".join(desc))
                except Exception:
                    pass
            elif line.startswith("@@R "):
                results.append(line[4:].strip())
            else:
                print(line, flush=True)
                if "%|" not in line and not line.startswith(("Loading", "Fetching", "Downloading")):
                    tail.append(line)
                    tail = tail[-30:]
    rc = p.wait()
    if rc != 0:
        errs = [l for l in tail if "Error" in l or "error" in l]
        msg = (errs[-1] if errs else "\n".join(tail[-6:]))[-600:]
        if rc in (-9, 137):
            msg = "Datorn i molnet fick slut på arbetsminne (RAM). Prova kortare klipp eller lägre kvalitet.\n" + msg
        elif "out of memory" in msg.lower():
            msg = "Grafikkortet fick slut på minne. Prova kortare klipp eller lägre kvalitet.\n" + msg
        raise WorkerError(msg)
    return results


# ---------- hjälp för arbetsskripten ----------
def wp(frac, text):
    print(f"@@P {frac:.3f} {text}", flush=True)


def wr(path):
    print(f"@@R {path}", flush=True)


def wargs():
    return json.loads(sys.argv[1])
