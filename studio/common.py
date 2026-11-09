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
KAGGLE = Path("/kaggle/working").exists()
if os.environ.get("FV_OUT"):
    OUT = Path(os.environ["FV_OUT"])
elif DRIVE_MOUNTED:
    OUT = DRIVE_ROOT / "resultat"
elif KAGGLE:
    OUT = Path("/kaggle/working/resultat")  # syns under Output i Kaggle och kan laddas ner
else:
    OUT = Path("/content/fv_resultat")
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


# Anropas innan ett arbetsskript startar (t.ex. för att stänga en modell som ligger kvar på grafikkortet)
BEFORE_WORKER = []


def iter_lines(stream):
    """Läser rader från en process (även förloppsrader som slutar med \\r)."""
    buf = b""
    while True:
        chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(1)
        if not chunk:
            break
        buf += chunk
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for raw in parts:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                yield line
    if buf.strip():
        yield buf.decode("utf-8", "replace").strip()


def run_worker(script, args, progress=None, python=None, cwd=None):
    """Kör ett arbetsskript i en egen process (så minnet frigörs efteråt).
    Skriptet skriver '@@P <0-1> <text>' för framsteg och '@@R <sökväg>' för resultat."""
    for hook in BEFORE_WORKER:
        try:
            hook(script)
        except Exception as e:
            print("Förberedelse inför", script, "misslyckades:", e, flush=True)
    cmd = [python or sys.executable, "-u", str(WORKERS / script), json.dumps(args)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env(), cwd=cwd)
    results, tail = [], []
    for line in iter_lines(p.stdout):
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


# ---------- efterbehandling av rörliga scener (LTX och Wan) ----------
def finish_scene(raw, smooth=True, sfx_prompt=None, progress=None):
    """Mjukare rörelser (60 bilder/s) + skärpa, och ev. ljudeffekter. Returnerar slutlig sökväg."""
    say = (lambda t: progress(0.97, t)) if progress else print
    path = raw
    if smooth:
        say("Mjukar upp rörelsen och skärper bilden")
        out = new_path("scen", "mp4")
        vf = ("minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
              "scale=trunc(iw*1.5/2)*2:trunc(ih*1.5/2)*2:flags=lanczos,unsharp=5:5:0.5")
        r = subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", raw, "-vf", vf, "-an", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", out])
        if r.returncode == 0:
            os.remove(raw)
            path = out
    if sfx_prompt is not None:
        import sfx
        out = new_path("scen_ljud", "mp4")
        res = sfx.add(path, sfx_prompt, out, say)
        if res == out:
            os.remove(path)
            path = out
    return path
