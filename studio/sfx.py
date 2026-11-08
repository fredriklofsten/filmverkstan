"""Ljudeffekter som passar bilden, med MMAudio (Sony AI m.fl., MIT-licens).

MMAudio tittar på klippet och skapar miljöljud och effekter (fåglar, steg, vind, vatten …)
som följer det som händer i bilden. Den har en egen Python-miljö (/content/mmenv) som
installeras första gången (ca 5 min). Misslyckas något behålls klippet utan ljud.
"""
import os
import subprocess
import sys
import time

from common import TMP, ffmpeg

MM_DIR = "/content/MMAudio"
MM_ENV = "/content/mmenv"
MM_PY = f"{MM_ENV}/bin/python"
OK = f"{MM_ENV}/.fv_ok"
NEG = "music, speech, talking, voice, singing, narration"


def _sh(cmd):
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure(say=print):
    if os.path.exists(OK):
        return
    say("Installerar ljudeffekter (bara första gången, ca 5 min)")
    if not os.path.exists(MM_DIR):
        _sh(["git", "clone", "-q", "--depth", "1", "https://github.com/hkchengrex/MMAudio.git", MM_DIR])
    _sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    _sh(["uv", "venv", "-q", "--clear", "--python", "3.11", MM_ENV])
    _sh(["uv", "pip", "install", "-q", "--python", MM_PY, "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0"])
    _sh(["uv", "pip", "install", "-q", "--python", MM_PY, "-e", MM_DIR])
    open(OK, "w").write("ok")


def add(video, prompt, out, say=print):
    """Lägger till ljudeffekter i ett (ljudlöst) klipp. Returnerar ny sökväg, eller originalet vid fel."""
    try:
        ensure(say)
        say("Skapar ljudeffekter som passar bilden")
        d = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                           capture_output=True, text=True).stdout.strip()
        dur = min(12.0, float(d or 8))
        odir = str(TMP / f"mm_{time.time_ns()}")
        cmd = [MM_PY, "demo.py", "--variant", "large_44k_v2", "--video", video, "--duration", f"{dur:.2f}",
               "--prompt", prompt or "", "--negative_prompt", NEG, "--full_precision",
               "--skip_video_composite", "--output", odir]
        r = subprocess.run(cmd, cwd=MM_DIR, capture_output=True, text=True)
        flac = os.path.join(odir, os.path.splitext(os.path.basename(video))[0] + ".flac")
        if r.returncode != 0 or not os.path.exists(flac):
            raise RuntimeError((r.stdout + r.stderr)[-800:])
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", video, "-i", flac, "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                       check=True)
        return out
    except Exception as e:
        print("Ljudeffekterna misslyckades – klippet blir utan ljud:", e, flush=True)
        return video
