"""Bättre läppsynk med LatentSync 1.5 (ByteDance, Apache-2.0).

SadTalker ger huvudrörelser och blinkningar, LatentSync ritar sedan om munnen
så att den följer ljudet tydligt. LatentSync kräver äldre paket, så den får en
egen Python 3.10-miljö (/content/lsenv) som installeras första gången (ca 5 min).
Går något fel behåller vi SadTalker-klippet, så du får alltid ett resultat.
"""
import os
import re
import subprocess
import sys

LS_DIR = "/content/LatentSync"
LS_ENV = "/content/lsenv"
LS_PY = f"{LS_ENV}/bin/python"
OK = f"{LS_ENV}/.fv_ok"
CKPT_REPO = "ByteDance/LatentSync-1.5"   # 256 px, ca 8 GB grafikminne – passar T4
SKIP = ("gradio", "lpips", "scenedetect")  # behövs inte för att skapa klipp


def _sh(cmd, **kw):
    print("  $", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, shell=isinstance(cmd, str), **kw)


def ready():
    return os.path.exists(OK)


def ensure(say=print):
    if ready():
        return
    say("Installerar bättre läppsynk (bara första gången, ca 5 min)")
    if not os.path.exists(LS_DIR):
        _sh(["git", "clone", "-q", "--depth", "1", "https://github.com/bytedance/LatentSync.git", LS_DIR])
    # T4 klarar float16 fint – originalet kräver nyare kort och skulle annars köra långsamt i float32
    inf = f"{LS_DIR}/scripts/inference.py"
    src = open(inf).read()
    src = src.replace("torch.cuda.get_device_capability()[0] > 7", "torch.cuda.get_device_capability()[0] >= 7")
    open(inf, "w").write(src)
    reqs = [l.strip() for l in open(f"{LS_DIR}/requirements.txt") if l.strip()]
    reqs = [l for l in reqs if not l.startswith(SKIP)]
    open(f"{LS_DIR}/fv_requirements.txt", "w").write("\n".join(reqs) + "\n")
    _sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    _sh(["uv", "venv", "-q", "--clear", "--python", "3.10", LS_ENV])
    # insightface byggs från källkod och behöver numpy/cython på plats först
    _sh(["uv", "pip", "install", "-q", "--python", LS_PY, "numpy==1.26.4", "cython", "setuptools", "wheel"])
    _sh(["uv", "pip", "install", "-q", "--python", LS_PY, "--index-strategy", "unsafe-best-match",
         "--no-build-isolation-package", "insightface", "-r", f"{LS_DIR}/fv_requirements.txt"])
    from huggingface_hub import hf_hub_download
    for f in ("latentsync_unet.pt", "whisper/tiny.pt"):
        hf_hub_download(CKPT_REPO, f, local_dir=f"{LS_DIR}/checkpoints")
    open(OK, "w").write("ok")


def run(video, audio, out, tmp, progress=None):
    """video (25 bilder/s) + ljud → nytt klipp med synkade läppar. progress(frac, text)."""
    cmd = [LS_PY, "-u", "-m", "scripts.inference", "--unet_config_path", "configs/unet/stage2.yaml",
           "--inference_ckpt_path", "checkpoints/latentsync_unet.pt", "--inference_steps", "20",
           "--guidance_scale", "1.5", "--enable_deepcache", "--video_path", video, "--audio_path", audio,
           "--video_out_path", out, "--temp_dir", tmp]
    e = dict(os.environ, PYTHONPATH=LS_DIR)
    p = subprocess.Popen(cmd, cwd=LS_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e)
    buf, tail = b"", []
    while True:
        chunk = p.stdout.read1(4096)
        if not chunk:
            break
        buf += chunk
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for raw in parts:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            m = re.search(r"(\d+)%\|", line)
            if m and progress:
                progress(int(m.group(1)) / 100, "Synkar läpparna")
                continue
            tail.append(line)
            tail = tail[-20:]
    if p.wait() != 0 or not os.path.exists(out):
        raise RuntimeError("\n".join(tail[-8:]))
    return out
