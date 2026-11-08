"""Installerar allt som Filmverkstan Studio behöver i Colab. Kan köras flera gånger (hoppar över det som redan finns)."""
import glob
import os
import re
import subprocess
import sys
import tarfile
import urllib.request
import json


def sh(cmd, check=True):
    print("  $", cmd[:110], flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def pip(*pkgs, extra=""):
    sh(f"{sys.executable} -m pip install -q {extra} " + " ".join(pkgs))


print("1/5 Installerar paket…", flush=True)
pip('"diffusers>=0.35"', "transformers", "accelerate", "sentencepiece", "protobuf", "imageio", "imageio-ffmpeg",
    '"gradio>=5"', "piper-tts", "facexlib", "kornia", "yacs", "librosa", "pydub", '"audioop-lts; python_version>=\'3.13\'"', "safetensors",
    "scikit-image", "filterpy", "addict", "lmdb", "yapf", "future")

# ---------- SadTalker ----------
ST = "/content/SadTalker"
print("2/5 SadTalker (pratande ansikten)…", flush=True)
if not os.path.exists(ST):
    sh(f"git clone -q --depth 1 https://github.com/OpenTalker/SadTalker.git {ST}")
GH = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/"
FX = "https://github.com/xinntao/facexlib/releases/download/"
files = {
    "checkpoints/mapping_00109-model.pth.tar": GH + "mapping_00109-model.pth.tar",
    "checkpoints/mapping_00229-model.pth.tar": GH + "mapping_00229-model.pth.tar",
    "checkpoints/SadTalker_V0.0.2_512.safetensors": GH + "SadTalker_V0.0.2_512.safetensors",
    "checkpoints/SadTalker_V0.0.2_256.safetensors": GH + "SadTalker_V0.0.2_256.safetensors",
    "gfpgan/weights/alignment_WFLW_4HG.pth": FX + "v0.1.0/alignment_WFLW_4HG.pth",
    "gfpgan/weights/detection_Resnet50_Final.pth": FX + "v0.1.0/detection_Resnet50_Final.pth",
    "gfpgan/weights/parsing_parsenet.pth": FX + "v0.2.2/parsing_parsenet.pth",
    "gfpgan/weights/GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
}
for rel, url in files.items():
    p = os.path.join(ST, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p):
        print("  hämtar", rel, flush=True)
        urllib.request.urlretrieve(url, p + ".part")
        os.rename(p + ".part", p)


def patch(rel, old, new):
    p = os.path.join(ST, rel)
    s = open(p).read()
    if new in s:
        return
    if old not in s:
        print("  (hittade inte rättningsstället i", rel, ")")
        return
    open(p, "w").write(s.replace(old, new))


# Rättningar så att SadTalker (2023) fungerar med nya Python/numpy
patch("src/utils/face_enhancer.py", "from gfpgan import GFPGANer",
      "try:\n    from gfpgan import GFPGANer\nexcept Exception:\n    GFPGANer = None")
patch("src/face3d/util/preprocess.py", 'warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)',
      'warnings.filterwarnings("ignore")')
patch("src/face3d/util/preprocess.py", "    s = rescale_factor/s\n",
      "    s = rescale_factor/s\n    t = np.asarray(t).reshape(-1); s = np.float64(np.squeeze(s))\n")
patch("src/face3d/util/preprocess.py", "trans_params = np.array([w0, h0, s, t[0], t[1]])",
      "trans_params = np.array([float(w0), float(h0), float(np.squeeze(s)), float(np.squeeze(t[0])), float(np.squeeze(t[1]))])")
patch("src/utils/preprocess.py", "[float(item) for item in np.hsplit(trans_params, 5)]",
      "[float(np.squeeze(item)) for item in np.hsplit(np.asarray(trans_params, dtype=float).reshape(-1), 5)]")
patch("src/face3d/util/my_awing_arch.py", "astype(np.float,", "astype(float,")

# ---------- Ansiktsförbättring (GFPGAN) ----------
print("3/5 Ansiktsförbättring (GFPGAN)…", flush=True)
try:
    import gfpgan  # noqa
    print("  redan installerad")
except Exception:
    try:
        meta = json.load(urllib.request.urlopen("https://pypi.org/pypi/basicsr/1.4.2/json"))
        url = [u["url"] for u in meta["urls"] if u["packagetype"] == "sdist"][0]
        urllib.request.urlretrieve(url, "/tmp/basicsr.tar.gz")
        import shutil
        shutil.rmtree("/tmp/bsr", ignore_errors=True)
        tarfile.open("/tmp/basicsr.tar.gz").extractall("/tmp/bsr")
        d = glob.glob("/tmp/bsr/basicsr-*")[0]
        s = open(f"{d}/setup.py").read()
        s = re.sub(r"def get_version\(\):.*?return locals\(\)\['__version__'\]", "def get_version():\n    return '1.4.2'",
                   s, flags=re.S)
        open(f"{d}/setup.py", "w").write(s)
        deg = f"{d}/basicsr/data/degradations.py"
        txt = open(deg).read().replace("torchvision.transforms.functional_tensor", "torchvision.transforms.functional")
        open(deg, "w").write(txt)
        sh(f"{sys.executable} -m pip install -q --force-reinstall --no-deps --no-build-isolation {d}")
        pip("gfpgan", extra="--no-deps")
        print("  klar")
    except Exception as e:
        print("  kunde inte installeras – pratklipp fungerar ändå, men utan skärpning:", e)

# ---------- Svensk röst (snabb) ----------
print("4/5 Svensk röst…", flush=True)
os.makedirs("/content/fv_models", exist_ok=True)
base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/sv/sv_SE/nst/medium/sv_SE-nst-medium.onnx"
for ext in ("", ".json"):
    f = "/content/fv_models/sv_SE-nst-medium.onnx" + ext
    if not os.path.exists(f):
        urllib.request.urlretrieve(base + ext, f)

print("5/5 Klart! ✅", flush=True)
