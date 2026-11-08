"""Porträtt + ljud → pratande person.
SadTalker ger huvudrörelser och blinkningar (+ GFPGAN-skärpa). Med lipsync="latentsync" ritas munnen
sedan om av LatentSync för tydligare läppsynk; misslyckas det behålls SadTalker-klippet."""
import sys, os, re, subprocess, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wargs, wp, wr, new_path, SADTALKER, TMP, ffmpeg  # noqa
import lipsync  # noqa

a = wargs()
wav = str(TMP / f"talk_{int(time.time())}.wav")
subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", a["audio"], "-ar", "16000", "-ac", "1", wav], check=True)

enhance = bool(a.get("enhance", True))
if enhance:
    try:
        import gfpgan  # noqa
    except Exception:
        enhance = False
        print("Ansiktsförbättring saknas – kör utan.")

size = 256 if a.get("fast") else 512
res_dir = str(TMP / f"talk_out_{int(time.time())}")
cmd = [sys.executable, "inference.py", "--driven_audio", wav, "--source_image", a["image"], "--result_dir", res_dir,
       "--size", str(size), "--preprocess", "full", "--still", "--expression_scale", str(a.get("expression", 1.5))]
if enhance:
    cmd += ["--enhancer", "gfpgan"]

STAGES = [("3DMM", 0.0, 0.05), ("mel", 0.05, 0.08), ("audio2exp", 0.08, 0.1), ("Face Renderer", 0.1, 0.7),
          ("Face Enhancer", 0.7, 0.88), ("seamlessClone", 0.88 if enhance else 0.7, 0.99)]
NAMES = {"Face Renderer": "Animerar ansiktet", "Face Enhancer": "Skärper ansiktet",
         "seamlessClone": "Klistrar in i bilden", "3DMM": "Analyserar ansiktet"}
LS = a.get("lipsync") == "latentsync"
SPAN = 0.45 if LS else 1.0  # hur stor del av förloppet SadTalker får
_wp = wp
def wp(frac, text):  # noqa: F811
    _wp(frac * SPAN, text)


wp(0.01, "Analyserar ansiktet")
p = subprocess.Popen(cmd, cwd=str(SADTALKER), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
buf, tail, result = b"", [], None
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
        m = re.match(r"([A-Za-z0-9 ]+?)::?\s*(\d+)%", line)
        if m:
            for key, lo, hi in STAGES:
                if line.startswith(key):
                    wp(lo + (hi - lo) * int(m.group(2)) / 100, NAMES.get(key, key))
            continue
        mm = re.search(r"generated video is named:?\s*(\S+\.mp4)", line)
        if mm:
            result = mm.group(1)
        tail.append(line)
        tail = tail[-25:]
rc = p.wait()
if rc != 0:
    print("\n".join(tail))
    sys.exit(rc if rc > 0 else 1)
if not result or not os.path.exists(result):
    found = sorted(glob.glob(res_dir + "/*.mp4"), key=os.path.getmtime)
    result = found[-1] if found else None
if not result:
    print("\n".join(tail)); sys.exit(1)
if LS:
    try:
        lipsync.ensure(lambda t: _wp(SPAN, t))
        ls_out = str(TMP / f"ls_{int(time.time())}.mp4")
        ls_tmp = str(TMP / f"ls_tmp_{int(time.time())}")
        base = str(TMP / f"ls_in_{int(time.time())}.mp4")  # LatentSync vill ha 25 bilder/s
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", result, "-r", "25", "-c:v", "libx264",
                        "-crf", "16", "-pix_fmt", "yuv420p", "-an", base], check=True)
        result = lipsync.run(base, wav, ls_out, ls_tmp, lambda f, t: _wp(SPAN + (0.97 - SPAN) * f, t))
    except Exception as e:
        print("LatentSync misslyckades – behåller SadTalker-klippet:", e, flush=True)
        _wp(0.97, "Läppsynken misslyckades – använder vanliga pratklippet")
path = new_path("prat", "mp4")
subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", result, "-i", a["audio"], "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-vf", "scale='min(1080,iw)':-2",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", path], check=True)
wr(path)
_wp(1.0, "Klar")
