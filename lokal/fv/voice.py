"""Text → tal på svenska med Piper (körs lokalt, gratis)."""
import subprocess
import urllib.request
import wave

from . import paths, tools

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
VOICES = {
    "Mansröst (lugn, äldre)": dict(file="sv/sv_SE/nst/medium/sv_SE-nst-medium", pitch=0.93, tempo=1.13, warm=True),
    "Mansröst (normal)": dict(file="sv/sv_SE/nst/medium/sv_SE-nst-medium", pitch=1.0, tempo=1.0, warm=False),
    "Ljusare röst": dict(file="sv/sv_SE/nst/medium/sv_SE-nst-medium", pitch=1.12, tempo=1.0, warm=False),
}


def _model(rel: str):
    name = rel.split("/")[-1]
    onnx = paths.VOICES / f"{name}.onnx"
    js = paths.VOICES / f"{name}.onnx.json"
    for f, url in ((onnx, f"{BASE}/{rel}.onnx"), (js, f"{BASE}/{rel}.onnx.json")):
        if not f.exists():
            tmp = f.with_suffix(f.suffix + ".tmp")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(f)
    return onnx


_loaded = {}


def speak(text: str, voice: str = "Mansröst (lugn, äldre)", speed: float = 1.0, out_path=None) -> str:
    from piper import PiperVoice, SynthesisConfig
    v = VOICES[voice]
    onnx = _model(v["file"])
    if str(onnx) not in _loaded:
        _loaded[str(onnx)] = PiperVoice.load(str(onnx))
    pv = _loaded[str(onnx)]
    raw = paths.new_file("rost_ra", "wav")
    with wave.open(str(raw), "wb") as wf:
        pv.synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=1.12 / speed))
    sr = pv.config.sample_rate
    out = out_path or paths.new_file("rost", "wav")
    af = f"asetrate={sr}*{v['pitch']},aresample=44100,atempo={v['tempo']:.3f},highpass=f=70,lowpass=f=9000"
    if v["warm"]:
        af += ",aecho=0.6:0.4:40:0.12"
    subprocess.run([tools.ffmpeg(), "-y", "-loglevel", "error", "-i", str(raw), "-af", af, str(out)], check=True)
    raw.unlink(missing_ok=True)
    return str(out)
