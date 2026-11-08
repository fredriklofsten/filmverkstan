"""Röster: 'Snabb' (Piper, robotaktig men direkt) och 'Naturlig' (Chatterbox, kan härma en röst)."""
import os
import subprocess
import time
import urllib.request
import wave

from common import PIPER_VOICE, TTS_PY, TMP, new_path, run_worker, ffmpeg

PIPER_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/sv/sv_SE/nst/medium/sv_SE-nst-medium.onnx"
ENGINES = ["Naturlig (Chatterbox)", "Snabb (Piper)"]
PIPER_STYLES = {
    "Mansröst, lugn och äldre": dict(pitch=0.93, tempo=1.07, warm=True),
    "Mansröst, normal": dict(pitch=1.0, tempo=1.0, warm=False),
    "Ljusare röst": dict(pitch=1.12, tempo=1.0, warm=False),
}
_piper = None


def piper(text, style="Mansröst, lugn och äldre", speed=1.0, out=None):
    global _piper
    from piper import PiperVoice, SynthesisConfig
    PIPER_VOICE.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("", ".json"):
        f = str(PIPER_VOICE) + ext
        if not os.path.exists(f):
            urllib.request.urlretrieve(PIPER_URL + ext, f)
    if _piper is None:
        _piper = PiperVoice.load(str(PIPER_VOICE))
    st = PIPER_STYLES[style]
    raw = str(TMP / f"piper_{time.time_ns()}.wav")
    with wave.open(raw, "wb") as w:
        _piper.synthesize_wav(text, w, syn_config=SynthesisConfig(length_scale=1.1 / speed))
    out = out or new_path("rost", "wav")
    af = f"asetrate={_piper.config.sample_rate}*{st['pitch']},aresample=44100,atempo={st['tempo']},highpass=f=70,lowpass=f=9000"
    if st["warm"]:
        af += ",aecho=0.6:0.4:40:0.12"
    af += ",loudnorm=I=-16:TP=-1.5"
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", raw, "-af", af, out], check=True)
    os.remove(raw)
    return out


def default_reference():
    """Om ingen röstinspelning laddas upp: en djup, lugn mansröst som utgångspunkt."""
    ref = str(TMP / "standard_referens.wav")
    if not os.path.exists(ref):
        piper("Jag har bott här i skogen i många år. Varje morgon går jag ut och lyssnar på fåglarna, "
              "och sedan börjar dagens arbete.", "Mansröst, lugn och äldre", out=ref)
    return ref


def ensure_tts_env(progress=None):
    ok = "/content/ttsenv/.fv_ok"
    if os.path.exists(ok):
        return
    if progress:
        progress(0.01, desc="Installerar den naturliga rösten (bara första gången, ca 3–5 min)")
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    subprocess.run(["uv", "venv", "-q", "--clear", "--python", sys.executable, "/content/ttsenv"], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "--python", TTS_PY, "chatterbox-tts", "setuptools<81"], check=True)
    open(ok, "w").write("ok")


def speak(text, engine=ENGINES[0], ref=None, style="Mansröst, lugn och äldre", speed=1.0, emotion=0.5,
          progress=None, out=None):
    out = out or new_path("rost", "wav")
    if engine.startswith("Snabb"):
        return piper(text, style, speed, out)
    ensure_tts_env(progress)
    raw = str(TMP / f"cb_{time.time_ns()}.wav")
    run_worker("tts.py", {"text": text, "ref": ref or default_reference(), "out": raw,
                          "exaggeration": emotion, "cfg": 0.5}, progress, python=TTS_PY)
    af = "loudnorm=I=-16:TP=-1.5"
    if speed != 1.0:
        af = f"atempo={speed}," + af
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", raw, "-af", af, "-ar", "44100", out], check=True)
    os.remove(raw)
    return out
