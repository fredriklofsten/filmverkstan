"""AI-musik med MusicGen (Meta, via transformers).

Skapar instrumental musik från en beskrivning på engelska. Modellen gör max 30 sekunder åt
gången, så längre låtar byggs i delar: varje ny del får lyssna på de sista 10 sekunderna och
fortsätter därifrån, så det låter som en sammanhängande låt.

Argument (JSON): prompt, seconds, size (small|medium|large), seed, guidance, out
Skriver en mono-WAV (32 kHz) och '@@R <sökväg>'.
"""
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wargs, wp, wr, new_path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoProcessor, MusicgenForConditionalGeneration  # noqa: E402
from transformers.generation.streamers import BaseStreamer  # noqa: E402

MODELS = {"small": "facebook/musicgen-small", "medium": "facebook/musicgen-medium", "large": "facebook/musicgen-large"}
GB = {"small": "1", "medium": "3,5", "large": "7"}
CHUNK = float(os.environ.get("FV_MG_CHUNK", 30))     # sekunder per del (modellens gräns)
OVERLAP = float(os.environ.get("FV_MG_OVERLAP", 10))  # så mycket av slutet som nästa del lyssnar på

a = wargs()
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
size = a.get("size", "medium") if a.get("size") in MODELS else "medium"
total = max(3.0, min(300.0, float(a.get("seconds", 30))))
prompt = (a.get("prompt") or "calm instrumental music").strip()
guidance = float(a.get("guidance", 3.0))
seed = int(a.get("seed", -1))
if seed < 0:
    seed = int.from_bytes(os.urandom(3), "little")
torch.manual_seed(seed)

parts = 1 if total <= CHUNK else 1 + int(np.ceil((total - CHUNK) / (CHUNK - OVERLAP)))


class Progress(BaseStreamer):
    """Räknar genererade steg så att studion kan visa hur långt det har kommit."""

    def __init__(self, lo, hi, steps, text):
        self.lo, self.hi, self.steps, self.text, self.n = lo, hi, max(1, steps), text, 0

    def put(self, value):
        self.n += 1
        if self.n % 25 == 0:
            wp(self.lo + (self.hi - self.lo) * min(1.0, self.n / self.steps), self.text)

    def end(self):
        pass


def load(dtype):
    wp(0.02, f"Laddar musikmodellen ({size}, första gången ca {GB[size]} GB)")
    proc = AutoProcessor.from_pretrained(MODELS[size])
    model = MusicgenForConditionalGeneration.from_pretrained(MODELS[size], torch_dtype=dtype).to(DEV).eval()
    return proc, model


def generate(proc, model, seconds, context, lo, hi, label):
    """Gör `seconds` ny musik. Med `context` (ljud) fortsätter den därifrån; svaret börjar då med context."""
    sr = model.config.audio_encoder.sampling_rate
    fr = model.config.audio_encoder.frame_rate
    kw = dict(text=[prompt], padding=True, return_tensors="pt")
    if context is not None:
        kw.update(audio=[context], sampling_rate=sr)
    inputs = proc(**kw)
    inputs = {k: (v.to(DEV, model.dtype) if v.is_floating_point() else v.to(DEV)) for k, v in inputs.items()}
    steps = int(np.ceil(seconds * fr)) + 8   # lite extra: ljudkodaren tappar några steg i slutet
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=True, guidance_scale=guidance, max_new_tokens=steps,
                             streamer=Progress(lo, hi, steps, label))
    audio = out[0, 0].float().cpu().numpy()
    if not np.isfinite(audio).all() or np.abs(audio).max() < 1e-4:
        raise FloatingPointError("tomt eller ogiltigt ljud")
    return audio


def make(proc, model):
    sr = model.config.audio_encoder.sampling_rate
    first = min(total, CHUNK)
    span = 0.9 / parts
    audio = generate(proc, model, first, None, 0.08, 0.08 + span, f"Komponerar del 1 av {parts}")
    audio = audio[: int(first * sr)]
    k = 1
    xf = int(0.08 * sr)
    while len(audio) / sr < total - 0.25 and k < parts + 2:
        k += 1
        step = min(CHUNK - OVERLAP, total - len(audio) / sr)
        ctx_len = int(OVERLAP * sr)
        ctx = audio[-ctx_len:].astype(np.float32)
        lo = 0.08 + span * (k - 1)
        new = generate(proc, model, step, ctx, lo, lo + span, f"Komponerar del {k} av {parts}")
        if len(new) < ctx_len + int(0.25 * sr):
            print("Fortsättningen blev för kort – avslutar låten här", flush=True)
            break
        # `new` börjar med (en kopia av) de sista 10 sekunderna – skarva mjukt där kopian börjar
        head = audio[:-ctx_len]
        if len(head) > xf and len(new) > xf:
            ramp = np.linspace(0, 1, xf, dtype=np.float32)
            seam = head[-xf:] * (1 - ramp) + new[:xf] * ramp
            audio = np.concatenate([head[:-xf], seam, new[xf:]])
        else:
            audio = np.concatenate([head, new])
        audio = audio[: int(min(total, len(audio) / sr) * sr)]
    return audio[: int(total * sr)], sr


try:
    proc, model = load(torch.float16 if DEV != "cpu" else torch.float32)
    audio, sr = make(proc, model)
except FloatingPointError:
    # Vissa kort ger trasigt ljud i halv precision – gör om i full precision
    print("Halv precision gav ogiltigt ljud – försöker igen i full precision", flush=True)
    del model
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    proc, model = load(torch.float32)
    audio, sr = make(proc, model)

wp(0.98, "Sparar musiken")
audio = audio / (np.abs(audio).max() + 1e-9) * 0.89
out = a.get("out") or new_path("ai_musik_raw", "wav")
with wave.open(out, "w") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
print(f"Frö (seed): {seed}", flush=True)
wr(out)
wp(1.0, "Klar")
