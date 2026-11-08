"""Naturlig svensk röst med Chatterbox Multilingual (körs i en egen Python-miljö: /content/ttsenv).
Kan härma en röst från en kort inspelning (5–15 sekunder)."""
import sys, os, re, json
a = json.loads(sys.argv[1])


def wp(frac, text):
    print(f"@@P {frac:.3f} {text}", flush=True)


wp(0.02, "Laddar röstmodellen")
import torch
import torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
text = " ".join(a["text"].split())
# Dela upp i meningar och slå ihop till bitar på max ~250 tecken (modellen klarar inte för långa bitar)
sents = re.split(r"(?<=[.!?])\s+", text)
chunks, cur = [], ""
for s in sents:
    if len(cur) + len(s) < 250:
        cur = (cur + " " + s).strip()
    else:
        if cur:
            chunks.append(cur)
        cur = s
if cur:
    chunks.append(cur)

ref = a.get("ref") or None
pieces = []
silence = torch.zeros(1, int(model.sr * 0.35))
for i, c in enumerate(chunks):
    wp(0.1 + 0.85 * i / len(chunks), f"Talar in del {i + 1} av {len(chunks)}")
    wav = model.generate(c, language_id="sv", audio_prompt_path=ref,
                         exaggeration=float(a.get("exaggeration", 0.5)), cfg_weight=float(a.get("cfg", 0.5)),
                         temperature=float(a.get("temperature", 0.8)))
    pieces += [wav.cpu(), silence]
out = torch.cat(pieces[:-1], dim=1)
torchaudio.save(a["out"], out, model.sr)
print(f"@@R {a['out']}", flush=True)
wp(1.0, "Klar")
