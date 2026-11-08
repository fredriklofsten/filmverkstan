"""Ljud → ord med exakta tider (Whisper large-v3-turbo), för undertexter i takt med talet.
Tar en lista ljudfiler och skriver en JSON-fil: [[[ord, start, slut], …], …] i samma ordning."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wargs, wp, wr, new_path  # noqa

import torch
from transformers import pipeline

a = wargs()
files = a["files"]
wp(0.05, "Laddar Whisper (första gången ca 1,6 GB)")
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
               torch_dtype=torch.float16, device="cuda:0")
result = []
for i, f in enumerate(files):
    wp(0.2 + 0.75 * i / max(1, len(files)), f"Lyssnar på replik {i + 1} av {len(files)}")
    words = []
    try:
        out = asr(f, return_timestamps="word", generate_kwargs={"language": "swedish", "task": "transcribe"})
        for c in out.get("chunks", []):
            s, e = c.get("timestamp", (None, None))
            t = c.get("text", "").strip()
            if t and s is not None:
                words.append([t, float(s), float(e if e is not None else s + 0.3)])
    except Exception as ex:
        print("Whisper misslyckades för", f, ex, flush=True)
    result.append(words)
path = new_path("ord", "json")
json.dump(result, open(path, "w"), ensure_ascii=False)
wr(path)
wp(1.0, "Klar")
