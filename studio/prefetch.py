"""Förladdar AI-modellerna i bakgrunden direkt när studion startar.

Medan du skriver din första prompt laddas modellerna ner, så att jobben
startar direkt i stället för att vänta på nedladdning. Körs som en egen
process från app.py. Om ett jobb startar samtidigt väntar det bara in samma
nedladdning (huggingface_hub låser filerna), inget laddas ner två gånger.
"""
import os
import shutil
import sys
import time

from huggingface_hub import snapshot_download

# (namn, repo, filer) – i den ordning de oftast används
MODELS = [
    ("översättning", "Helsinki-NLP/opus-mt-sv-en", ["*.json", "*.spm", "*.txt", "*.safetensors", "pytorch_model.bin"], 1),
    ("bildmodell", "madebyollin/sdxl-vae-fp16-fix", ["*.json", "diffusion_pytorch_model.safetensors"], 1),
    ("bildmodell", "SG161222/RealVisXL_V5.0_Lightning", ["*.json", "*.txt", "*.fp16.safetensors"], 8),
    ("textmodell för video", "google-t5/t5-base", ["*.json", "spiece.model"], 1),
    ("textmodell för video", "city96/t5-v1_1-xxl-encoder-bf16", ["*.json", "*.safetensors"], 12),
    ("videomodell", "Lightricks/LTX-Video-0.9.5",
     ["model_index.json", "scheduler/*", "transformer/*", "vae/*"], 10),
]


def free_gb():
    d = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    os.makedirs(d, exist_ok=True)
    return shutil.disk_usage(d).free / 1024**3


def main():
    t0 = time.time()
    for name, repo, patterns, need_gb in MODELS:
        if free_gb() < need_gb + 5:
            print(f"[förladdning] hoppar över {repo} – för lite diskutrymme", flush=True)
            continue
        try:
            t = time.time()
            snapshot_download(repo, allow_patterns=patterns, max_workers=8)
            print(f"[förladdning] {name} klar ({repo}, {time.time() - t:.0f} s)", flush=True)
        except Exception as e:  # nätverksfel m.m. – jobbet laddar då ner själv senare
            print(f"[förladdning] kunde inte hämta {repo}: {e}", flush=True)
    print(f"[förladdning] ✅ alla modeller redo efter {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
