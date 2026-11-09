"""Låtar med sång (eller instrumentalt) med ACE-Step 1.5 (ACE Studio & StepFun, MIT-licens).

Körs i ACE-Steps egen Python-miljö (/content/ACE-Step-1.5/.venv), som installeras första gången.
Kan sjunga på svenska, engelska m.fl. Kan också skriva låttexten själv utifrån en beskrivning.

Argument (JSON): caption, query, lyrics, write_lyrics, language (sv|en|…|none), seconds (0 = auto),
                 bpm (0 = auto), seed, steps, ace_dir
Skriver WAV + en JSON med låttext och beskrivning ('@@R <wav>' och '@@R <json>').
"""
import json
import os
import shutil
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # ett grafikkort räcker (Kaggle har två)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wargs, wp, wr, new_path, TMP  # noqa: E402

a = wargs()
ACE = a.get("ace_dir") or "/content/ACE-Step-1.5"
sys.path.insert(0, ACE)
os.chdir(ACE)

wp(0.02, "Startar ACE-Step")
import torch  # noqa: E402
from acestep.gpu_config import get_gpu_config, set_global_gpu_config  # noqa: E402
from acestep.handler import AceStepHandler  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402
from acestep.inference import GenerationParams, GenerationConfig, generate_music, create_sample  # noqa: E402
from acestep.model_downloader import ensure_main_model, get_checkpoints_dir, check_main_model_exists  # noqa: E402

gc = get_gpu_config()
set_global_gpu_config(gc)
ck = get_checkpoints_dir()
if not check_main_model_exists(ck):
    wp(0.04, "Laddar ner ACE-Step (bara första gången, ca 10 GB)")
    ok, msg = ensure_main_model(ck)
    if not ok:
        raise RuntimeError(f"Kunde inte ladda ner ACE-Step: {msg}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
offload = dev == "cuda" and gc.gpu_memory_gb < 20

wp(0.10, "Laddar musikmodellen")
dit = AceStepHandler()
msg, ok = dit.initialize_service(project_root=ACE, config_path="acestep-v15-turbo", device=dev,
                                 use_flash_attention=False, compile_model=False,
                                 offload_to_cpu=offload, offload_dit_to_cpu=False)
if not ok:
    raise RuntimeError(f"Musikmodellen startade inte: {msg}")

wp(0.16, "Laddar språkmodellen som planerar låten")
llm = LLMHandler()
try:
    msg, ok = llm.initialize(checkpoint_dir=str(ck), lm_model_path="acestep-5Hz-lm-1.7B", backend="pt",
                             device=dev, offload_to_cpu=offload)
    if not ok:
        print("Språkmodellen startade inte – kör utan den:", msg, flush=True)
except Exception as e:  # låten går att göra ändå, men blir lite sämre
    print("Språkmodellen startade inte – kör utan den:", e, flush=True)
lm_ok = bool(getattr(llm, "llm_initialized", False))

lang = (a.get("language") or "sv").strip()
instrumental = lang == "none"
caption = (a.get("caption") or "").strip()
lyrics = (a.get("lyrics") or "").strip()
bpm = int(a.get("bpm") or 0) or None
seconds = float(a.get("seconds") or 0)
duration = max(10.0, min(240.0, seconds)) if seconds > 0 else -1.0
seed = int(a.get("seed", -1))
if seed < 0:
    seed = int.from_bytes(os.urandom(3), "little")

# Låt AI:n skriva texten (och en utförligare beskrivning) om ingen text skrevs
if not instrumental and (a.get("write_lyrics") or not lyrics):
    if not lm_ok:
        raise RuntimeError("Språkmodellen behövs för att skriva låttexten men startade inte. Skriv texten själv.")
    wp(0.22, "Skriver låttexten")
    s = create_sample(llm, query=(a.get("query") or caption or "a happy pop song")[:1000],
                      instrumental=False, vocal_language=lang)
    if not s.success or not (s.lyrics or "").strip():
        raise RuntimeError(f"Kunde inte skriva låttexten: {s.error or s.status_message}")
    lyrics = s.lyrics.strip()
    if s.caption:
        caption = f"{caption}, {s.caption}" if caption else s.caption
    bpm = bpm or s.bpm
    if duration <= 0 and s.duration:
        duration = float(s.duration)

if not caption:
    caption = "instrumental music" if instrumental else "pop song"


def progress(*args, **kw):
    try:
        f = float(args[0]) if args else float(kw.get("value", 0) or 0)
    except Exception:
        f = 0.0
    wp(0.3 + 0.65 * max(0.0, min(1.0, f)), kw.get("desc") or (args[1] if len(args) > 1 else "") or "Skapar låten")


wp(0.3, "Komponerar låten")
params = GenerationParams(
    caption=caption[:512],
    lyrics="[Instrumental]" if instrumental else lyrics[:4096],
    instrumental=instrumental,
    vocal_language="unknown" if instrumental else lang,
    bpm=bpm,
    duration=duration,
    inference_steps=int(a.get("steps") or 8),
    seed=seed,
    shift=3.0,
    thinking=lm_ok,
)
config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
save_dir = str(TMP / f"ace_{time.time_ns()}")
os.makedirs(save_dir, exist_ok=True)
res = generate_music(dit, llm, params, config, save_dir=save_dir, progress=progress)
if not res.success or not res.audios:
    raise RuntimeError(f"ACE-Step misslyckades: {res.error or res.status_message}")

src = res.audios[0]["path"]
out = new_path(f"lat_{seed}", "wav")
if os.path.splitext(src)[1].lower() == ".wav":
    shutil.move(src, out)
else:
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, out], check=True)
meta = {"seed": seed, "caption": caption, "lyrics": "" if instrumental else lyrics, "language": lang,
        "bpm": bpm, "duration": duration}
mpath = os.path.splitext(out)[0] + ".json"
json.dump(meta, open(mpath, "w"), ensure_ascii=False, indent=1)
shutil.rmtree(save_dir, ignore_errors=True)
print(f"Frö (seed): {seed}", flush=True)
wr(out)
wr(mpath)
wp(1.0, "Klar")
