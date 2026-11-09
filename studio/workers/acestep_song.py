"""Låtar med sång (eller instrumentalt) med ACE-Step 1.5 (ACE Studio & StepFun, MIT-licens).

Körs i ACE-Steps egen Python-miljö (/content/ACE-Step-1.5/.venv), som installeras första gången.
Kan sjunga på svenska, engelska m.fl., skriva låttexten själv, göra om en del av en låt
och göra en ny version av en låt i en annan stil.

Två sätt att köra:
- en gång:  argumenten (JSON) är ett jobb, se run_job()
- server:   {"server": true} – laddar modellerna en gång och tar sedan jobb (en JSON per rad) på stdin.
            Varje jobb avslutas med '@@DONE' eller '@@ERR <text>'. Då slipper varje låt ladda om modellerna.

Jobb (JSON): task (song|repaint|cover), caption, query, lyrics, write_lyrics, language (sv|en|…|none),
             seconds (0 = auto), bpm (0 = auto), seed, variants (1–4), quality (fast|normal|high),
             src (ljudfil, för repaint/cover), start, end (sekunder, repaint), strength (0–1, cover)
Resultat: '@@R <wav>' per version och '@@R <json>' med låttext och inställningar.
"""
import gc as _gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

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
from acestep.model_downloader import (ensure_main_model, get_checkpoints_dir, check_main_model_exists,  # noqa: E402
                                      check_model_exists, ensure_lm_model)
from acestep.constants import TASK_INSTRUCTIONS  # noqa: E402

gc = get_gpu_config()
set_global_gpu_config(gc)
ck = get_checkpoints_dir()
if not check_main_model_exists(ck):
    wp(0.04, "Laddar ner ACE-Step (bara första gången, ca 10 GB)")
    ok, msg = ensure_main_model(ck)
    if not ok:
        raise RuntimeError(f"Kunde inte ladda ner ACE-Step: {msg}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
# Äldre kort (T4, P100 – Colab/Kaggle gratis) saknar bfloat16. Då kör ACE-Step i float16, som svämmar över
# (NaN, tyst/trasig låt). Lösning: musikmodellen i float32 och den mindre språkmodellen (0.6B) så att allt ryms.
OLD_GPU = dev == "cuda" and torch.cuda.get_device_capability()[0] < 8
# Avlastning till vanligt minne bara på små kort: gratis-Colab har bara ca 12 GB RAM, och 2B-modellen
# (ca 5 GB) + språkmodellen (ca 3,5 GB) får plats direkt på T4-kortets 15 GB.
offload = dev == "cuda" and gc.gpu_memory_gb < 12

# Kvalitetsnivåer: modell, antal steg, guidance (bara för icke-turbo) och shift
QUALITY = {
    "fast":   dict(model="acestep-v15-turbo", steps=8, guidance=None, shift=3.0),     # 2B turbo
    "normal": dict(model="acestep-v15-xl-turbo", steps=8, guidance=None, shift=3.0),  # 4B turbo, bättre ljud
    "high":   dict(model="acestep-v15-xl-sft", steps=50, guidance=7.0, shift=1.0),    # 4B sft, bäst men långsam
}
dit = None
DIT_MODEL = None


def load_dit(model):
    """Laddar (eller byter till) musikmodellen. Byte tar ca 1–3 min, första gången också nedladdning."""
    global dit, DIT_MODEL
    if dit is not None and DIT_MODEL == model:
        return
    big = "-xl-" in model
    if big and OLD_GPU:
        # XL i float32 (ca 18 GB) får inte plats på T4/P100, och i float16 blir det NaN
        print("XL-modellen kräver ett nyare grafikkort (A100/L4) – använder 2B-modellen i stället.", flush=True)
        model, big = QUALITY["fast"]["model"], False
    if dit is not None and DIT_MODEL == model:
        return
    if dit is not None:
        wp(0.05, "Byter musikmodell")
        dit = None
        _gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
    if big and not check_model_exists(model, ck):
        free = shutil.disk_usage(str(ck)).free / 1024**3
        if free < 25:
            raise RuntimeError(f"XL-modellen behöver ca 20 GB ledigt diskutrymme men bara {free:.0f} GB finns. "
                               "Välj kvalitet Snabb, eller starta om sessionen så töms disken.")
        wp(0.06, "Laddar ner XL-modellen (bara första gången, ca 10–20 GB)")
    wp(0.10, "Laddar musikmodellen" + (" (XL)" if big else ""))
    h = AceStepHandler()
    # XL (4B) behöver avlastning till vanligt minne på kort under 20 GB (T4 i Colab/Kaggle har 15–16 GB)
    msg, ok = h.initialize_service(project_root=ACE, config_path=model, device=dev,
                                   use_flash_attention=False, compile_model=False,
                                   offload_to_cpu=offload or (big and gc.gpu_memory_gb < 20),
                                   offload_dit_to_cpu=False)
    if not ok:
        raise RuntimeError(f"Musikmodellen ({model}) startade inte: {msg}")
    if OLD_GPU:
        wp(0.12, "Ställer om till float32 (äldre grafikkort)")
        h.dtype = torch.float32
        for name in ("model", "text_encoder"):
            m = getattr(h, name, None)
            if m is not None and not isinstance(m, torch.Tensor):
                setattr(h, name, m.to(torch.float32))
        sl = getattr(h, "silence_latent", None)
        if isinstance(sl, torch.Tensor):
            h.silence_latent = sl.to(torch.float32)
        torch.cuda.empty_cache()
    dit, DIT_MODEL = h, model


load_dit(QUALITY["fast"]["model"])

wp(0.16, "Laddar språkmodellen som planerar låten")
llm = LLMHandler()
try:
    LM_NAME = "acestep-5Hz-lm-0.6B" if OLD_GPU else "acestep-5Hz-lm-1.7B"
    if OLD_GPU:
        ensure_lm_model(LM_NAME, ck)
    msg, ok = llm.initialize(checkpoint_dir=str(ck), lm_model_path=LM_NAME, backend="pt",
                             device=dev, offload_to_cpu=offload)
    if not ok:
        print("Språkmodellen startade inte – kör utan den:", msg, flush=True)
except Exception as e:  # låten går att göra ändå, men blir lite sämre
    print("Språkmodellen startade inte – kör utan den:", e, flush=True)
LM_OK = bool(getattr(llm, "llm_initialized", False))


def to_wav(src, out):
    if os.path.splitext(src)[1].lower() == ".wav":
        shutil.move(src, out)
    else:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, out], check=True)


def run_job(j):
    task = j.get("task") or "song"
    lang = (j.get("language") or "sv").strip()
    instrumental = lang == "none"
    caption = (j.get("caption") or "").strip()
    lyrics = (j.get("lyrics") or "").strip()
    bpm = int(j.get("bpm") or 0) or None
    seconds = float(j.get("seconds") or 0)
    duration = max(10.0, min(240.0, seconds)) if seconds > 0 else -1.0
    variants = max(1, min(4, int(j.get("variants") or 1)))
    q = QUALITY.get(j.get("quality") or "fast", QUALITY["fast"])
    if task == "song" or j.get("quality"):
        load_dit(q["model"])
    seed = int(j.get("seed", -1))
    if seed < 0:
        seed = int.from_bytes(os.urandom(3), "little")
    seeds = [seed + i for i in range(variants)]

    # Låt AI:n skriva texten (och en utförligare beskrivning) om ingen text skrevs
    if task == "song" and not instrumental and (j.get("write_lyrics") or not lyrics):
        if not LM_OK:
            raise RuntimeError("Språkmodellen behövs för att skriva låttexten men startade inte. Skriv texten själv.")
        wp(0.22, "Skriver låttexten")
        s = create_sample(llm, query=(j.get("query") or caption or "a happy pop song")[:1000],
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

    kw = dict(caption=caption[:512], lyrics="[Instrumental]" if instrumental else lyrics[:4096],
              instrumental=instrumental, vocal_language="unknown" if instrumental else lang,
              bpm=bpm, duration=duration, inference_steps=int(j.get("steps") or q["steps"]), seed=seed,
              shift=q["shift"], thinking=LM_OK and task == "song")
    if q["guidance"] and "turbo" not in (DIT_MODEL or ""):
        kw.update(guidance_scale=q["guidance"])
    if task in ("repaint", "cover"):
        if not j.get("src") or not os.path.exists(j["src"]):
            raise RuntimeError("Ladda upp låten som ska ändras.")
        kw.update(task_type=task, instruction=TASK_INSTRUCTIONS[task], src_audio=j["src"], duration=-1.0)
        if task == "repaint":
            kw.update(repainting_start=float(j.get("start", 0)), repainting_end=float(j.get("end", -1)),
                      chunk_mask_mode="explicit")
        else:
            kw.update(audio_cover_strength=max(0.05, min(1.0, float(j.get("strength", 0.5)))))
    wp(0.3, {"song": "Komponerar låten", "repaint": "Gör om den valda delen", "cover": "Gör en ny version"}[task])
    params = GenerationParams(**kw)
    # Flera versioner görs i omgångar så att grafikkortets minne räcker (XL på T4: en i taget)
    big = "-xl-" in (DIT_MODEL or "")
    cap = 1 if (OLD_GPU or big and gc.gpu_memory_gb < 20) else (2 if gc.gpu_memory_gb < 20 else 4)
    save_dir = str(TMP / f"ace_{time.time_ns()}")
    os.makedirs(save_dir, exist_ok=True)
    prefix = {"song": "lat", "repaint": "lat_andrad", "cover": "lat_ny_stil"}[task]
    outs, used = [], []
    for start in range(0, len(seeds), cap):
        part = seeds[start:start + cap]
        n0, n = start, len(seeds)

        def prog(*args, _n0=n0, _k=len(part), **kw2):
            try:
                f = float(args[0]) if args else float(kw2.get("value", 0) or 0)
            except Exception:
                f = 0.0
            f = (_n0 + max(0.0, min(1.0, f)) * _k) / n
            d = kw2.get("desc") or (args[1] if len(args) > 1 else "") or "Skapar låten"
            wp(0.3 + 0.65 * f, (f"Version {_n0 + 1}–{_n0 + _k} av {n}: " if n > 1 else "") + d)
        config = GenerationConfig(batch_size=len(part), use_random_seed=False, seeds=part, audio_format="wav")
        res = generate_music(dit, llm, params, config, save_dir=save_dir, progress=prog)
        if not res.success or not res.audios:
            if outs:   # några versioner blev klara – lämna dem i stället för att kasta allt
                print("Fler versioner misslyckades:", res.error or res.status_message, flush=True)
                break
            raise RuntimeError(f"ACE-Step misslyckades: {res.error or res.status_message}")
        for i, au in enumerate(res.audios):
            sd = (au.get("params") or {}).get("seed", part[min(i, len(part) - 1)])
            out = new_path(f"{prefix}_{sd}_v{len(outs) + 1}", "wav")
            to_wav(au["path"], out)
            outs.append(out)
            used.append(sd)
    meta = {"task": task, "seed": seed, "seeds": used, "caption": caption, "model": DIT_MODEL,
            "lyrics": "" if instrumental else lyrics, "language": lang, "bpm": bpm, "duration": duration}
    mpath = os.path.splitext(outs[0])[0] + ".json"
    json.dump(meta, open(mpath, "w"), ensure_ascii=False, indent=1)
    shutil.rmtree(save_dir, ignore_errors=True)
    print(f"Frö (seed): {seed}", flush=True)
    return outs + [mpath]


if a.get("server"):
    print("@@READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            for r in run_job(json.loads(line)):
                wr(r)
            print("@@DONE", flush=True)
        except Exception as e:
            traceback.print_exc()
            msg = str(e)
            if "out of memory" in msg.lower():
                msg = "Grafikkortet fick slut på minne. Prova kortare låt eller en version i taget."
            print("@@ERR " + msg.replace("\n", " ")[:600], flush=True)
        finally:
            if dev == "cuda":
                torch.cuda.empty_cache()
else:
    for r in run_job(a):
        wr(r)
    wp(1.0, "Klar")
