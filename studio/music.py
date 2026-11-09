"""Musik till filmerna.

Två sätt:
- AI-musik med MusicGen (riktiga instrumentljud, valfri stil, tar 1–5 min)
- Enkel musik som skapas med kod (direkt, ingen AI, ingen nedladdning)
Båda kan få naturljud (vind och fåglar) under.
"""
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import wave

import numpy as np

from common import BEFORE_WORKER, TMP, WORKERS, env as _env, ffmpeg, iter_lines, new_path, run_worker

SR = 44100
MOODS = {
    "Lugn (skog, natur)": dict(chords=[[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]], bar=4.0, pluck=True),
    "Varm och hoppfull": dict(chords=[[48, 52, 55], [55, 59, 62], [57, 60, 64], [53, 57, 60]], bar=3.0, pluck=True),
    "Drömsk": dict(chords=[[50, 53, 57, 60], [46, 50, 53, 57], [53, 57, 60, 64], [48, 52, 55, 59]], bar=5.0, pluck=False),
    "Allvarlig": dict(chords=[[45, 48, 52], [41, 45, 48], [43, 47, 50], [40, 44, 47]], bar=4.5, pluck=False),
}

# Färdiga stilar för AI-musiken (MusicGen förstår engelska bäst)
STYLES = {
    "Lugn akustisk (skog, natur)": "calm acoustic folk, fingerpicked guitar and soft piano, warm, peaceful nature documentary, slow tempo",
    "Varm och hoppfull": "warm hopeful cinematic music, piano and strings, gentle build, uplifting, inspiring",
    "Episk orkester": "epic orchestral film score, big drums, brass and strings, heroic, powerful, cinematic trailer",
    "Drömsk ambient": "dreamy ambient, soft synth pads, reverb, slow evolving textures, relaxing, no drums",
    "Lo-fi": "lofi hip hop, mellow rhodes piano, dusty drums, vinyl crackle, chill, relaxed, 80 bpm",
    "Glad pop": "upbeat happy pop instrumental, catchy melody, bright synths, claps, summer vibe, 118 bpm",
    "Spänning": "suspenseful dark cinematic tension, low strings, pulsing synth, mysterious, thriller",
    "Sorglig piano": "sad emotional solo piano, slow, melancholic, gentle, intimate",
    "Synthwave": "retro 80s synthwave, driving bassline, analog synths, gated drums, neon night drive",
    "Elektronisk dans": "energetic electronic dance music, four on the floor kick, big synth lead, festival, 126 bpm",
    "Jazz": "smooth jazz trio, upright bass, brushed drums, piano, relaxed cafe atmosphere",
    "Folkmusik (nordisk)": "nordic folk music, nyckelharpa and fiddle, acoustic, atmospheric, scandinavian",
}
SIZES = {"Snabb (small)": "small", "Bra (medium)": "medium", "Bäst (large)": "large"}

# AI-stilar som kan väljas i Snabbfilm och Bygg film (faller tillbaka på kodmusik om något går fel)
AI_PREFIX = "🤖 AI: "   # i filmerna: ACE-Step (instrumentalt), annars MusicGen
AI_MOODS = {AI_PREFIX + k: v for k, v in STYLES.items()}
ALL_MOODS = list(MOODS) + list(AI_MOODS)


def _note(m):
    return 440 * 2 ** ((m - 69) / 12)


def _nature(n, sr, seconds, rng):
    """Vind och fågelkvitter som en numpy-array."""
    t = np.arange(n) / sr
    wind = np.convolve(rng.standard_normal(n), np.ones(int(400 * sr / SR)) / int(400 * sr / SR), mode="same")
    wind *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t)
    out = 0.04 * wind / (np.max(np.abs(wind)) + 1e-9)
    for bt in rng.uniform(0.5, max(1.0, seconds - 1), size=max(2, int(seconds / 5))):
        for c in range(3):
            cs, L = int((bt + c * 0.13) * sr), int(0.08 * sr)
            if cs + L >= n:
                continue
            ct = np.arange(L) / sr
            f = 3800 + 1800 * ct / 0.08
            out[cs:cs + L] += 0.05 * np.sin(2 * np.pi * np.cumsum(f) / sr) * np.sin(np.pi * ct / 0.08)
    return out


def _fade(x, sr, fin=1.5, fout=2.5):
    n = len(x)
    fade = np.ones(n)
    fi, fo = min(n, int(fin * sr)), min(n, int(fout * sr))
    fade[:fi] = np.linspace(0, 1, fi)
    fade[-fo:] = np.minimum(fade[-fo:], np.linspace(1, 0, fo))
    return x * fade


def _write(path, x, sr):
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def _read(path):
    with wave.open(str(path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def _finish(wav, seconds, nature, out_path, lufs=-16, seed=0):
    """Klipper till rätt längd, mjuka in/uttoningar, ev. naturljud och jämn ljudstyrka."""
    x, sr = _read(wav)
    n = int(seconds * sr)
    if len(x) < n:   # för kort: spela om från början med övertoning
        reps = []
        while sum(len(r) for r in reps) < n + sr:
            reps.append(x)
        x = np.concatenate(reps)
    x = x[:n]
    if nature:
        x = x * 0.9 + 1.2 * _nature(len(x), sr, len(x) / sr, np.random.default_rng(seed))
    x = _fade(x, sr, fin=0.5, fout=min(2.5, len(x) / sr / 3))
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
    mid = str(TMP / f"fin_{time.time_ns()}.wav")
    _write(mid, x, sr)
    out = out_path or new_path("musik", "wav")
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", mid, "-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11",
                    "-ar", str(SR), out], check=True)
    os.remove(mid)
    return str(out)


def make(seconds: float, mood: str = "Lugn (skog, natur)", nature: bool = True, seed: int = 3, out_path=None) -> str:
    """Musik till en film. AI-stilar (som börjar med '🤖 AI: ') görs med MusicGen, annars med kod."""
    if mood in AI_MOODS:
        try:   # ACE-Step (MIT – filmen får säljas), instrumentalt
            wavs, _ = song(AI_MOODS[mood] + ", instrumental, no vocals", language="none",
                           seconds=max(10.0, min(240.0, seconds + 2)))
            return _finish(wavs[0], seconds, nature, out_path, lufs=-22)
        except Exception as e:
            print("ACE-Step-musiken misslyckades – provar MusicGen:", e, flush=True)
        try:
            return ai(AI_MOODS[mood], seconds, "medium", nature=nature, out_path=out_path, fade_out=2.5, lufs=-22)
        except Exception as e:
            print("AI-musiken misslyckades – använder den enkla musiken i stället:", e, flush=True)
            mood = list(MOODS)[0]
    cfg = MOODS.get(mood, MOODS[list(MOODS)[0]])
    rng = np.random.default_rng(seed)
    n = int((seconds + 0.5) * SR)
    t = np.arange(n) / SR
    mus = np.zeros(n)
    bar = cfg["bar"]
    for k in range(int(seconds / bar) + 2):
        s = int(k * bar * SR)
        if s >= n:
            break
        e = min(n, int((k + 1) * bar * SR + 1.5 * SR))
        seg = t[s:e] - k * bar
        env = np.clip(seg / 1.5, 0, 1) * np.clip((bar + 1.5 - seg) / 1.5, 0, 1)
        chord = cfg["chords"][k % len(cfg["chords"])]
        for m in chord:
            for det in (-0.12, 0.12):
                f = _note(m - 12) * 2 ** (det / 12)
                mus[s:e] += 0.05 * env * (np.sin(2 * np.pi * f * seg) + 0.3 * np.sin(4 * np.pi * f * seg))
        if cfg["pluck"]:
            arp = chord + [chord[0] + 12]
            step = bar / len(arp)
            for j, m in enumerate(arp):
                ps = s + int(j * step * SR)
                L = min(int(2.5 * SR), n - ps)
                if L <= 0:
                    continue
                pt = np.arange(L) / SR
                f = _note(m + 12)
                mus[ps:ps + L] += 0.06 * np.exp(-pt * 2.2) * (
                    np.sin(2 * np.pi * f * pt) + 0.25 * np.sin(4 * np.pi * f * pt) + 0.08 * np.sin(6 * np.pi * f * pt))
    if nature:
        mus += _nature(n, SR, seconds, rng)
    mus = _fade(mus, SR)
    mus = mus / (np.max(np.abs(mus)) + 1e-9) * 0.5
    out = out_path or new_path("musik", "wav")
    _write(out, mus, SR)
    return str(out)


def ai(prompt_en, seconds, size="medium", seed=-1, guidance=3.0, nature=False, out_path=None, fade_out=3.0,
       progress=None, lufs=-16):
    """AI-musik med MusicGen. Returnerar sökväg till en WAV (44,1 kHz) med jämn ljudstyrka.
    I filmer används lägre ljudstyrka (lufs=-22) så att musiken inte tar över rösten."""
    seed = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, 999999)
    raw = str(TMP / f"musicgen_{seed}_{os.getpid()}.wav")
    run_worker("musicgen.py", {"prompt": prompt_en, "seconds": float(seconds), "size": size, "seed": seed,
                               "guidance": float(guidance), "out": raw}, progress)
    x, sr = _read(raw)
    if nature:
        x = x * 0.9 + 1.2 * _nature(len(x), sr, len(x) / sr, np.random.default_rng(seed))
    x = _fade(x, sr, fin=0.3, fout=min(fade_out, len(x) / sr / 3))
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
    mid = str(TMP / f"musicgen_{seed}_{os.getpid()}_mix.wav")
    _write(mid, x, sr)
    out = out_path or new_path(f"ai_musik_{seed}", "wav")
    # Jämn ljudstyrka och samma format som resten av studion
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", mid, "-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11",
                    "-ar", str(SR), out], check=True)
    for f in (raw, mid):
        try:
            os.remove(f)
        except OSError:
            pass
    return str(out)


# ---------------- Låtar med sång: ACE-Step 1.5 (MIT-licens – får användas kommersiellt) ----------------
ACE_DIR = "/content/ACE-Step-1.5"
ACE_PY = f"{ACE_DIR}/.venv/bin/python"
ACE_OK = f"{ACE_DIR}/.fv_ok"
ACE_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"   # testad version – byt bara efter test
ACE_IDLE_MIN = 10   # stäng servern efter så här många minuter utan låtar (frigör grafikkort och minne)
LANGUAGES = {"Svenska": "sv", "Engelska": "en", "Norska": "no", "Danska": "da", "Finska": "fi", "Tyska": "de",
             "Spanska": "es", "Franska": "fr", "Instrumental (ingen sång)": "none"}

_install_lock = threading.Lock()
_ace_lock = threading.Lock()
_ace = None          # serverprocessen
_ace_timer = None    # stänger servern när den inte används


def ace_installed():
    return os.path.exists(ACE_OK)


def ensure_ace(progress=None):
    """Installerar ACE-Step i en egen Python-miljö första gången (ca 5–10 min)."""
    if ace_installed():
        return
    with _install_lock:   # om installationen redan pågår i bakgrunden väntar vi in den
        if ace_installed():
            return
        free = shutil.disk_usage("/content" if os.path.exists("/content") else "/").free / 1024**3
        if free < 22:
            raise RuntimeError(f"ACE-Step behöver ca 20 GB ledigt diskutrymme, men bara {free:.0f} GB finns. "
                               "Starta om sessionen (då töms disken) och prova igen.")
        if progress:
            progress(0.01, desc="Installerar ACE-Step (bara första gången, ca 5–10 min)")
        print("🎤 Installerar ACE-Step …", flush=True)

        def sh(cmd, cwd=None):
            print("  $", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=cwd, stdout=subprocess.DEVNULL if cwd else None)

        if not os.path.exists(f"{ACE_DIR}/pyproject.toml"):
            os.makedirs(ACE_DIR, exist_ok=True)
            sh(["git", "init", "-q"], ACE_DIR)
            sh(["git", "remote", "add", "origin", "https://github.com/ACE-Step/ACE-Step-1.5.git"], ACE_DIR)
            sh(["git", "fetch", "-q", "--depth", "1", "origin", ACE_COMMIT], ACE_DIR)
            sh(["git", "checkout", "-q", "FETCH_HEAD"], ACE_DIR)
        sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
        sh(["uv", "sync", "--frozen", "--python", "3.11"], ACE_DIR)
        open(ACE_OK, "w").write("ok")
        print("🎤 ACE-Step är installerat.", flush=True)


def install_in_background():
    """Startar installationen direkt när studion startar, så att första låten inte behöver vänta."""
    if ace_installed():
        return
    try:
        free = shutil.disk_usage("/content" if os.path.exists("/content") else "/").free / 1024**3
    except Exception:
        return
    if free < 40:   # lämna plats åt bild- och videomodellerna – installera då först när någon gör en låt
        print("🎤 ACE-Step installeras när du gör din första låt (för lite disk för att göra det i förväg).", flush=True)
        return

    def run():
        try:
            ensure_ace()
        except Exception as e:
            print("🎤 Bakgrundsinstallationen av ACE-Step misslyckades (försöker igen vid första låten):", e,
                  flush=True)
    threading.Thread(target=run, daemon=True).start()


def _gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).stdout
        return sum(1 for l in out.splitlines() if l.startswith("GPU"))
    except Exception:
        return 0


GPUS = _gpu_count()


def stop_ace():
    """Stänger ACE-Steps server så att grafikkortet blir ledigt."""
    global _ace
    p, _ace = _ace, None
    if p and p.poll() is None:
        print("🎤 Stänger ACE-Step för att frigöra grafikkortet", flush=True)
        try:
            p.stdin.close()
            p.wait(timeout=20)
        except Exception:
            p.kill()


def _before_worker(script):
    # Två kort (Kaggle): ACE-Step ligger på kort 2, där bara videomodellernas textmodell också vill vara.
    # Ett kort (Colab): stäng alltid, så att varje modell får hela kortet.
    if _ace is not None and (GPUS < 2 or script in ("video.py", "wan.py")):
        with _ace_lock:
            stop_ace()


BEFORE_WORKER.append(_before_worker)


def _arm_idle_timer():
    global _ace_timer
    if _ace_timer:
        _ace_timer.cancel()

    def idle():
        if _ace_lock.acquire(blocking=False):
            try:
                stop_ace()
            finally:
                _ace_lock.release()
    _ace_timer = threading.Timer(ACE_IDLE_MIN * 60, idle)
    _ace_timer.daemon = True
    _ace_timer.start()


def _read_until(p, progress, end_markers):
    """Läser serverns utskrifter tills en av markeringarna kommer. Returnerar (markering, resultat, rad)."""
    results, tail = [], []
    for line in iter_lines(p.stdout):
        if line.startswith("@@P "):
            try:
                _, frac, *desc = line.split(" ")
                if progress:
                    progress(float(frac), desc=" ".join(desc))
            except Exception:
                pass
        elif line.startswith("@@R "):
            results.append(line[4:].strip())
        elif any(line.startswith(m) for m in end_markers):
            return line.split(" ")[0], results, line
        else:
            print(line, flush=True)
            if "%|" not in line:
                tail = (tail + [line])[-20:]
    errs = [l for l in tail if "rror" in l]
    raise RuntimeError("ACE-Step stängdes oväntat: " + ((errs[-1] if errs else " ".join(tail[-4:]))[-500:]))


def _start_ace(progress):
    global _ace
    ensure_ace(progress)
    e = _env()
    if GPUS >= 2:
        e["CUDA_VISIBLE_DEVICES"] = "1"   # Kaggle: kort 2, så att bild- och pratmodellerna har kort 1 fritt
    cmd = [ACE_PY, "-u", str(WORKERS / "acestep_song.py"), json.dumps({"server": True, "ace_dir": ACE_DIR})]
    _ace = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e)
    _read_until(_ace, progress, ["@@READY"])


def ace_job(job, progress=None):
    """Skickar ett jobb till ACE-Steps server (startar den vid behov). Returnerar (wav-lista, info)."""
    with _ace_lock:
        if _ace_timer:
            _ace_timer.cancel()
        try:
            if _ace is None or _ace.poll() is not None:
                _start_ace(progress)
            _ace.stdin.write((json.dumps(job, ensure_ascii=False) + "\n").encode())
            _ace.stdin.flush()
            mark, res, line = _read_until(_ace, progress, ["@@DONE", "@@ERR"])
        except Exception:
            stop_ace()
            raise
        finally:
            _arm_idle_timer()
    if mark == "@@ERR":
        raise RuntimeError(line[6:].strip() or "ACE-Step misslyckades.")
    wavs = [r for r in res if r.endswith(".wav")]
    js = next((r for r in res if r.endswith(".json")), None)
    info = json.load(open(js)) if js else {}
    return wavs, info


def to_mp3(wav):
    """Sparar en MP3-kopia (320 kbit/s) bredvid WAV-filen och returnerar sökvägen."""
    mp3 = os.path.splitext(wav)[0] + ".mp3"
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", wav, "-codec:a", "libmp3lame", "-b:a", "320k", mp3],
                   check=True)
    return mp3


def song(caption_en, query="", lyrics="", write_lyrics=False, language="sv", seconds=0, bpm=0, seed=-1,
         variants=1, progress=None):
    """Gör en låt (1–2 versioner) med ACE-Step. Returnerar (wav-lista, info med låttext m.m.)."""
    return ace_job({"task": "song", "caption": caption_en, "query": query, "lyrics": lyrics,
                    "write_lyrics": bool(write_lyrics), "language": language, "seconds": float(seconds or 0),
                    "bpm": int(bpm or 0), "seed": int(seed if seed is not None else -1),
                    "variants": int(variants or 1)}, progress)


def repaint(src, start, end, caption_en, lyrics="", language="sv", seed=-1, progress=None):
    """Gör om en del (start–slut i sekunder) av en låt. Resten behålls."""
    return ace_job({"task": "repaint", "src": src, "start": float(start), "end": float(end),
                    "caption": caption_en, "lyrics": lyrics, "language": language, "seed": int(seed)}, progress)


def cover(src, caption_en, strength=0.5, lyrics="", language="sv", seed=-1, progress=None):
    """Ny version av en låt i en annan stil. strength: hur mycket av originalet som behålls (0–1)."""
    return ace_job({"task": "cover", "src": src, "strength": float(strength), "caption": caption_en,
                    "lyrics": lyrics, "language": language, "seed": int(seed)}, progress)
