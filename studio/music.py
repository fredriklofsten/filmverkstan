"""Musik till filmerna.

Två sätt:
- AI-musik med MusicGen (riktiga instrumentljud, valfri stil, tar 1–5 min)
- Enkel musik som skapas med kod (direkt, ingen AI, ingen nedladdning)
Båda kan få naturljud (vind och fåglar) under.
"""
import os
import random
import subprocess
import wave

import numpy as np

from common import TMP, ffmpeg, new_path, run_worker

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
AI_PREFIX = "🤖 AI: "
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


def make(seconds: float, mood: str = "Lugn (skog, natur)", nature: bool = True, seed: int = 3, out_path=None) -> str:
    """Musik till en film. AI-stilar (som börjar med '🤖 AI: ') görs med MusicGen, annars med kod."""
    if mood in AI_MOODS:
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
LANGUAGES = {"Svenska": "sv", "Engelska": "en", "Norska": "no", "Danska": "da", "Finska": "fi", "Tyska": "de",
             "Spanska": "es", "Franska": "fr", "Instrumental (ingen sång)": "none"}


def ensure_ace(progress=None):
    """Installerar ACE-Step i en egen Python-miljö första gången (ca 5–10 min)."""
    if os.path.exists(ACE_OK):
        return
    say = (lambda t: progress(0.01, desc=t)) if progress else print
    import shutil
    free = shutil.disk_usage("/content" if os.path.exists("/content") else "/").free / 1024**3
    if free < 22:
        raise RuntimeError(f"ACE-Step behöver ca 20 GB ledigt diskutrymme, men bara {free:.0f} GB finns. "
                           "Starta om sessionen (då töms disken) och prova igen.")
    say("Installerar ACE-Step (bara första gången, ca 5–10 min)")
    import sys

    def sh(cmd, cwd=None):
        print("  $", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=cwd)

    if not os.path.exists(f"{ACE_DIR}/pyproject.toml"):
        os.makedirs(ACE_DIR, exist_ok=True)
        sh(["git", "init", "-q"], ACE_DIR)
        sh(["git", "remote", "add", "origin", "https://github.com/ACE-Step/ACE-Step-1.5.git"], ACE_DIR)
        sh(["git", "fetch", "-q", "--depth", "1", "origin", ACE_COMMIT], ACE_DIR)
        sh(["git", "checkout", "-q", "FETCH_HEAD"], ACE_DIR)
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    sh(["uv", "sync", "--frozen", "--python", "3.11"], ACE_DIR)
    open(ACE_OK, "w").write("ok")


def song(caption_en, query="", lyrics="", write_lyrics=False, language="sv", seconds=0, bpm=0, seed=-1,
         progress=None):
    """Gör en låt med ACE-Step. Returnerar (wav-sökväg, info-dict med låttext m.m.)."""
    ensure_ace(progress)
    res = run_worker("acestep_song.py", {"caption": caption_en, "query": query, "lyrics": lyrics,
                                         "write_lyrics": bool(write_lyrics), "language": language,
                                         "seconds": float(seconds or 0), "bpm": int(bpm or 0),
                                         "seed": int(seed if seed is not None else -1), "ace_dir": ACE_DIR},
                     progress, python=ACE_PY)
    wav = next(r for r in res if r.endswith(".wav"))
    js = next((r for r in res if r.endswith(".json")), None)
    import json
    info = json.load(open(js)) if js else {}
    return wav, info
