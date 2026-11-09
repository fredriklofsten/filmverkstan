"""Musikvideo från en låt: bilder som följer texten, mjuka kamerarörelser (eller rörliga scener),
undertexter i takt med sången och låten som ljud."""
import difflib
import json
import re
import subprocess
import time

from common import TMP, duration, ffmpeg, new_path, run_worker
from montage import FORMATS, X, _captions, _scene, _subtitle
from translate import to_english

STYLES = {
    "Fotorealistisk": "",
    "Filmisk": "cinematic film still, dramatic lighting, shallow depth of field, anamorphic",
    "Målning": "oil painting, painterly brush strokes, rich colors, fine art",
    "Anime": "anime style illustration, vibrant colors, detailed background",
    "Drömsk": "dreamy soft focus, pastel colors, ethereal glow, film grain",
    "Retro 80-tal": "retro 1980s aesthetic, neon lights, synthwave colors, VHS look",
}
WHISPER_LANG = {"sv": "swedish", "en": "english", "no": "norwegian", "da": "danish", "fi": "finnish",
                "de": "german", "es": "spanish", "fr": "french"}


def lyric_lines(lyrics):
    """Låttext → sångrader (utan [verse]/[chorus]-märken och tomma rader)."""
    out = []
    for ln in (lyrics or "").splitlines():
        ln = re.sub(r"\[[^\]]*\]", "", ln).strip()
        if ln:
            out.append(ln)
    return out


def _norm(w):
    return re.sub(r"[^\wåäöæø]", "", w.lower())


def align(lines, words):
    """Ger varje sångrad en start- och sluttid genom att matcha raden mot orden Whisper hörde.
    Rader som inte hittas får tider mellan grannarna."""
    toks, owner = [], []
    for i, ln in enumerate(lines):
        for w in ln.split():
            if _norm(w):
                toks.append(_norm(w))
                owner.append(i)
    heard = [_norm(w[0]) for w in words]
    times = [[] for _ in lines]
    sm = difflib.SequenceMatcher(None, toks, heard, autojunk=False)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            ws = words[blk.b + k]
            times[owner[blk.a + k]].append((ws[1], ws[2]))
    spans = [(min(t[0] for t in ts), max(t[1] for t in ts)) if ts else None for ts in times]
    # fyll i rader som saknas genom att dela upp tiden mellan kända grannar
    i = 0
    while i < len(spans):
        if spans[i] is None:
            j = i
            while j < len(spans) and spans[j] is None:
                j += 1
            lo = spans[i - 1][1] if i > 0 else (spans[j][0] - 3 * (j - i) if j < len(spans) else 0)
            hi = spans[j][0] if j < len(spans) else lo + 3 * (j - i)
            lo = max(0.0, lo)
            step = max(0.8, (hi - lo) / (j - i))
            for k in range(i, j):
                spans[k] = (lo + (k - i) * step, lo + (k - i + 1) * step)
            i = j
        else:
            i += 1
    caps = [[ln, s, e] for ln, (s, e) in zip(lines, spans)]
    for k in range(len(caps) - 1):   # stå kvar tills nästa rad börjar (högst 1 s glapp)
        caps[k][2] = min(caps[k + 1][1], caps[k][2] + 1.0)
    return caps


def make(song, lyrics="", desc="", style="Filmisk", aspect="9:16", language="sv", subtitles=True,
         moving=False, face=None, progress=None):
    if not song:
        raise ValueError("Välj en låt först.")
    dur = duration(song)
    if dur < 5:
        raise ValueError("Låten är för kort.")
    W, H = FORMATS[aspect]
    work = TMP / f"mv_{int(time.time())}"
    work.mkdir(parents=True, exist_ok=True)

    def step(f, d):
        if progress:
            progress(f, desc=d)

    from autofilm import Part

    # 1. var i låten sjungs vad?
    lines = lyric_lines(lyrics)
    caps = []
    if subtitles or lines:
        step(0.02, "Lyssnar på sången")
        try:
            res = run_worker("words.py", {"files": [song], "language": WHISPER_LANG.get(language, ""),
                                          "label": "Lyssnar på sången"}, Part(progress, 0.02, 0.15, "Text i takt"))
            words = json.load(open(res[-1]))[0]
            if words:
                caps = align(lines, words) if lines else [c for c in _captions(words)]
        except Exception as e:
            print("Kunde inte lyssna in sången – fortsätter utan text i takt:", e, flush=True)
    if lines and not caps:   # sprid raderna jämnt över låten
        span = dur / (len(lines) + 1)
        caps = [[ln, span * (i + 0.5), span * (i + 1.4)] for i, ln in enumerate(lines)]

    # 2. scener: ca en per 6–8 sekunder, med det som sjungs just då som motiv
    n = max(3, min(24, round(dur / 7)))
    seg = dur / n
    base = to_english(desc) if (desc or "").strip() else "atmospheric scene that matches the mood of the song"
    look = STYLES.get(style, "")
    prompts = []
    for i in range(n):
        t0, t1 = i * seg, (i + 1) * seg
        sung = " ".join(c[0] for c in caps if c[1] < t1 and c[2] > t0)[:200]
        p = base + (", " + to_english(sung) if sung else "")
        prompts.append(f"{p}, {look}" if look else p)

    # 3. bilder (en modellstart för alla)
    step(0.15, "Skapar bilder")
    img_aspect = aspect if aspect in ("9:16", "16:9", "1:1") else "9:16"
    args = {"prompts": prompts, "aspect": img_aspect, "style": bool(look) and style != "Filmisk"}
    if face:
        args["face"] = face
    imgs = run_worker("image.py", args, Part(progress, 0.15, 0.45 if moving else 0.7, "Bilder"))
    imgs = [p for p in imgs if p.endswith(".png")][:n]

    # 4. ev. rörliga scener (LTX) – tar ca 1–2 min per scen
    media = list(imgs)
    if moving:
        span = 0.3 / len(imgs)
        for i, img in enumerate(imgs):
            try:
                clip = run_worker("video.py", {"image": img, "prompt": "slow cinematic camera movement, gentle motion",
                                               "aspect": img_aspect, "seconds": min(6, max(3, round(seg))),
                                               "steps": 40, "smooth": True},
                                  Part(progress, 0.45 + i * span, 0.45 + (i + 1) * span,
                                       f"Rörlig scen {i + 1} av {len(imgs)}"))[-1]
                media[i] = clip
            except Exception as e:
                print(f"Rörlig scen {i + 1} misslyckades – använder bilden:", e, flush=True)

    # 5. bildspår med mjuka övertoningar
    n = len(media)
    seg = (dur + X * (n - 1)) / n
    for i, src in enumerate(media):
        step(0.75 + 0.12 * i / n, f"Bygger scen {i + 1} av {n}")
        _scene(src, seg, W, H, i, str(work / f"c{i}.mp4"))

    # 6. ihopsättning: scener + text + låten
    step(0.88, "Sätter ihop musikvideon")
    inputs, fc = [], []
    for i in range(n):
        inputs += ["-i", str(work / f"c{i}.mp4")]
    prev, off = "[0:v]", 0.0
    for i in range(1, n):
        off += seg - X
        fc.append(f"{prev}[{i}:v]xfade=transition=fade:duration={X}:offset={off:.3f}[x{i}]")
        prev = f"[x{i}]"
    idx = n
    if subtitles:
        for k, (txt, s0, s1) in enumerate(caps):
            png = work / f"t{k}.png"
            _subtitle(txt, W, H, png)
            inputs += ["-i", str(png)]
            fc.append(f"{prev}[{idx}:v]overlay=0:0:enable='between(t,{s0:.2f},{s1:.2f})'[o{k}]")
            prev = f"[o{k}]"
            idx += 1
    fc.append(f"{prev}fade=t=in:st=0:d=0.8,fade=t=out:st={max(0, dur - 1.5):.2f}:d=1.5,format=yuv420p[vout]")
    inputs += ["-i", song]
    fc.append(f"[{idx}:a]afade=t=out:st={max(0, dur - 1.5):.2f}:d=1.5,aformat=channel_layouts=stereo[aout]")
    out = new_path("musikvideo", "mp4")
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(fc),
                    "-map", "[vout]", "-map", "[aout]", "-t", f"{dur:.2f}", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "20", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", out], check=True)
    step(1.0, "Klar!")
    return out
