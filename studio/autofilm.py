"""Ett klick: manus → bilder → rörliga scener / pratklipp → färdig film.

Varje steg använder samma arbetare som flikarna, så förbättringar där gäller även här.
Allt som skapas sparas också som egna filer, så du kan byta ut en enskild scen efteråt.
"""
import json
import re

import montage
import voice
from common import run_worker
from translate import to_english

TYPES = ("prat", "scen")


class Part:
    """Gör om ett delsteg (0–1) till en bit av hela förloppet."""

    def __init__(self, progress, lo, hi, label):
        self.p, self.lo, self.hi, self.label = progress, lo, hi, label

    def __call__(self, frac, desc=""):
        if self.p:
            self.p(self.lo + (self.hi - self.lo) * max(0.0, min(1.0, float(frac))),
                   desc=f"{self.label} · {desc}" if desc else self.label)


def write_script(idea, person, scenes, talk, progress=None):
    if not (idea or "").strip():
        raise ValueError("Skriv en idé till filmen först.")
    r = run_worker("script.py", {"idea": idea.strip(), "person": (person or "").strip(),
                                 "scenes": int(scenes), "talk": talk}, progress)
    return json.load(open(r[-1]))


def clean_rows(rows):
    out = []
    for row in rows or []:
        row = [("" if c is None else str(c)).strip() for c in list(row) + [""] * 4][:4]
        if not any(row):
            continue
        typ = "prat" if row[0].lower().startswith("p") else "scen"
        out.append([typ, row[1], row[2], row[3]])
    if not out:
        raise ValueError("Manuset är tomt – skriv minst en scen.")
    return out


def scene_seconds(line):
    """Ungefär hur länge repliken tar att läsa (ca 2,5 ord/s), 3–8 sekunder."""
    words = len(re.findall(r"\w+", line or ""))
    return max(3, min(8, round(words / 2.5 + 1)))


def make_film(rows, person, face, aspect, engine, ref, lipsync, mood, nature, subtitles, music_on,
              progress=None):
    rows = clean_rows(rows)
    n = len(rows)
    img_aspect = aspect if aspect in ("9:16", "16:9", "1:1") else "9:16"
    person_en = to_english(person) if (person or "").strip() else ""

    # 1. huvudpersonen: uppladdat ansikte, annars ett porträtt som blir förlaga för resten
    if not face:
        pr = f"close-up portrait of {person_en}, looking into the camera, calm natural expression, mouth closed" \
            if person_en else "close-up portrait of the main character, looking into the camera, mouth closed"
        face = run_worker("image.py", {"prompt": pr, "aspect": "9:16", "count": 1},
                          Part(progress, 0.0, 0.05, "Skapar huvudpersonen"))[-1]

    media, lines = [], []
    span = 0.85 / n
    for i, (typ, bild, rorelse, replik) in enumerate(rows):
        lo = 0.05 + i * span
        label = f"Scen {i + 1} av {n}"
        desc = to_english(bild) if bild else "the main character in a natural setting"
        if typ == "prat":
            desc += ", close-up portrait looking straight into the camera, mouth closed, calm expression"
        if person_en:
            desc = f"{person_en}, {desc}"
        img = run_worker("image.py", {"prompt": desc, "aspect": img_aspect, "count": 1, "face": face},
                         Part(progress, lo, lo + span * 0.15, label + " – bild"))[-1]
        if typ == "prat" and replik:
            audio = voice.speak(replik, engine, ref, progress=Part(progress, lo + span * 0.15, lo + span * 0.3,
                                                                   label + " – röst"))
            clip = run_worker("talk.py", {"image": img, "audio": audio, "enhance": True, "fast": False,
                                          "expression": 1.5, "lipsync": "latentsync" if lipsync else ""},
                              Part(progress, lo + span * 0.3, lo + span, label + " – pratklipp"))[-1]
            media.append(clip); lines.append("")      # klippet har redan ljud
        else:
            move = to_english(rorelse) if rorelse else "subtle natural movement, gentle camera push-in"
            clip = run_worker("video.py", {"image": img, "prompt": move, "aspect": img_aspect,
                                           "seconds": scene_seconds(replik), "steps": 40, "smooth": True},
                              Part(progress, lo + span * 0.15, lo + span, label + " – rörlig scen"))[-1]
            media.append(clip); lines.append(replik)

    return montage.build(media, lines, aspect, engine, ref, 1.0, mood, nature, subtitles, music_on,
                         Part(progress, 0.9, 1.0, "Klipper ihop filmen"))
