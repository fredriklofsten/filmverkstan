"""Enkel bakgrundsmusik som skapas med kod (ingen upphovsrätt, inget internet)."""
import wave
import numpy as np

from . import paths

SR = 44100
MOODS = {
    "Lugn (skog, natur)": dict(chords=[[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]], bar=4.0, pluck=True),
    "Varm och hoppfull": dict(chords=[[48, 52, 55], [55, 59, 62], [57, 60, 64], [53, 57, 60]], bar=3.0, pluck=True),
    "Drömsk": dict(chords=[[50, 53, 57, 60], [46, 50, 53, 57], [53, 57, 60, 64], [48, 52, 55, 59]], bar=5.0, pluck=False),
    "Allvarlig": dict(chords=[[45, 48, 52], [41, 45, 48], [43, 47, 50], [40, 44, 47]], bar=4.5, pluck=False),
}


def _note(m):
    return 440 * 2 ** ((m - 69) / 12)


def make(seconds: float, mood: str = "Lugn (skog, natur)", nature: bool = True, seed: int = 3, out_path=None) -> str:
    cfg = MOODS[mood]
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
        wind = np.convolve(rng.standard_normal(n), np.ones(400) / 400, mode="same")
        wind *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t)
        mus += 0.04 * wind / (np.max(np.abs(wind)) + 1e-9)
        for bt in rng.uniform(0.5, max(1.0, seconds - 1), size=max(2, int(seconds / 5))):
            for c in range(3):
                cs, L = int((bt + c * 0.13) * SR), int(0.08 * SR)
                if cs + L >= n:
                    continue
                ct = np.arange(L) / SR
                f = 3800 + 1800 * ct / 0.08
                mus[cs:cs + L] += 0.05 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * ct / 0.08)
    fade = np.ones(n)
    fi, fo = min(n, int(1.5 * SR)), min(n, int(2.5 * SR))
    fade[:fi] = np.linspace(0, 1, fi)
    fade[-fo:] = np.minimum(fade[-fo:], np.linspace(1, 0, fo))
    mus = mus * fade
    mus = mus / (np.max(np.abs(mus)) + 1e-9) * 0.5
    out = out_path or paths.new_file("musik", "wav")
    with wave.open(str(out), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mus * 32767).astype(np.int16).tobytes())
    return str(out)
