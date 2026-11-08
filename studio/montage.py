"""Sätter ihop scener (bilder, rörliga scener, pratklipp) med röst, text och musik till en färdig film."""
import os
import subprocess
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import music
import voice
from common import TMP, new_path, ffmpeg, duration, has_audio

FORMATS = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080)}
FPS = 30
X = 0.6  # övertoning mellan scener (sekunder)
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
FONTS = ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def _font(size):
    for f in FONTS:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size=size)


def _wrap(text, font, maxw, d):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= maxw:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _subtitle(text, W, H, out):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    size = int(min(W, H) * 0.046)
    font = _font(size)
    txt = _wrap(text, font, W * 0.84, d)
    bb = d.multiline_textbbox((0, 0), txt, font=font, align="center", spacing=12)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x0, y0 = (W - tw) // 2, int(H * (0.76 if H > W else 0.80)) - th // 2
    pad = int(size * 0.6)
    d.rounded_rectangle([x0 - pad, y0 + bb[1] - pad, x0 + tw + pad, y0 + bb[1] + th + pad], 24, fill=(0, 0, 0, 120))
    d.multiline_text((x0 - bb[0], y0), txt, font=font, fill="white", align="center", spacing=12)
    img.save(out)


def _scene(src, L, W, H, idx, out, slow_ok=True):
    n = int(round(L * FPS))
    fill = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
    if Path(src).suffix.lower() in VIDEO_EXT:
        d = duration(src)
        slow = min(1.5, max(1.0, L / max(d, 0.1))) if slow_ok else 1.0
        vf = f"{fill},setpts={slow:.3f}*PTS,fps={FPS},tpad=stop_mode=clone:stop_duration={L:.2f},format=yuv420p"
        cmd = [ffmpeg(), "-y", "-loglevel", "error", "-i", src, "-vf", vf, "-frames:v", str(n), "-an"]
    else:
        z = f"1.0+0.12*on/{n}" if idx % 2 == 0 else f"1.12-0.12*on/{n}"
        vf = (f"scale={W*2}:{H*2}:force_original_aspect_ratio=increase,crop={W*2}:{H*2},"
              f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s={W}x{H}:fps={FPS},format=yuv420p")
        cmd = [ffmpeg(), "-y", "-loglevel", "error", "-loop", "1", "-i", src, "-vf", vf, "-frames:v", str(n)]
    subprocess.run(cmd + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", out], check=True)


def build(media, lines, aspect="9:16", engine=voice.ENGINES[0], ref=None, speed=1.0, mood="Lugn (skog, natur)",
          nature=True, subtitles=True, music_on=True, progress=None):
    if not media:
        raise ValueError("Lägg till minst en bild eller ett klipp.")
    lines = (lines + [""] * len(media))[:len(media)]
    W, H = FORMATS[aspect]
    work = TMP / f"film_{int(time.time())}"
    work.mkdir(parents=True, exist_ok=True)

    def step(f, d):
        if progress:
            progress(f, desc=d)

    # 1. ljud per scen: egen replik → röst; pratklipp utan replik → klippets eget ljud
    auds, adur, talk = [], [], []
    for i, (src, line) in enumerate(zip(media, lines)):
        step(0.02 + 0.35 * i / len(media), f"Röst för scen {i + 1} av {len(media)}")
        is_vid = Path(src).suffix.lower() in VIDEO_EXT
        if line.strip():
            f = voice.speak(line.strip(), engine, ref, speed=speed, out=str(work / f"v{i}.wav"))
            auds.append(f); adur.append(duration(f)); talk.append(False)
        elif is_vid and has_audio(src):
            f = str(work / f"v{i}.wav")
            subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", src, "-vn", "-ar", "44100", f], check=True)
            auds.append(f); adur.append(duration(src)); talk.append(True)
        else:
            auds.append(None); adur.append(0.0); talk.append(False)

    # 2. tider
    clips, astart, t = [], [], 0.0
    for i in range(len(media)):
        last = i == len(media) - 1
        if talk[i]:
            pre = 0.0
            c = adur[i] + (0.5 if last else X)
        else:
            pre = 0.4 if i == 0 else X + 0.1
            c = max(3.0, pre + adur[i] + 0.35) + (0.8 if last else 0)
        clips.append(c)
        astart.append(t + pre)
        t += c - X
    total = sum(clips) - X * (len(clips) - 1)

    # 3. bildspår
    for i, (src, c) in enumerate(zip(media, clips)):
        step(0.38 + 0.4 * i / len(media), f"Bygger scen {i + 1} av {len(media)}")
        _scene(src, c, W, H, i, str(work / f"c{i}.mp4"), slow_ok=not talk[i])
        if subtitles and lines[i].strip():
            _subtitle(lines[i].strip(), W, H, work / f"s{i}.png")

    mfile = None
    if music_on:
        step(0.8, "Skapar musik")
        mfile = music.make(total, mood, nature, out_path=str(work / "music.wav"))

    # 4. ihopsättning
    step(0.86, "Sätter ihop filmen")
    inputs, fc = [], []
    for i in range(len(media)):
        inputs += ["-i", str(work / f"c{i}.mp4")]
    idx = len(media)
    prev, off = "[0:v]", 0.0
    for i in range(1, len(media)):
        off += clips[i - 1] - X
        fc.append(f"{prev}[{i}:v]xfade=transition=fade:duration={X}:offset={off:.3f}[x{i}]")
        prev = f"[x{i}]"
    for i in range(len(media)):
        if subtitles and lines[i].strip():
            inputs += ["-i", str(work / f"s{i}.png")]
            s, e = astart[i] - 0.3, astart[i] + adur[i] + 0.3
            fc.append(f"{prev}[{idx}:v]overlay=0:0:enable='between(t,{s:.2f},{e:.2f})'[o{i}]")
            prev = f"[o{i}]"
            idx += 1
    fc.append(f"{prev}fade=t=in:st=0:d=0.5,fade=t=out:st={max(0, total - 1):.2f}:d=1,format=yuv420p[vout]")
    al = []
    for i, f in enumerate(auds):
        if f:
            inputs += ["-i", f]
            ms = int(astart[i] * 1000)
            fc.append(f"[{idx}:a]aformat=sample_rates=44100:channel_layouts=mono,adelay={ms}|{ms}[a{i}]")
            al.append(f"[a{i}]")
            idx += 1
    amap = []
    if al or mfile:
        if al:
            fc.append("".join(al) + f"amix=inputs={len(al)}:normalize=0,asplit=2[vo][vsc]")
        if mfile:
            inputs += ["-i", mfile]
            fc.append(f"[{idx}:a]volume=0.5[m]")
            if al:
                fc.append("[m][vsc]sidechaincompress=threshold=0.05:ratio=4:attack=50:release=600[md]")
                fc.append("[md][vo]amix=inputs=2:normalize=0,loudnorm=I=-14:TP=-1.5,aformat=channel_layouts=stereo[aout]")
            else:
                fc.append("[m]loudnorm=I=-16:TP=-1.5,aformat=channel_layouts=stereo[aout]")
        else:
            fc.append("[vsc]anullsink")
            fc.append("[vo]loudnorm=I=-14:TP=-1.5,aformat=channel_layouts=stereo[aout]")
        amap = ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    out = new_path("film", "mp4")
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(fc), "-map", "[vout]",
                    *amap, "-t", f"{total:.2f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-movflags", "+faststart", out], check=True)
    step(1.0, "Klar!")
    return out
