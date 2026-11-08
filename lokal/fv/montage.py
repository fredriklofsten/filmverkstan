"""Sätter ihop scener (bilder eller videoklipp) med röst, text och musik till en färdig film."""
import os
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from . import paths, tools, voice, music

FORMATS = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080)}
FPS = 30
X = 0.7  # övertoning mellan scener (sekunder)
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeuisb.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/opentype/inter/InterDisplay-Medium.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _font(size):
    for f in FONT_CANDIDATES:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size=size)


def _wrap(text, font, maxw, draw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= maxw:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _subtitle_png(text, W, H, out):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    size = int(min(W, H) * 0.046)
    font = _font(size)
    txt = _wrap(text, font, W * 0.84, d)
    bb = d.multiline_textbbox((0, 0), txt, font=font, align="center", spacing=12)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x0, y0 = (W - tw) // 2, int(H * (0.74 if H > W else 0.80)) - th // 2
    pad = int(size * 0.6)
    d.rounded_rectangle([x0 - pad, y0 + bb[1] - pad, x0 + tw + pad, y0 + bb[1] + th + pad], 24, fill=(0, 0, 0, 120))
    d.multiline_text((x0 - bb[0], y0), txt, font=font, fill="white", align="center", spacing=12)
    img.save(out)


def _scene_clip(src, L, W, H, idx, out):
    ff = tools.ffmpeg()
    n = int(round(L * FPS))
    fill = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
    if Path(src).suffix.lower() in VIDEO_EXT:
        d = tools.duration(src)
        slow = min(1.5, max(1.0, L / max(d, 0.1)))  # sakta ner lite om klippet är kortare än repliken
        vf = f"{fill},setpts={slow:.3f}*PTS,fps={FPS},tpad=stop_mode=clone:stop_duration={L:.2f},format=yuv420p"
        cmd = [ff, "-y", "-loglevel", "error", "-i", src, "-vf", vf, "-frames:v", str(n), "-an"]
    else:
        z = f"1.0+0.12*on/{n}" if idx % 2 == 0 else f"1.12-0.12*on/{n}"
        vf = (f"scale={W*2}:{H*2}:force_original_aspect_ratio=increase,crop={W*2}:{H*2},"
              f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s={W}x{H}:fps={FPS},format=yuv420p")
        cmd = [ff, "-y", "-loglevel", "error", "-loop", "1", "-i", src, "-vf", vf, "-frames:v", str(n)]
    subprocess.run(cmd + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", out], check=True)


def build(media: list, lines: list, aspect="9:16", voice_name="Mansröst (lugn, äldre)", speed=1.0,
          mood="Lugn (skog, natur)", nature=True, subtitles=True, music_on=True, progress=None) -> str:
    if not media:
        raise ValueError("Lägg till minst en bild eller ett videoklipp.")
    lines = (lines + [""] * len(media))[:len(media)]
    W, H = FORMATS[aspect]
    work = paths.OUTPUT / "_tmp"
    work.mkdir(exist_ok=True)

    def step(frac, desc):
        if progress:
            progress(frac, desc=desc)

    # 1. röster
    vfiles, vd = [], []
    for i, line in enumerate(lines):
        step(0.05 + 0.25 * i / len(lines), f"Läser in röst {i + 1} av {len(lines)}")
        if line.strip():
            f = voice.speak(line.strip(), voice_name, speed, out_path=work / f"v{i}.wav")
            vfiles.append(f)
            vd.append(tools.duration(f))
        else:
            vfiles.append(None)
            vd.append(0.0)

    # 2. tider
    clips, vstart, t = [], [], 0.0
    for i, v in enumerate(vd):
        pre = 0.4 if i == 0 else X + 0.1
        c = max(3.0, pre + v + 0.35) + (0.8 if i == len(vd) - 1 else 0)
        clips.append(c)
        vstart.append(t + pre)
        t += c - X
    total = sum(clips) - X * (len(clips) - 1)

    # 3. scener
    for i, (src, c) in enumerate(zip(media, clips)):
        step(0.3 + 0.45 * i / len(media), f"Bygger scen {i + 1} av {len(media)}")
        _scene_clip(src, c, W, H, i, str(work / f"c{i}.mp4"))
        if subtitles and lines[i].strip():
            _subtitle_png(lines[i].strip(), W, H, work / f"s{i}.png")

    # 4. musik
    mfile = None
    if music_on:
        step(0.78, "Skapar musik")
        mfile = music.make(total, mood, nature, out_path=work / "music.wav")

    # 5. ihopsättning
    step(0.85, "Sätter ihop filmen")
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
            s, e = vstart[i] - 0.3, vstart[i] + vd[i] + 0.3
            fc.append(f"{prev}[{idx}:v]overlay=0:0:enable='between(t,{s:.2f},{e:.2f})'[o{i}]")
            prev = f"[o{i}]"
            idx += 1
    fc.append(f"{prev}fade=t=in:st=0:d=0.6,fade=t=out:st={max(0, total - 1):.2f}:d=1,format=yuv420p[vout]")

    alabels = []
    for i, f in enumerate(vfiles):
        if f:
            inputs += ["-i", f]
            ms = int(vstart[i] * 1000)
            fc.append(f"[{idx}:a]adelay={ms}|{ms},aformat=channel_layouts=mono[a{i}]")
            alabels.append(f"[a{i}]")
            idx += 1
    amap = []
    if alabels or mfile:
        if alabels:
            fc.append("".join(alabels) + f"amix=inputs={len(alabels)}:normalize=0,asplit=2[vo][vsc]")
        if mfile:
            inputs += ["-i", mfile]
            fc.append(f"[{idx}:a]volume=0.55[m]")
            if alabels:
                fc.append("[m][vsc]sidechaincompress=threshold=0.05:ratio=4:attack=50:release=600[md]")
                fc.append("[md][vo]amix=inputs=2:normalize=0,loudnorm=I=-14:TP=-1.5,aformat=channel_layouts=stereo[aout]")
            else:
                fc.append("[m]loudnorm=I=-16:TP=-1.5,aformat=channel_layouts=stereo[aout]")
        else:
            fc.append("[vsc]anullsink")
            fc.append("[vo]loudnorm=I=-14:TP=-1.5,aformat=channel_layouts=stereo[aout]")
        amap = ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]

    out = paths.new_file("film", "mp4")
    cmd = [tools.ffmpeg(), "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(fc),
           "-map", "[vout]", *amap, "-t", f"{total:.2f}", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    for f in work.iterdir():
        f.unlink(missing_ok=True)
    step(1.0, "Klar!")
    return str(out)
