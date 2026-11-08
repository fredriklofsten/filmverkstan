"""Små hjälpverktyg: ffmpeg och längd på filer."""
import shutil
import wave


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg  # har med ett eget ffmpeg-program, så du behöver inte installera något
    return imageio_ffmpeg.get_ffmpeg_exe()


def duration(path: str) -> float:
    p = str(path)
    if p.lower().endswith(".wav"):
        with wave.open(p) as w:
            return w.getnframes() / w.getframerate()
    import imageio_ffmpeg
    frames, secs = imageio_ffmpeg.count_frames_and_secs(p)
    return secs
