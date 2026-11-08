"""Bild → video med öppna AI-modeller (Wan 2.2 och LTX-Video).

Modellerna laddas ner automatiskt första gången (flera GB) och sparas i mappen 'modeller'.
"""
import gc
from PIL import Image

from . import paths

MODELS = {
    "wan": {
        "label": "Wan 2.2 5B – bäst kvalitet (20 GB+ videominne)",
        "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "fps": 24,
        "steps": 40,
        "guidance": 5.0,
        # (bredd, höjd) – måste vara delbart med 32
        "sizes": {"9:16": (704, 1280), "16:9": (1280, 704), "1:1": (960, 960)},
        "frames": {5: 121, 3: 73},
    },
    "ltx": {
        "label": "LTX-Video – snabb, för mindre grafikkort (8 GB+)",
        "repo": "Lightricks/LTX-Video",
        "fps": 24,
        "steps": 40,
        "guidance": 3.0,
        "sizes": {"9:16": (480, 832), "16:9": (832, 480), "1:1": (640, 640)},
        "frames": {5: 121, 3: 73},
    },
}

NEGATIVE = ("blurry, low quality, distorted face, deformed hands, extra fingers, jittery, flickering, "
            "static image, text, watermark, subtitles, oversaturated")

_pipe = None
_loaded = None


def _load(key: str):
    global _pipe, _loaded
    if _loaded == key:
        return _pipe
    unload()
    import torch
    cfg = MODELS[key]
    cache = str(paths.MODELS / "hf")
    if key == "wan":
        from diffusers import WanImageToVideoPipeline
        pipe = WanImageToVideoPipeline.from_pretrained(cfg["repo"], torch_dtype=torch.bfloat16, cache_dir=cache)
    else:
        from diffusers import LTXImageToVideoPipeline
        pipe = LTXImageToVideoPipeline.from_pretrained(cfg["repo"], torch_dtype=torch.bfloat16, cache_dir=cache)
    # Flyttar delar av modellen till RAM när de inte används → klarar mindre grafikkort
    pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    _pipe, _loaded = pipe, key
    return pipe


def unload():
    """Släpper modellen ur minnet."""
    global _pipe, _loaded
    _pipe, _loaded = None, None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _fit(img: Image.Image, w: int, h: int) -> Image.Image:
    """Beskär bilden så den fyller formatet utan att förvrängas."""
    img = img.convert("RGB")
    scale = max(w / img.width, h / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def generate(image: Image.Image, prompt: str, model: str = "wan", aspect: str = "9:16",
             seconds: int = 5, steps: int | None = None, seed: int = -1, progress=None) -> str:
    import torch
    from diffusers.utils import export_to_video

    cfg = MODELS[model]
    w, h = cfg["sizes"][aspect]
    frames = cfg["frames"].get(seconds, 121)
    steps = steps or cfg["steps"]
    pipe = _load(model)

    gen = None
    if seed is not None and seed >= 0:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))

    kwargs = {}
    if progress is not None:
        def cb(p, i, t, k):
            progress((i + 1) / steps, desc=f"Skapar video – steg {i + 1} av {steps}")
            return k
        kwargs["callback_on_step_end"] = cb

    out = pipe(image=_fit(image, w, h), prompt=prompt, negative_prompt=NEGATIVE,
               width=w, height=h, num_frames=frames, num_inference_steps=steps,
               guidance_scale=cfg["guidance"], generator=gen, **kwargs).frames[0]
    path = paths.new_file("klipp", "mp4")
    export_to_video(out, str(path), fps=cfg["fps"])
    return str(path)
