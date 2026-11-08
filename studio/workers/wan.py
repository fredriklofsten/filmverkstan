"""Bild → rörlig scen med Wan 2.2 TI2V-5B (Alibaba, Apache-2.0). Snyggare rörelser än LTX men långsammare.
Textmodellen (UMT5-XXL) och videomodellen turas om på grafikkortet (på Kaggle ligger textmodellen på kort 2).
Scener över 5 sekunder byggs i delar där varje del börjar på sista bilden i förra delen."""
import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wargs, wp, wr, new_path, finish_scene  # noqa

import torch
from PIL import Image

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
a = wargs()
SIZES = {"9:16": (480, 832), "16:9": (832, 480), "1:1": (640, 640)}   # (bredd, höjd), delbart med 32
w, h = SIZES.get(a.get("aspect", "9:16"), (480, 832))
secs = max(1.0, float(a.get("seconds", 3)))
steps = int(a.get("steps", 30))
MAXF = 121                                    # 5 s per del
total = 4 * round(secs * 24 / 4) + 1
parts = [min(MAXF, total)]
left = total - parts[0]
while left > 0:
    nf = min(MAXF, 4 * -(-left // 4) + 1)   # 4k+1 bildrutor; första bilden = förra delens sista
    parts.append(nf)
    left -= nf - 1
NEG = ("bright colors, overexposed, static, blurred details, subtitles, text, watermark, worst quality, low quality, "
       "JPEG artifacts, ugly, deformed, disfigured, extra fingers, poorly drawn hands, poorly drawn face, "
       "fused fingers, still picture, messy background, walking backwards")


def fit(img):
    img = img.convert("RGB")
    s = max(w / img.width, h / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    l, t = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((l, t, l + w, t + h))


from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, AutoencoderKLWan, UniPCMultistepScheduler

# --- steg 1: texten ---
wp(0.02, "Läser in beskrivningen (första gången laddas Wan-modellerna ner, ca 20 GB)")
TE_DEV = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
tok = AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer")
te = UMT5EncoderModel.from_pretrained(REPO, subfolder="text_encoder", torch_dtype=torch.bfloat16,
                                      device_map=TE_DEV, low_cpu_mem_usage=True)
enc = WanImageToVideoPipeline(tokenizer=tok, text_encoder=te, vae=None, scheduler=None, transformer=None)
with torch.no_grad():
    pe, ne = enc.encode_prompt(a["prompt"], NEG, do_classifier_free_guidance=True, device=TE_DEV)
pe, ne = pe.half().to("cuda:0"), ne.half().to("cuda:0")
del enc, te
gc.collect(); torch.cuda.empty_cache()

# --- steg 2: videon ---
wp(0.12, "Laddar Wan 2.2")
transformer = WanTransformer3DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=torch.float16,
                                                    device_map="cuda:0", low_cpu_mem_usage=True)
vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).to("cuda:0")
try:
    vae.enable_tiling()
except Exception:
    pass
sched = UniPCMultistepScheduler.from_pretrained(REPO, subfolder="scheduler")
pipe = WanImageToVideoPipeline(tokenizer=None, text_encoder=None, vae=vae, scheduler=sched, transformer=transformer,
                               expand_timesteps=True)
seed = int(a.get("seed", -1))
g = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None

frames_out, img = [], fit(Image.open(a["image"]))
for k, nf in enumerate(parts):
    def cb(p, i, t, kw, k=k):
        dl = f" (del {k + 1} av {len(parts)})" if len(parts) > 1 else ""
        wp(0.15 + 0.78 * (k + (i + 1) / steps) / len(parts), f"Skapar video med Wan – steg {i + 1} av {steps}{dl}")
        return kw
    with torch.no_grad():
        out = pipe(image=img, prompt_embeds=pe, negative_prompt_embeds=ne, height=h, width=w, num_frames=nf,
                   num_inference_steps=steps, guidance_scale=5.0, generator=g, callback_on_step_end=cb,
                   output_type="pil").frames[0]
    out = list(out)
    frames_out += out if k == 0 else out[1:]   # nästa del börjar på förra delens sista bild
    img = out[-1]
    gc.collect(); torch.cuda.empty_cache()

from diffusers.utils import export_to_video
raw = new_path("scen_ra", "mp4")
export_to_video(frames_out, raw, fps=24)
del pipe, transformer, vae
gc.collect(); torch.cuda.empty_cache()
path = finish_scene(raw, a.get("smooth", True), a.get("sfx"), wp)
wr(path)
wp(1.0, "Klar")
