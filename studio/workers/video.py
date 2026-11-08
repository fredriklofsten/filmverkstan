"""Bild → rörlig scen med LTX-Video 0.9.5.
Textmodellen och videomodellen turas om på grafikkortet så att allt får plats på ett gratis T4-kort."""
import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wargs, wp, wr, new_path  # noqa

import torch
from PIL import Image

a = wargs()
SIZES = {"9:16": (480, 832), "16:9": (832, 480), "1:1": (640, 640)}
w, h = SIZES.get(a.get("aspect", "9:16"), (480, 832))
secs = float(a.get("seconds", 3))
frames = 8 * round(secs * 24 / 8) + 1
steps = int(a.get("steps", 40))
NEG = ("worst quality, blurry, distorted face, deformed hands, extra fingers, inconsistent motion, "
       "jittery, flickering, static, text, watermark")


def fit(img, w, h):
    img = img.convert("RGB")
    s = max(w / img.width, h / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    l, t = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((l, t, l + w, t + h))


# --- steg 1: texten ---
wp(0.02, "Läser in beskrivningen (första gången laddas textmodellen ner, ca 10 GB)")
from transformers import T5EncoderModel, AutoTokenizer
from diffusers import LTXImageToVideoPipeline, LTXConditionPipeline, FlowMatchEulerDiscreteScheduler
tok = AutoTokenizer.from_pretrained("google-t5/t5-base")  # samma ordlista som LTX, färdigkonverterad
te = T5EncoderModel.from_pretrained("city96/t5-v1_1-xxl-encoder-bf16", torch_dtype=torch.bfloat16,
                                    device_map="cuda", low_cpu_mem_usage=True)
enc = LTXImageToVideoPipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=None, text_encoder=te,
                              tokenizer=tok, transformer=None)
with torch.no_grad():
    pe, pm, ne, nm = enc.encode_prompt(a["prompt"], NEG, do_classifier_free_guidance=True, device="cuda")
pe, ne = pe.half(), ne.half()
del enc, te
gc.collect(); torch.cuda.empty_cache()

# --- steg 2: videon ---
wp(0.15, "Laddar videomodellen")
pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.5", text_encoder=None, tokenizer=None,
                                            torch_dtype=torch.float16).to("cuda")
pipe.vae.enable_tiling()
seed = int(a.get("seed", -1))
g = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None


def cb(p, i, t, k):
    wp(0.2 + 0.75 * (i + 1) / steps, f"Skapar video – steg {i + 1} av {steps}")
    return k


with torch.no_grad():
    out = pipe(image=fit(Image.open(a["image"]), w, h), prompt_embeds=pe, prompt_attention_mask=pm,
               negative_prompt_embeds=ne, negative_prompt_attention_mask=nm, width=w, height=h,
               num_frames=frames, num_inference_steps=steps, guidance_scale=3.0, generator=g,
               callback_on_step_end=cb).frames[0]
from diffusers.utils import export_to_video
path = new_path("scen", "mp4")
export_to_video(out, path, fps=24)
wr(path)
wp(1.0, "Klar")
