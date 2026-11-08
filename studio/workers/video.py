"""Bild → rörlig scen med LTX-Video 0.9.5.
Textmodellen och videomodellen turas om på grafikkortet så att allt får plats på ett gratis T4-kort.
Längre scener byggs i delar om ca 3 sekunder: varje ny del fortsätter från de sista bildrutorna i
förra delen. Efteråt kan rörelsen mjukas upp (60 bilder/s) och bilden skärpas."""
import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wargs, wp, wr, new_path  # noqa

import torch
from PIL import Image

a = wargs()
SIZES = {"9:16": (480, 832), "16:9": (832, 480), "1:1": (640, 640)}
w, h = SIZES.get(a.get("aspect", "9:16"), (480, 832))
secs = max(1.0, float(a.get("seconds", 3)))
SEG = 81            # bildrutor per del (ca 3,4 s) – det T4-kortet klarar bäst
OVERLAP = 9         # bildrutor som nästa del fortsätter från
total = 8 * round(secs * 24 / 8) + 1
if total <= SEG:
    parts = [total]
else:
    parts, left = [SEG], total - SEG
    while left > 0:
        n = min(SEG, 8 * ((left + OVERLAP - 1) // 8 + 1) + 1)  # 8k+1 bildrutor, inkl. överlapp
        parts.append(n)
        left -= n - OVERLAP
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
TE_DEV = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"  # två kort (Kaggle): textmodellen på kort 2
tok = AutoTokenizer.from_pretrained("google-t5/t5-base")  # samma ordlista som LTX, färdigkonverterad
te = T5EncoderModel.from_pretrained("city96/t5-v1_1-xxl-encoder-bf16", torch_dtype=torch.bfloat16,
                                    device_map=TE_DEV, low_cpu_mem_usage=True)
enc = LTXImageToVideoPipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=None, text_encoder=te,
                              tokenizer=tok, transformer=None)
with torch.no_grad():
    pe, pm, ne, nm = enc.encode_prompt(a["prompt"], NEG, do_classifier_free_guidance=True, device=TE_DEV)
pe, ne = pe.half().to("cuda:0"), ne.half().to("cuda:0")
pm, nm = pm.to("cuda:0"), nm.to("cuda:0")
del enc, te
gc.collect(); torch.cuda.empty_cache()

# --- steg 2: videon ---
wp(0.15, "Laddar videomodellen")
pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.5", text_encoder=None, tokenizer=None,
                                            torch_dtype=torch.float16).to("cuda")
pipe.vae.enable_tiling()
seed = int(a.get("seed", -1))
g = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None


from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
SPAN = 0.75 if a.get("smooth", True) else 0.8
frames_out = []
for n_part, nf in enumerate(parts):
    def cb(p, i, t, k, n_part=n_part):
        done = (n_part + (i + 1) / steps) / len(parts)
        del_txt = f" (del {n_part + 1} av {len(parts)})" if len(parts) > 1 else ""
        wp(0.2 + SPAN * done, f"Skapar video – steg {i + 1} av {steps}{del_txt}")
        return k
    common_kw = dict(prompt_embeds=pe, prompt_attention_mask=pm, negative_prompt_embeds=ne,
                     negative_prompt_attention_mask=nm, width=w, height=h, num_frames=nf,
                     num_inference_steps=steps, guidance_scale=3.0, generator=g, callback_on_step_end=cb)
    with torch.no_grad():
        if n_part == 0:
            out = pipe(image=fit(Image.open(a["image"]), w, h), **common_kw).frames[0]
            frames_out += list(out)
        else:
            cond = LTXVideoCondition(video=frames_out[-OVERLAP:], frame_index=0)
            out = pipe(conditions=[cond], **common_kw).frames[0]
            frames_out += list(out)[OVERLAP:]
    gc.collect(); torch.cuda.empty_cache()

from diffusers.utils import export_to_video
raw = new_path("scen_ra", "mp4")
export_to_video(frames_out, raw, fps=24)
path = raw
if a.get("smooth", True):
    # mjukare rörelse (60 bilder/s, rörelsekompenserad) + 1,5× större och lite skarpare
    wp(0.96, "Mjukar upp rörelsen och skärper bilden")
    import subprocess
    from common import ffmpeg
    path = new_path("scen", "mp4")
    vf = ("minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
          "scale=trunc(iw*1.5/2)*2:trunc(ih*1.5/2)*2:flags=lanczos,unsharp=5:5:0.5")
    r = subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", raw, "-vf", vf, "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", path])
    if r.returncode == 0:
        os.remove(raw)
    else:
        path = raw
wr(path)
wp(1.0, "Klar")
