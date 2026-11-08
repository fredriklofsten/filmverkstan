"""Text → bild med RealVisXL Lightning (fotorealistisk, snabb)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wargs, wp, wr, new_path  # noqa

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler

a = wargs()
SIZES = {"9:16": (768, 1344), "16:9": (1344, 768), "1:1": (1024, 1024), "4:5": (896, 1120)}
w, h = SIZES.get(a.get("aspect", "9:16"), (768, 1344))
steps = int(a.get("steps", 6))

wp(0.02, "Laddar bildmodellen")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V5.0_Lightning", vae=vae,
                                                 torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++",
                                                         use_karras_sigmas=True)
pipe.enable_vae_slicing()

neg = ("cartoon, painting, illustration, 3d render, deformed, distorted face, bad hands, extra fingers, "
       "blurry, low quality, text, watermark, logo")
prompt = a["prompt"].strip()
if "photo" not in prompt.lower():
    prompt = "RAW photo, photorealistic, " + prompt
count = int(a.get("count", 1))
seed = int(a.get("seed", -1))
for i in range(count):
    s = seed + i if seed >= 0 else int.from_bytes(os.urandom(4), "little")
    g = torch.Generator("cuda").manual_seed(s)
    def cb(p, step, t, k, i=i):
        wp((i + (step + 1) / steps) / count * 0.98, f"Bild {i + 1} av {count}")
        return k
    img = pipe(prompt=prompt, negative_prompt=neg, width=w, height=h, num_inference_steps=steps,
               guidance_scale=float(a.get("guidance", 1.5)), generator=g, callback_on_step_end=cb).images[0]
    path = new_path("bild", "png")
    img.save(path)
    wr(path)
wp(1.0, "Klar")
