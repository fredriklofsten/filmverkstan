"""Text → bild med RealVisXL Lightning (fotorealistisk, snabb).
Med "face" används IP-Adapter Plus Face så att samma person återkommer i alla bilder."""
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
try:
    pipe.vae.enable_slicing()
except Exception:
    pass


def face_crop(img):
    """Skär ut ansiktet (med lite marginal) så att ansiktsmodellen ser det tydligt."""
    import numpy as np
    img = img.convert("RGB")
    try:
        from facexlib.detection import init_detection_model
        det = init_detection_model("retinaface_resnet50", half=False, device="cuda")
        with torch.no_grad():
            boxes = det.detect_faces(np.array(img)[:, :, ::-1], 0.6)
        del det
        if len(boxes):
            x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))[:4]
            cx, cy, r = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 0.8
            return img.crop((int(max(0, cx - r)), int(max(0, cy - r)),
                             int(min(img.width, cx + r)), int(min(img.height, cy + r))))
    except Exception as e:
        print("ansiktsdetektering misslyckades, använder övre mitten:", e, flush=True)
    side = min(img.width, img.height)
    left = (img.width - side) // 2
    return img.crop((left, 0, left + side, side))


face = None
if a.get("face"):
    wp(0.05, "Lär in huvudpersonens ansikte")
    from PIL import Image
    face = face_crop(Image.open(a["face"]))
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                         weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
                         image_encoder_folder="models/image_encoder")
    pipe.set_ip_adapter_scale(float(a.get("likeness", 0.6)))

neg = ("cartoon, painting, illustration, 3d render, deformed, distorted face, bad hands, extra fingers, "
       "blurry, low quality, text, watermark, logo")
if a.get("style"):   # konstnärlig stil (t.ex. målning, anime): förbjud inte målningar/illustrationer
    neg = "deformed, distorted face, bad hands, extra fingers, blurry, low quality, text, watermark, logo"
def fix(p):
    p = p.strip()
    return p if "photo" in p.lower() or a.get("style") else "RAW photo, photorealistic, " + p


# "prompts": en lista = en bild per beskrivning (t.ex. musikvideo), annars "count" bilder av samma beskrivning
prompts = [fix(p) for p in a["prompts"]] if a.get("prompts") else [fix(a["prompt"])] * int(a.get("count", 1))
count = len(prompts)
seed = int(a.get("seed", -1))
for i in range(count):
    s = seed + i if seed >= 0 else int.from_bytes(os.urandom(4), "little")
    g = torch.Generator("cuda").manual_seed(s)
    def cb(p, step, t, k, i=i):
        wp((i + (step + 1) / steps) / count * 0.98, f"Bild {i + 1} av {count}")
        return k
    extra = {"ip_adapter_image": face} if face is not None else {}
    img = pipe(prompt=prompts[i], negative_prompt=neg, width=w, height=h, num_inference_steps=steps,
               guidance_scale=float(a.get("guidance", 1.5)), generator=g, callback_on_step_end=cb,
               **extra).images[0]
    path = new_path("bild", "png")
    img.save(path)
    wr(path)
wp(1.0, "Klar")
