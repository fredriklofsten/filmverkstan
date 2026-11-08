"""Idé → manus. En liten språkmodell (Qwen3 4B) skriver scenerna som JSON."""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wargs, wp, wr, new_path  # noqa

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

a = wargs()
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
n = int(a.get("scenes", 5))
talk_share = a.get("talk", "Blandat")

wp(0.05, "Laddar manusförfattaren (första gången ca 8 GB)")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda:0")

rule = {"Mest prat": "De flesta scener ska vara typ \"prat\".",
        "Mest berättarröst": "De flesta scener ska vara typ \"scen\" med berättarröst.",
        }.get(talk_share, "Blanda: ungefär hälften \"prat\" och hälften \"scen\". Börja gärna med \"prat\".")
system = ("Du är en prisbelönt manusförfattare för korta vertikala filmer (Reels/TikTok). "
          "Du svarar ENDAST med giltig JSON, utan förklaringar.")
user = f"""Skriv ett manus på svenska till en kort film med exakt {n} scener.

Idé: {a["idea"]}
Huvudperson: {a.get("person") or "välj en passande huvudperson"}

Varje scen är ett JSON-objekt med fälten:
- "typ": "prat" (huvudpersonen tittar in i kameran och säger repliken) eller "scen" (en rörlig bild medan en berättarröst läser repliken)
- "bild": kort beskrivning av bilden på svenska (miljö, ljus, vad huvudpersonen gör). Vid "prat": nära porträtt rakt framifrån.
- "rorelse": vad som rör sig i bilden, på svenska (t.ex. "han går långsamt mellan träden, löven rör sig i vinden"). Tom vid "prat".
- "replik": det som sägs, 1–2 naturliga meningar (högst 25 ord), i jag-form vid "prat".
{rule}
Filmen ska ha en tydlig början, mitt och ett starkt slut. Svara med en JSON-lista med {n} objekt."""

msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
if isinstance(ids, dict) or hasattr(ids, "input_ids"):
    ids = ids["input_ids"]
ids = ids.to("cuda:0")
wp(0.4, "Skriver manus")
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=1500, do_sample=True, temperature=0.7, top_p=0.9)
text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

m = re.search(r"\[.*\]", text, re.S)
if not m:
    print("Kunde inte läsa manuset:\n" + text[:800]); sys.exit(1)
scenes = json.loads(m.group(0))
rows = []
for s in scenes[:n]:
    typ = "prat" if str(s.get("typ", "")).lower().startswith("p") else "scen"
    rows.append([typ, str(s.get("bild", "")).strip(), str(s.get("rorelse", "")).strip() if typ == "scen" else "",
                 str(s.get("replik", "")).strip()])
path = new_path("manus", "json")
json.dump(rows, open(path, "w"), ensure_ascii=False, indent=1)
wr(path)
wp(1.0, "Klar")
