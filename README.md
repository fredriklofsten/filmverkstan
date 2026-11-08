# 🎬 Filmverkstan

Gratis AI-filmstudio som körs på Googles grafikkort i Colab.

**[▶ Öppna Filmverkstan Studio i Colab](https://colab.research.google.com/github/fredriklofsten/filmverkstan/blob/main/Filmverkstan_Studio.ipynb)**

| Flik | Vad den gör | AI-modell |
|---|---|---|
| 1. Bild | Fotorealistiska bilder från en beskrivning | RealVisXL V5 Lightning |
| 2. Rörlig scen | Gör en bild till en 2–6 sekunders videoscen | LTX-Video 0.9.5 |
| 3. Pratklipp | Får en person på en bild att prata | SadTalker + GFPGAN |
| 4. Röst | Svensk röst – naturlig eller snabb, kan härma en inspelning | Chatterbox Multilingual / Piper |
| 5. Musik | Lugn bakgrundsmusik med naturljud | (skapas med kod) |
| 6. Bygg film | Sätter ihop allt med röst, text och musik | ffmpeg |

Beskrivningar kan skrivas på svenska – de översätts automatiskt (Helsinki-NLP opus-mt-sv-en).

## Struktur
- `Filmverkstan_Studio.ipynb` – startar allt i Colab (hämtar alltid senaste koden härifrån)
- `studio/app.py` – gränssnittet (Gradio)
- `studio/setup_env.py` – installation, inkl. rättningar så att SadTalker/basicsr fungerar med nya Python
- `studio/workers/` – varje AI-modell körs i en egen process så att minnet frigörs efteråt

## Licenser
Modellerna har egna licenser (LTX-Video: Lightricks open weights-licens, RealVisXL: OpenRAIL++, SadTalker: Apache 2.0, Chatterbox: MIT, Piper-röst: se rhasspy/piper-voices). Läs dem innan du använder resultat kommersiellt.
