"""Filmverkstan – starta med start.bat (Windows) eller start.sh (Linux/Mac)."""
import traceback
import gradio as gr

from fv import __version__, gpu, video_ai, voice, music, montage, paths

GPU = gpu.info()
DEFAULT_MODEL = gpu.recommended_model(GPU["vram_gb"])
MODEL_CHOICES = [(v["label"], k) for k, v in video_ai.MODELS.items()]

PROMPT_TIPS = """**Tips:** Skriv på engelska – modellerna förstår det bäst. Beskriv *rörelsen*, inte bara bilden.
Exempel: *The old man slowly splits a log with an axe, wood chips fly, gentle morning light, camera slowly pushes in.*"""


def _friendly(e: Exception) -> str:
    msg = str(e)
    if "out of memory" in msg.lower():
        return ("Grafikkortet fick slut på minne. Prova LTX-Video, 3 sekunder eller färre steg, "
                "och stäng andra program som använder grafikkortet.")
    if "connection" in msg.lower() or "resolve" in msg.lower():
        return "Kunde inte ladda ner modellen. Kontrollera internetanslutningen och försök igen."
    return f"Något gick fel: {msg}"


# ---------- funktioner som knapparna kör ----------
def do_video(image, prompt, model, aspect, seconds, steps, seed, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Ladda upp en bild först.")
    if not prompt.strip():
        raise gr.Error("Skriv vad som ska hända i videon.")
    if not GPU["cuda"]:
        raise gr.Error("Videogenerering kräver ett Nvidia-grafikkort med CUDA.")
    try:
        progress(0, desc="Laddar modellen (första gången laddas den ner – kan ta en stund)")
        return video_ai.generate(image, prompt, model, aspect, int(seconds), int(steps), int(seed), progress)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(_friendly(e))


def add_to_film(clip, current):
    if not clip:
        raise gr.Error("Skapa ett klipp först.")
    current = list(current or []) + [clip]
    gr.Info(f"Klippet är tillagt i filmen ({len(current)} scener).")
    return current, current


def do_voice(text, v, speed):
    if not text.strip():
        raise gr.Error("Skriv texten som ska läsas upp.")
    try:
        return voice.speak(text, v, speed)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(_friendly(e))


def do_music(seconds, mood, nature):
    return music.make(float(seconds), mood, nature)


def do_film(files, lines, aspect, v, speed, mood, nature, subs, music_on, progress=gr.Progress()):
    media = [f if isinstance(f, str) else f.name for f in (files or [])]
    if not media:
        raise gr.Error("Lägg till bilder eller klipp – en per scen.")
    lst = [l.strip() for l in (lines or "").splitlines()]
    try:
        return montage.build(media, lst, aspect, v, speed, mood, nature, subs, music_on, progress)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(_friendly(e))


# ---------- gränssnittet ----------
with gr.Blocks(title="Filmverkstan", theme=gr.themes.Soft(primary_hue="green")) as app:
    film_state = gr.State([])
    gr.Markdown(f"# 🎬 Filmverkstan <small>v{__version__}</small>\n{gpu.summary()}")

    with gr.Tab("1. Bild → video"):
        with gr.Row():
            with gr.Column():
                in_img = gr.Image(label="Startbild", type="pil")
                in_prompt = gr.Textbox(label="Vad ska hända?", lines=3,
                                       placeholder="The old man slowly picks blueberries into a wooden bowl...")
                gr.Markdown(PROMPT_TIPS)
                with gr.Row():
                    in_aspect = gr.Radio(["9:16", "16:9", "1:1"], value="9:16", label="Format")
                    in_secs = gr.Radio([3, 5], value=5, label="Längd (sek)")
                in_model = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label="AI-modell")
                with gr.Accordion("Avancerat", open=False):
                    in_steps = gr.Slider(15, 60, value=40, step=1, label="Kvalitet (fler steg = bättre men långsammare)")
                    in_seed = gr.Number(value=-1, label="Seed (-1 = slumpa, samma tal = samma resultat)", precision=0)
                btn_video = gr.Button("Skapa video", variant="primary")
            with gr.Column():
                out_clip = gr.Video(label="Resultat")
                btn_add = gr.Button("➕ Lägg till i filmen")
        btn_video.click(do_video, [in_img, in_prompt, in_model, in_aspect, in_secs, in_steps, in_seed], out_clip)

    with gr.Tab("2. Röst"):
        with gr.Row():
            with gr.Column():
                v_text = gr.Textbox(label="Text att läsa upp", lines=4)
                v_voice = gr.Dropdown(list(voice.VOICES), value=list(voice.VOICES)[0], label="Röst")
                v_speed = gr.Slider(0.7, 1.3, value=1.0, step=0.05, label="Tempo")
                btn_voice = gr.Button("Skapa röst", variant="primary")
            v_out = gr.Audio(label="Resultat", type="filepath")
        btn_voice.click(do_voice, [v_text, v_voice, v_speed], v_out)

    with gr.Tab("3. Musik"):
        with gr.Row():
            with gr.Column():
                m_secs = gr.Slider(5, 180, value=30, step=1, label="Längd (sek)")
                m_mood = gr.Dropdown(list(music.MOODS), value=list(music.MOODS)[0], label="Stämning")
                m_nature = gr.Checkbox(True, label="Naturljud (vind och fåglar)")
                btn_music = gr.Button("Skapa musik", variant="primary")
            m_out = gr.Audio(label="Resultat", type="filepath")
        btn_music.click(do_music, [m_secs, m_mood, m_nature], m_out)

    with gr.Tab("4. Bygg film"):
        gr.Markdown("Lägg in **en bild eller ett klipp per scen** i rätt ordning, och skriv **en replik per rad** "
                    "(rad 1 = scen 1 osv.). Lämna en rad tom om scenen ska vara utan röst.")
        with gr.Row():
            with gr.Column():
                f_files = gr.File(label="Scener (bilder och klipp)", file_count="multiple",
                                  file_types=["image", "video"])
                f_lines = gr.Textbox(label="Repliker – en rad per scen", lines=6)
                with gr.Row():
                    f_aspect = gr.Radio(list(montage.FORMATS), value="9:16", label="Format")
                    f_subs = gr.Checkbox(True, label="Visa text")
                    f_music = gr.Checkbox(True, label="Musik")
                with gr.Row():
                    f_voice = gr.Dropdown(list(voice.VOICES), value=list(voice.VOICES)[0], label="Röst")
                    f_speed = gr.Slider(0.7, 1.3, value=1.0, step=0.05, label="Tempo")
                with gr.Row():
                    f_mood = gr.Dropdown(list(music.MOODS), value=list(music.MOODS)[0], label="Musikstämning")
                    f_nature = gr.Checkbox(True, label="Naturljud")
                btn_film = gr.Button("Bygg filmen", variant="primary")
            f_out = gr.Video(label="Färdig film")
        btn_film.click(do_film, [f_files, f_lines, f_aspect, f_voice, f_speed, f_mood, f_nature, f_subs, f_music], f_out)

    btn_add.click(add_to_film, [out_clip, film_state], [film_state, f_files])

    with gr.Tab("Om"):
        gr.Markdown(f"""
**Allt sparas i mappen:** `{paths.OUTPUT}`

**Modeller:** Wan 2.2 (Alibaba) och LTX-Video (Lightricks) – öppna AI-modeller som körs på ditt eget grafikkort.
Första gången du skapar en video laddas modellen ner – det är många GB, så räkna med en stund. Sedan går det snabbare.

**Röst:** Piper (svensk röst, körs lokalt). **Musik:** skapas med kod, fri att använda.

Grafikkort: {GPU['name']} ({GPU['vram_gb']} GB)
""")

if __name__ == "__main__":
    app.queue().launch(inbrowser=True)
