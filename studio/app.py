"""Filmverkstan Studio – gränssnittet. Startas från Colab-notebooken."""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gradio as gr  # noqa: E402

import montage  # noqa: E402
import music  # noqa: E402
import voice  # noqa: E402
from common import OUT, DRIVE_MOUNTED, run_worker, WorkerError  # noqa: E402
from translate import to_english  # noqa: E402

VERSION = "2.0"


def gpu_text():
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"{p.name}, {p.total_memory / 1024**3:.0f} GB"
    except Exception:
        pass
    return "inget grafikkort"


def err(e):
    traceback.print_exc()
    msg = str(e)
    if len(msg) > 900:
        msg = msg[-900:]
    raise gr.Error(msg if isinstance(e, (WorkerError, ValueError)) else f"Något gick fel: {msg}")


# ---------------- funktioner ----------------
def do_images(prompt, aspect, count, seed, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Beskriv bilden först.")
    en = to_english(prompt)
    try:
        paths = run_worker("image.py", {"prompt": en, "aspect": aspect, "count": int(count), "seed": int(seed)}, progress)
    except Exception as e:
        err(e)
    return paths, paths[0] if paths else None, f"Engelsk beskrivning: {en}"


def pick_image(evt: gr.SelectData, paths):
    return paths[evt.index] if paths and evt.index < len(paths) else None


def do_scene(image, prompt, aspect, seconds, steps, seed, progress=gr.Progress()):
    if not image:
        raise gr.Error("Välj en startbild.")
    if not prompt.strip():
        raise gr.Error("Beskriv vad som ska hända.")
    en = to_english(prompt)
    try:
        r = run_worker("video.py", {"image": image, "prompt": en, "aspect": aspect, "seconds": seconds,
                                    "steps": int(steps), "seed": int(seed)}, progress)
    except Exception as e:
        err(e)
    return r[-1], f"Engelsk beskrivning: {en}"


def do_voice(text, engine, ref, style, speed, emotion, progress=gr.Progress()):
    if not text.strip():
        raise gr.Error("Skriv texten som ska läsas upp.")
    try:
        return voice.speak(text, engine, ref, style, speed, emotion, progress)
    except Exception as e:
        err(e)


def do_talk(image, text, audio, engine, ref, enhance, fast, expr=1.5, progress=gr.Progress()):
    if not image:
        raise gr.Error("Välj ett porträtt där ansiktet syns tydligt.")
    try:
        if not audio:
            if not (text or "").strip():
                raise gr.Error("Skriv vad personen ska säga, eller ladda upp en ljudfil.")
            audio = voice.speak(text, engine, ref, progress=progress)
        r = run_worker("talk.py", {"image": image, "audio": audio, "enhance": enhance, "fast": fast, "expression": expr}, progress)
    except gr.Error:
        raise
    except Exception as e:
        err(e)
    return r[-1]


def do_music(secs, mood, nature):
    return music.make(float(secs), mood, nature)


def add_film(path, cur):
    if not path:
        raise gr.Error("Skapa något först.")
    cur = list(cur or []) + [path]
    gr.Info(f"Tillagd i filmen ({len(cur)} scener)")
    return cur, cur


def do_film(files, lines, aspect, engine, ref, speed, mood, nature, subs, music_on, progress=gr.Progress()):
    media = [f if isinstance(f, str) else f.name for f in (files or [])]
    try:
        return montage.build(media, [l.strip() for l in (lines or "").splitlines()], aspect, engine, ref, speed,
                             mood, nature, subs, music_on, progress)
    except Exception as e:
        err(e)


# ---------------- gränssnitt ----------------
TIP = "💡 Skriv gärna på **svenska** – det översätts automatiskt. Beskriv *rörelsen*, t.ex. *hugger ved långsamt, flisor yr, kameran åker närmare*."
with gr.Blocks(title="Filmverkstan Studio") as app:
    film = gr.State([])
    gallery_paths = gr.State([])
    gr.Markdown(f"# 🎬 Filmverkstan Studio <small>v{VERSION}</small>\n"
                f"Grafikkort: **{gpu_text()}** · Resultat sparas i: `{OUT}`"
                + (" (Google Drive ✅)" if DRIVE_MOUNTED else " (försvinner när Colab stängs – ladda ner det du vill spara)"))

    with gr.Tab("1. Bild"):
        with gr.Row():
            with gr.Column():
                i_prompt = gr.Textbox(label="Beskriv bilden", lines=3,
                                      placeholder="En äldre man med långt vitt skägg utanför en timmerstuga i en svensk tallskog, morgonljus")
                with gr.Row():
                    i_aspect = gr.Radio(["9:16", "16:9", "1:1", "4:5"], value="9:16", label="Format")
                    i_count = gr.Slider(1, 4, value=2, step=1, label="Antal bilder")
                i_seed = gr.Number(-1, label="Seed (-1 = slumpa)", precision=0)
                i_btn = gr.Button("Skapa bilder", variant="primary")
                i_info = gr.Markdown()
            with gr.Column():
                i_gal = gr.Gallery(label="Klicka på en bild för att välja den", columns=2, height=520)
                i_sel = gr.Image(label="Vald bild", type="filepath", height=220)
                with gr.Row():
                    i_to_scene = gr.Button("➡ Använd i rörlig scen")
                    i_to_talk = gr.Button("➡ Använd i pratklipp")

    with gr.Tab("2. Rörlig scen"):
        with gr.Row():
            with gr.Column():
                s_img = gr.Image(label="Startbild", type="filepath")
                s_prompt = gr.Textbox(label="Vad ska hända?", lines=3)
                gr.Markdown(TIP)
                with gr.Row():
                    s_aspect = gr.Radio(["9:16", "16:9", "1:1"], value="9:16", label="Format")
                    s_secs = gr.Slider(2, 6, value=3, step=1, label="Sekunder (3 ger skarpast)")
                with gr.Accordion("Avancerat", open=False):
                    s_steps = gr.Slider(20, 50, value=40, step=5, label="Kvalitet (steg)")
                    s_seed = gr.Number(-1, label="Seed", precision=0)
                s_btn = gr.Button("Skapa rörlig scen", variant="primary")
                s_info = gr.Markdown()
            with gr.Column():
                s_out = gr.Video(label="Resultat")
                s_add = gr.Button("➕ Lägg till i filmen")

    with gr.Tab("3. Pratklipp"):
        with gr.Row():
            with gr.Column():
                t_img = gr.Image(label="Porträtt (rakt framifrån, gärna stängd mun – bred tandsmil ger sämre läpprörelser)", type="filepath")
                t_text = gr.Textbox(label="Vad ska personen säga?", lines=5)
                t_audio = gr.Audio(label="…eller ladda upp en egen ljudfil (då används den i stället)", type="filepath")
                with gr.Row():
                    t_engine = gr.Radio(voice.ENGINES, value=voice.ENGINES[0], label="Röst")
                    t_ref = gr.Audio(label="Röst att härma (valfritt, 5–15 sek)", type="filepath")
                with gr.Row():
                    t_enh = gr.Checkbox(True, label="Skärp ansiktet (GFPGAN)")
                    t_fast = gr.Checkbox(False, label="Snabbläge (lägre upplösning, ca 3× snabbare)")
                t_expr = gr.Slider(1.0, 2.5, value=1.5, step=0.1, label="Munrörelser (högre = tydligare läpprörelser)")
                t_btn = gr.Button("Skapa pratklipp", variant="primary")
                gr.Markdown("⏱ Räkna med ca 1 minut per sekund tal (snabbläge: ca 20 sek per sekund).")
            with gr.Column():
                t_out = gr.Video(label="Resultat")
                t_add = gr.Button("➕ Lägg till i filmen")

    with gr.Tab("4. Röst"):
        with gr.Row():
            with gr.Column():
                v_text = gr.Textbox(label="Text att läsa upp", lines=5)
                v_engine = gr.Radio(voice.ENGINES, value=voice.ENGINES[0], label="Röst")
                v_ref = gr.Audio(label="Röst att härma – t.ex. din egen inspelning (valfritt, Naturlig)", type="filepath")
                with gr.Accordion("Inställningar", open=False):
                    v_style = gr.Dropdown(list(voice.PIPER_STYLES), value=list(voice.PIPER_STYLES)[0], label="Röstläge (Snabb)")
                    v_speed = gr.Slider(0.8, 1.2, value=1.0, step=0.05, label="Tempo")
                    v_emo = gr.Slider(0.25, 1.0, value=0.5, step=0.05, label="Inlevelse (Naturlig)")
                v_btn = gr.Button("Skapa röst", variant="primary")
            v_out = gr.Audio(label="Resultat", type="filepath")

    with gr.Tab("5. Musik"):
        with gr.Row():
            with gr.Column():
                m_secs = gr.Slider(5, 180, value=30, step=1, label="Längd (sek)")
                m_mood = gr.Dropdown(list(music.MOODS), value=list(music.MOODS)[0], label="Stämning")
                m_nat = gr.Checkbox(True, label="Naturljud")
                m_btn = gr.Button("Skapa musik", variant="primary")
            m_out = gr.Audio(label="Resultat", type="filepath")

    with gr.Tab("6. Bygg film"):
        gr.Markdown("Lägg in **en bild eller ett klipp per scen** i rätt ordning och skriv **en replik per rad**. "
                    "Lämna raden tom för pratklipp (de har redan ljud) eller tysta scener.")
        with gr.Row():
            with gr.Column():
                f_files = gr.File(label="Scener", file_count="multiple", file_types=["image", "video"])
                f_lines = gr.Textbox(label="Repliker – en rad per scen", lines=6)
                with gr.Row():
                    f_aspect = gr.Radio(list(montage.FORMATS), value="9:16", label="Format")
                    f_subs = gr.Checkbox(True, label="Visa text")
                    f_music = gr.Checkbox(True, label="Musik")
                with gr.Row():
                    f_engine = gr.Radio(voice.ENGINES, value=voice.ENGINES[0], label="Röst")
                    f_ref = gr.Audio(label="Röst att härma (valfritt)", type="filepath")
                with gr.Row():
                    f_speed = gr.Slider(0.8, 1.2, value=1.0, step=0.05, label="Tempo")
                    f_mood = gr.Dropdown(list(music.MOODS), value=list(music.MOODS)[0], label="Musik")
                    f_nat = gr.Checkbox(True, label="Naturljud")
                f_btn = gr.Button("Bygg filmen", variant="primary")
            f_out = gr.Video(label="Färdig film")

    # kopplingar
    i_btn.click(do_images, [i_prompt, i_aspect, i_count, i_seed], [gallery_paths, i_sel, i_info]).then(
        lambda p: p, gallery_paths, i_gal)
    i_gal.select(pick_image, gallery_paths, i_sel)
    i_to_scene.click(lambda p: p, i_sel, s_img)
    i_to_talk.click(lambda p: p, i_sel, t_img)
    s_btn.click(do_scene, [s_img, s_prompt, s_aspect, s_secs, s_steps, s_seed], [s_out, s_info])
    t_btn.click(do_talk, [t_img, t_text, t_audio, t_engine, t_ref, t_enh, t_fast, t_expr], t_out)
    v_btn.click(do_voice, [v_text, v_engine, v_ref, v_style, v_speed, v_emo], v_out)
    m_btn.click(do_music, [m_secs, m_mood, m_nat], m_out)
    s_add.click(add_film, [s_out, film], [film, f_files])
    t_add.click(add_film, [t_out, film], [film, f_files])
    f_btn.click(do_film, [f_files, f_lines, f_aspect, f_engine, f_ref, f_speed, f_mood, f_nat, f_subs, f_music], f_out)

if __name__ == "__main__":
    app.queue(default_concurrency_limit=1).launch(share=True, allowed_paths=[str(OUT), "/content"], debug=True,
                                                 theme=gr.themes.Soft(primary_hue="green"))
