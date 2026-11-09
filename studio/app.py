"""Filmverkstan Studio – gränssnittet. Startas från Colab- eller Kaggle-notebooken."""
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gradio as gr  # noqa: E402

import autofilm  # noqa: E402
import montage  # noqa: E402
import music  # noqa: E402
import voice  # noqa: E402
from common import OUT, DRIVE_MOUNTED, KAGGLE, run_worker, WorkerError  # noqa: E402
from translate import to_english  # noqa: E402

VERSION = "2.3"


def gpu_text():
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            n = torch.cuda.device_count()
            return (f"{n} × " if n > 1 else "") + f"{p.name}, {p.total_memory / 1024**3:.0f} GB"
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
def do_images(prompt, aspect, count, seed, face=None, likeness=0.6, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Beskriv bilden först.")
    en = to_english(prompt)
    args = {"prompt": en, "aspect": aspect, "count": int(count), "seed": int(seed)}
    if face:
        args.update(face=face, likeness=float(likeness))
    try:
        paths = run_worker("image.py", args, progress)
    except Exception as e:
        err(e)
    return paths, paths[0] if paths else None, f"Engelsk beskrivning: {en}"


def pick_image(evt: gr.SelectData, paths):
    return paths[evt.index] if paths and evt.index < len(paths) else None


VIDEO_MODELS = ["LTX (snabb)", "Wan 2.2 (snyggare, ca 3× långsammare)"]


def scene_args(image, en, aspect, seconds, steps, seed, smooth, sfx, model):
    wan = model.startswith("Wan")
    args = {"image": image, "prompt": en, "aspect": aspect, "seconds": seconds,
            "steps": int(steps) if not wan else min(int(steps), 30), "seed": int(seed), "smooth": bool(smooth)}
    if sfx:
        args["sfx"] = en
    return ("wan.py" if wan else "video.py"), args


def do_scene(image, prompt, aspect, seconds, steps, seed, smooth=True, sfx=True, model=VIDEO_MODELS[0],
             progress=gr.Progress()):
    if not image:
        raise gr.Error("Välj en startbild.")
    if not prompt.strip():
        raise gr.Error("Beskriv vad som ska hända.")
    en = to_english(prompt)
    try:
        r = run_worker(*scene_args(image, en, aspect, seconds, steps, seed, smooth, sfx, model), progress)
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


def do_talk(image, text, audio, engine, ref, enhance, fast, expr=1.5, better_lips=True, progress=gr.Progress()):
    if not image:
        raise gr.Error("Välj ett porträtt där ansiktet syns tydligt.")
    try:
        if not audio:
            if not (text or "").strip():
                raise gr.Error("Skriv vad personen ska säga, eller ladda upp en ljudfil.")
            audio = voice.speak(text, engine, ref, progress=progress)
        r = run_worker("talk.py", {"image": image, "audio": audio, "enhance": enhance, "fast": fast, "expression": expr,
                                   "lipsync": "latentsync" if better_lips else ""}, progress)
    except gr.Error:
        raise
    except Exception as e:
        err(e)
    return r[-1]


def do_script(idea, person, scenes, talk, progress=gr.Progress()):
    try:
        return autofilm.write_script(idea, person, scenes, talk, progress)
    except Exception as e:
        err(e)


def do_autofilm(rows, person, face, aspect, engine, ref, lips, mood, nature, subs, music_on, model=VIDEO_MODELS[0],
                sfx=True, progress=gr.Progress()):
    if hasattr(rows, "values"):  # pandas-tabell från gr.Dataframe
        rows = rows.values.tolist()
    try:
        return autofilm.make_film(rows, person, face, aspect, engine, ref, lips, mood, nature, subs, music_on, progress,
                                  wan=model.startswith("Wan"), sfx=sfx)
    except Exception as e:
        err(e)


def do_music(secs, mood, nature):
    return music.make(float(secs), mood, nature)


def do_music_ai(desc, style, secs, size, seed, guidance, nature, progress=gr.Progress()):
    text = (desc or "").strip()
    if not text and style not in music.STYLES:
        raise gr.Error("Skriv en beskrivning eller välj en färdig stil.")
    en = to_english(text) if text else music.STYLES[style]
    if text and style in music.STYLES:
        en = f"{en}, {music.STYLES[style]}"
    try:
        out = music.ai(en, float(secs), music.SIZES[size], int(seed), float(guidance), bool(nature), progress=progress)
    except Exception as e:
        err(e)
    name = os.path.basename(out)
    return out, f"✅ Klar: `{name}`  \n🎼 Beskrivning till AI:n: *{en}*  \n🎲 Frö: **{name.split('_')[2]}** (skriv in det för att få samma låt igen)"


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
                + (" (Google Drive ✅)" if DRIVE_MOUNTED else
                   " (Kaggle: syns under Output – spara versionen eller ladda ner)" if KAGGLE else
                   " (försvinner när Colab stängs – ladda ner det du vill spara)"))

    with gr.Tab("⚡ Snabbfilm"):
        gr.Markdown("**Från idé till färdig film.** 1) Skriv idén och tryck *Skriv manus*. "
                    "2) Ändra fritt i tabellen. 3) Tryck *Gör filmen* – sedan sköter studion resten. "
                    "Räkna med ca 5–10 minuter per scen.")
        with gr.Row():
            with gr.Column():
                a_idea = gr.Textbox(label="Idé", lines=3, placeholder="En äldre man som bor i skogen berättar varför "
                                    "han lämnade stan och hur naturen gjorde honom friskare")
                a_person = gr.Textbox(label="Huvudperson (utseende)", placeholder="äldre man, 70 år, långt vitt skägg, "
                                      "rutig flanellskjorta")
                a_face = gr.Image(label="Huvudpersonens ansikte (valfritt – annars skapas ett)", type="filepath",
                                  height=220)
                with gr.Row():
                    a_n = gr.Slider(2, 8, value=4, step=1, label="Antal scener")
                    a_talk = gr.Radio(["Blandat", "Mest prat", "Mest berättarröst"], value="Blandat", label="Stil")
                a_write = gr.Button("1. Skriv manus", variant="secondary")
            with gr.Column():
                a_rows = gr.Dataframe(headers=["Typ (prat/scen)", "Bild", "Rörelse", "Replik"],
                                      datatype=["str", "str", "str", "str"], col_count=(4, "fixed"),
                                      row_count=(4, "dynamic"), interactive=True, wrap=True, label="Manus")
                with gr.Row():
                    a_aspect = gr.Radio(["9:16", "16:9", "1:1"], value="9:16", label="Format")
                    a_engine = gr.Radio(voice.ENGINES, value=voice.ENGINES[0], label="Röst")
                with gr.Row():
                    a_ref = gr.Audio(label="Röst att härma (valfritt)", type="filepath")
                    with gr.Column():
                        a_lips = gr.Checkbox(True, label="Bättre läppsynk (LatentSync)")
                        a_subs = gr.Checkbox(True, label="Text i takt med talet")
                        a_music = gr.Checkbox(True, label="Musik")
                        a_nat = gr.Checkbox(True, label="Naturljud")
                with gr.Row():
                    a_model = gr.Radio(VIDEO_MODELS, value=VIDEO_MODELS[0], label="Videomodell")
                    a_sfx = gr.Checkbox(True, label="Ljudeffekter (MMAudio)")
                a_mood = gr.Dropdown(music.ALL_MOODS, value=list(music.MOODS)[0], label="Musik (🤖 = AI-musik, tar några min extra)")
                a_go = gr.Button("2. Gör filmen", variant="primary")
                a_out = gr.Video(label="Färdig film")

    with gr.Tab("1. Bild"):
        with gr.Row():
            with gr.Column():
                i_prompt = gr.Textbox(label="Beskriv bilden", lines=3,
                                      placeholder="En äldre man med långt vitt skägg utanför en timmerstuga i en svensk tallskog, morgonljus")
                with gr.Row():
                    i_aspect = gr.Radio(["9:16", "16:9", "1:1", "4:5"], value="9:16", label="Format")
                    i_count = gr.Slider(1, 4, value=2, step=1, label="Antal bilder")
                i_seed = gr.Number(-1, label="Seed (-1 = slumpa)", precision=0)
                with gr.Accordion("👤 Samma person i alla bilder", open=False):
                    i_face = gr.Image(label="Huvudpersonens ansikte (tydligt, framifrån)", type="filepath", height=200)
                    i_like = gr.Slider(0.3, 0.9, value=0.6, step=0.05,
                                       label="Likhet (högre = mer lik, lägre = friare bild)")
                i_btn = gr.Button("Skapa bilder", variant="primary")
                i_info = gr.Markdown()
            with gr.Column():
                i_gal = gr.Gallery(label="Klicka på en bild för att välja den", columns=2, height=520)
                i_sel = gr.Image(label="Vald bild", type="filepath", height=220)
                with gr.Row():
                    i_to_scene = gr.Button("➡ Använd i rörlig scen")
                    i_to_talk = gr.Button("➡ Använd i pratklipp")
                i_to_face = gr.Button("👤 Gör vald bild till huvudperson")

    with gr.Tab("2. Rörlig scen"):
        with gr.Row():
            with gr.Column():
                s_img = gr.Image(label="Startbild", type="filepath")
                s_prompt = gr.Textbox(label="Vad ska hända?", lines=3)
                gr.Markdown(TIP)
                with gr.Row():
                    s_aspect = gr.Radio(["9:16", "16:9", "1:1"], value="9:16", label="Format")
                    s_secs = gr.Slider(2, 10, value=3, step=1, label="Sekunder (över 3 s byggs scenen i delar)")
                s_model = gr.Radio(VIDEO_MODELS, value=VIDEO_MODELS[0], label="Videomodell")
                with gr.Row():
                    s_smooth = gr.Checkbox(True, label="Mjukare rörelser (60 bilder/s) och skarpare bild")
                    s_sfx = gr.Checkbox(True, label="Ljudeffekter som passar bilden (MMAudio)")
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
                t_lips = gr.Checkbox(True, label="Bättre läppsynk (LatentSync) – tydligare mun, tar ca dubbelt så lång tid")
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

    with gr.Tab("5. Musik"), gr.Tabs():
        with gr.Tab("🤖 AI-musik (MusicGen)"):
            gr.Markdown("Beskriv musiken – gärna **stil, instrument, stämning och tempo** – eller välj en färdig stil. "
                        "Svenska översätts automatiskt. Ingen sång, bara instrument. "
                        "Ca 1–2 min per 30 sekunder musik (första gången laddas modellen ner).")
            with gr.Row():
                with gr.Column():
                    am_desc = gr.Textbox(label="Beskrivning (valfritt)", lines=3,
                                         placeholder="lugn akustisk gitarr och piano, varm känsla, som en morgon i skogen, 80 bpm")
                    am_style = gr.Dropdown(["(ingen)"] + list(music.STYLES), value="Lugn akustisk (skog, natur)",
                                           label="Färdig stil (läggs till beskrivningen)")
                    with gr.Row():
                        am_secs = gr.Slider(5, 180, value=30, step=5, label="Längd (sek)")
                        am_size = gr.Radio(list(music.SIZES), value="Bra (medium)", label="Kvalitet")
                    with gr.Accordion("Mer", open=False):
                        am_seed = gr.Number(-1, precision=0, label="Frö (-1 = slumpa, samma frö = samma låt)")
                        am_guid = gr.Slider(1.0, 6.0, value=3.0, step=0.5,
                                            label="Hur noga beskrivningen följs (högre = noggrannare, mindre variation)")
                        am_nat = gr.Checkbox(False, label="Lägg till naturljud (vind och fåglar)")
                    am_btn = gr.Button("🎵 Skapa AI-musik", variant="primary")
                with gr.Column():
                    am_out = gr.Audio(label="Resultat", type="filepath")
                    am_info = gr.Markdown()
                    gr.Markdown("<small>MusicGen-modellerna har licensen CC-BY-NC 4.0: fritt för egna och "
                                "icke-kommersiella filmer, men inte för att sälja eller tjäna pengar på.</small>")
        with gr.Tab("Enkel musik (kod)"):
            gr.Markdown("Snabb bakgrundsmusik som skapas direkt med kod – ingen AI, inga licensfrågor.")
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
                    f_subs = gr.Checkbox(True, label="Text i takt med talet")
                    f_music = gr.Checkbox(True, label="Musik")
                with gr.Row():
                    f_engine = gr.Radio(voice.ENGINES, value=voice.ENGINES[0], label="Röst")
                    f_ref = gr.Audio(label="Röst att härma (valfritt)", type="filepath")
                with gr.Row():
                    f_speed = gr.Slider(0.8, 1.2, value=1.0, step=0.05, label="Tempo")
                    f_mood = gr.Dropdown(music.ALL_MOODS, value=list(music.MOODS)[0], label="Musik (🤖 = AI)")
                    f_nat = gr.Checkbox(True, label="Naturljud")
                f_btn = gr.Button("Bygg filmen", variant="primary")
            f_out = gr.Video(label="Färdig film")

    # kopplingar
    i_btn.click(do_images, [i_prompt, i_aspect, i_count, i_seed, i_face, i_like], [gallery_paths, i_sel, i_info]).then(
        lambda p: p, gallery_paths, i_gal)
    i_gal.select(pick_image, gallery_paths, i_sel)
    i_to_scene.click(lambda p: p, i_sel, s_img)
    i_to_talk.click(lambda p: p, i_sel, t_img)
    i_to_face.click(lambda p: (p, p), i_sel, [i_face, a_face])
    a_write.click(do_script, [a_idea, a_person, a_n, a_talk], a_rows)
    a_go.click(do_autofilm, [a_rows, a_person, a_face, a_aspect, a_engine, a_ref, a_lips, a_mood, a_nat, a_subs,
                             a_music, a_model, a_sfx], a_out)
    s_btn.click(do_scene, [s_img, s_prompt, s_aspect, s_secs, s_steps, s_seed, s_smooth, s_sfx, s_model],
                [s_out, s_info])
    t_btn.click(do_talk, [t_img, t_text, t_audio, t_engine, t_ref, t_enh, t_fast, t_expr, t_lips], t_out)
    v_btn.click(do_voice, [v_text, v_engine, v_ref, v_style, v_speed, v_emo], v_out)
    m_btn.click(do_music, [m_secs, m_mood, m_nat], m_out)
    am_btn.click(do_music_ai, [am_desc, am_style, am_secs, am_size, am_seed, am_guid, am_nat], [am_out, am_info])
    s_add.click(add_film, [s_out, film], [film, f_files])
    t_add.click(add_film, [t_out, film], [film, f_files])
    f_btn.click(do_film, [f_files, f_lines, f_aspect, f_engine, f_ref, f_speed, f_mood, f_nat, f_subs, f_music], f_out)

if __name__ == "__main__":
    # Förladda modellerna i bakgrunden (inte när de sparas i Drive – det skulle fylla din Drive)
    if os.environ.get("FV_PREFETCH", "1") == "1" and os.environ.get("FV_DRIVE_MODELS") != "1":
        import subprocess as _sp
        from common import env as _env
        _sp.Popen([sys.executable, str(Path(__file__).with_name("prefetch.py"))], env={**_env(), "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
        print("⏬ Laddar ner AI-modellerna i bakgrunden – du kan börja direkt.", flush=True)
    app.queue(default_concurrency_limit=1).launch(share=True, allowed_paths=[str(OUT), "/content", "/kaggle/working"], debug=True,
                                                 theme=gr.themes.Soft(primary_hue="green"))
