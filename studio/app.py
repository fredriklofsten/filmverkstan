"""Filmverkstan Studio – gränssnittet. Startas från Colab- eller Kaggle-notebooken."""
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gradio as gr  # noqa: E402

import autofilm  # noqa: E402
import montage  # noqa: E402
import music  # noqa: E402
import musicvideo  # noqa: E402
import voice  # noqa: E402
from common import OUT, DRIVE_MOUNTED, KAGGLE, run_worker, WorkerError  # noqa: E402
from translate import to_english  # noqa: E402

VERSION = "2.6"


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
    mp3 = music.to_mp3(out)
    return mp3, f"✅ Klar: `{name}` (även som MP3)  \n🎼 Beskrivning till AI:n: *{en}*  \n🎲 Frö: **{name.split('_')[2]}** (skriv in det för att få samma låt igen)", [out, mp3]


def _lang(name):
    return music.LANGUAGES.get(name, "sv")


MAX_VERSIONS = 4


def _song_md(vers, info, write, quality, mastering, preview=False):
    seeds = ", ".join(f"v{i + 1}: **{v['seed']}**" for i, v in enumerate(vers))
    md = (f"✅ {'Provlyssning' if preview else 'Klar'}: {len(vers)} version{'er' if len(vers) > 1 else ''} · "
          f"{quality.split(' (')[0]}  \n🎲 Frö: {seeds} – tryck *🎲 Behåll fröet* för att göra om en favorit  \n"
          f"🎼 Stil: *{(info.get('caption') or '')[:300]}*")
    if mastering and vers and vers[0].get("lufs_before") is not None:
        md += f"  \n🎚️ Mastrad: {vers[0]['lufs_before']:.1f} → {music.MASTER_LUFS} LUFS (jämför under *Före mastering*)"
    if write:
        md += "  \n✍️ AI:n skrev texten – den står i rutan *Låttext*. Ändra den gärna och tryck *Skapa låt* igen."
    if preview:
        md += "  \n👂 Låter orden rätt? Rätta under *Uttal* och provlyssna igen – eller tryck *Skapa låt*."
    return md


def _finish_versions(wavs, info, mastering, meta):
    """Mastring, MP3 och sparad info för varje version. Returnerar lista med dict per version."""
    seeds = info.get("seeds") or [info.get("seed")]
    vers = []
    for i, w in enumerate(wavs):
        v = {"raw": w, "seed": seeds[min(i, len(seeds) - 1)], "master": None, "lufs_before": None}
        if mastering:
            try:
                v["lufs_before"] = music.loudness(w)
                v["master"] = music.master(w)
            except Exception as e:
                print("Mastringen misslyckades – använder originalet:", e, flush=True)
        v["final"] = v["master"] or w
        v["mp3"] = music.to_mp3(v["final"])
        music.save_meta(w, {**meta, "version": i + 1, "seed": v["seed"], "wav": w, "master": v["master"],
                            "mp3": v["mp3"], "model": info.get("model"), "caption": info.get("caption"),
                            "bpm": info.get("bpm") or meta.get("bpm"), "created": time.strftime("%Y-%m-%d %H:%M")})
        vers.append(v)
    return vers


def _song_outputs(vers, md, lyrics_box, write_flag):
    audio = [gr.update(value=vers[i]["mp3"], visible=True) if i < len(vers) else gr.update(value=None, visible=False)
             for i in range(MAX_VERSIONS)]
    rows = [gr.update(visible=i < len(vers)) for i in range(MAX_VERSIONS)]
    files = [f for v in vers for f in (v["final"], v["mp3"]) + ((v["raw"],) if v["master"] else ())]
    before = gr.update(value=vers[0]["raw"] if vers[0]["master"] else None, visible=bool(vers[0]["master"]))
    return (*audio, *rows, md, lyrics_box, write_flag, vers, files, vers[0]["raw"], vers[0]["final"], before,
            gr.update(choices=music.library()))


def do_song(desc, lyrics, write, lang_name, secs, bpm, seed, variants, quality, mastering, pron, preview=False,
            progress=gr.Progress()):
    desc = (desc or "").strip()
    lyrics = (lyrics or "").strip()
    lang = _lang(lang_name)
    if not desc and not lyrics:
        raise gr.Error("Beskriv låten eller skriv en låttext.")
    write = lang != "none" and (bool(write) or not lyrics)
    n = int(str(variants)[0]) if str(variants)[:1].isdigit() else 1
    q = music.QUALITIES.get(quality, "fast")
    music.save_settings({"lang": lang_name, "quality": quality, "variants": variants, "mastering": bool(mastering),
                         "pron": pron or ""})
    sung = music.apply_pron(lyrics, pron) if lang != "none" else lyrics
    seconds = float(secs)
    if preview:
        if write:
            raise gr.Error("Skriv (eller låt AI:n skriva) texten först – provlyssningen sjunger början av din text.")
        sung = music.preview_lyrics(sung) if lang != "none" else sung
        seconds, n, write = 30.0, 1, False
    try:
        wavs, info = music.song(to_english(desc) if desc else "", query=desc, lyrics=sung, write_lyrics=write,
                                language=lang, seconds=seconds, bpm=int(bpm or 0), seed=int(seed),
                                variants=n, quality=q, progress=progress)
        out_lyrics = (info.get("lyrics") or lyrics) if write else lyrics
        vers = _finish_versions(wavs, info, bool(mastering), {
            "task": "song", "preview": preview, "query": desc, "lyrics": out_lyrics, "lyrics_sung": sung,
            "pron": pron or "", "language": lang_name, "seconds": float(secs), "bpm": int(bpm or 0),
            "quality": quality, "mastering": bool(mastering)})
    except Exception as e:
        err(e)
    return _song_outputs(vers, _song_md(vers, info, write, quality, mastering, preview), out_lyrics, False)


def do_song_preview(desc, lyrics, write, lang_name, secs, bpm, seed, variants, quality, mastering, pron,
                    progress=gr.Progress()):
    """Gör bara början (ca 30 sek, en version) för att höra om orden låter rätt."""
    return do_song(desc, lyrics, write, lang_name, secs, bpm, seed, variants, quality, mastering, pron,
                   preview=True, progress=progress)


def pick_song(cur, which):
    """Använd en version i redigering (originalet) och musikvideo (den mastrade)."""
    if not cur:
        raise gr.Error("Skapa en låt först.")
    v = cur[min(which, len(cur)) - 1]
    gr.Info(f"Version {min(which, len(cur))} vald")
    return v["raw"], v["final"]


def keep_seed(cur, which):
    if not cur or which > len(cur):
        raise gr.Error("Skapa en låt först.")
    gr.Info(f"Fröet {cur[which - 1]['seed']} är inskrivet – samma text och kvalitet ger samma låt igen.")
    return cur[which - 1]["seed"], "1 version"


def do_edit(src, mode, start, end, desc, lyrics, lang_name, strength, seed, mastering, pron, progress=gr.Progress()):
    if not src:
        raise gr.Error("Välj eller ladda upp en låt först.")
    cap = to_english(desc) if (desc or "").strip() else ""
    sung = music.apply_pron(lyrics or "", pron)
    try:
        if mode.startswith("Gör om"):
            if float(end) <= float(start):
                raise ValueError("Slutet måste komma efter början.")
            wavs, info = music.repaint(src, start, end, cap or "same style as the rest of the song", sung,
                                       _lang(lang_name), int(seed), progress)
        else:
            if not cap:
                raise ValueError("Beskriv den nya stilen, t.ex. 'akustisk version med piano och stråkar'.")
            wavs, info = music.cover(src, cap, float(strength), sung, _lang(lang_name), int(seed), progress)
        vers = _finish_versions(wavs[:1], info, bool(mastering), {
            "task": info.get("task") or ("repaint" if mode.startswith("Gör om") else "cover"), "query": desc or "",
            "lyrics": lyrics or "", "lyrics_sung": sung, "pron": pron or "", "language": lang_name,
            "mastering": bool(mastering), "src": src})
    except Exception as e:
        err(e)
    v = vers[0]
    files = [v["final"], v["mp3"]] + ([v["raw"]] if v["master"] else [])
    return v["mp3"], f"✅ Klar · 🎲 Frö: **{info.get('seed')}**", files, v["final"], gr.update(choices=music.library())


# ---------- Bibliotek ----------
def lib_refresh():
    return gr.update(choices=music.library(), value=None)


def lib_show(js):
    if not js:
        return None, "", None
    d = json.load(open(js))
    final = d.get("master") or d["wav"]
    md = (f"**{d.get('created', '')}** · version {d.get('version', 1)} · 🎲 frö **{d.get('seed')}** · "
          f"{(d.get('quality') or '').split(' (')[0]} · {d.get('language', '')}"
          f"{' · mastrad' if d.get('master') else ''}  \n"
          f"🎼 *{(d.get('query') or d.get('caption') or '')[:300]}*")
    if d.get("lyrics"):
        md += "\n\n```\n" + d["lyrics"][:1500] + "\n```"
    files = [f for f in (final, d.get("mp3"), d["wav"] if d.get("master") else None) if f and os.path.exists(f)]
    return d.get("mp3") if d.get("mp3") and os.path.exists(d["mp3"]) else final, md, files


def lib_load(js):
    if not js:
        raise gr.Error("Välj en låt i listan först.")
    d = json.load(open(js))
    gr.Info("Inställningarna är inladdade i fliken 🎤 Låt med sång")
    q = d.get("quality") if d.get("quality") in music.QUALITIES else list(music.QUALITIES)[0]
    lang = d.get("language") if d.get("language") in music.LANGUAGES else "Svenska"
    return (d.get("query") or "", d.get("lyrics") or "", lang, d.get("seconds") or 0, d.get("bpm") or 0,
            d.get("seed", -1), q, d.get("pron") or "", "1 version", False)


def lib_use(js):
    if not js:
        raise gr.Error("Välj en låt i listan först.")
    d = json.load(open(js))
    gr.Info("Låten är vald i Redigera låt och Musikvideo")
    return d["wav"], d.get("master") or d["wav"], d.get("lyrics") or "", d.get("lyrics") or ""


def load_song_settings():
    st = music.load_settings()
    g = lambda k, ok, default: st.get(k) if st.get(k) in ok else default  # noqa: E731
    return (g("lang", music.LANGUAGES, "Svenska"),
            g("quality", music.QUALITIES, list(music.QUALITIES)[0]),
            g("variants", VARIANT_CHOICES, VARIANT_CHOICES[1]),
            bool(st.get("mastering", True)), st.get("pron", ""))


VARIANT_CHOICES = ["1 version", "2 versioner", "3 versioner", "4 versioner"]


def do_video(song, lyrics, desc, style, aspect, lang_name, subs, moving, face, progress=gr.Progress()):
    try:
        return musicvideo.make(song, lyrics or "", desc or "", style, aspect, _lang(lang_name), bool(subs),
                               bool(moving), face, progress)
    except Exception as e:
        err(e)


def do_music_mp3(*args, **kw):
    out = do_music(*args, **kw)
    return out, [out, music.to_mp3(out)]


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
                a_mood = gr.Dropdown(music.ALL_MOODS, value=list(music.MOODS)[0], label="Musik (🤖 = AI-musik med ACE-Step – får säljas, tar några min extra)")
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
        cur_song = gr.State([])
        with gr.Tab("🎤 Låt med sång (ACE-Step)"):
            gr.Markdown("Hela låtar med **sång** – på svenska, engelska m.fl. – eller instrumentalt. "
                        "Skriv egen text eller låt AI:n skriva den. "
                        "(Första gången installeras ACE-Step, ca 10–15 min – det startar redan när studion startar.)")
            with gr.Row():
                with gr.Column():
                    sg_desc = gr.Textbox(label="Beskriv låten (stil, instrument, stämning, röst)", lines=3,
                                         placeholder="glad svensk sommarpop med akustisk gitarr och kvinnlig sång, "
                                                     "handlar om att bada i sjön med vänner")
                    sg_lyrics = gr.Textbox(label="Låttext (tom = AI:n skriver den)", lines=10,
                                           placeholder="[verse]\nSolen glittrar på sjön\nvi springer ner till bryggan\n\n"
                                                       "[chorus]\nHela sommaren är vår\n…")
                    sg_write = gr.Checkbox(False, label="Låt AI:n skriva en ny text (ersätter texten ovan)")
                    with gr.Accordion("🗣️ Uttal – om AI:n sjunger ett ord fel", open=False):
                        sg_pron = gr.Textbox(label="Ett ord per rad:  ord = så som det ska sjungas", lines=4,
                                             placeholder="sjön = schön\nkärlek = schärlek\nhärifrån = här-i-från")
                        gr.Markdown("<small>Orden byts bara i det som skickas till AI:n – din text i rutan och i "
                                    "musikvideons undertexter är oförändrad. Dela gärna upp svåra ord med bindestreck. "
                                    "Testa med <b>🎧 Provlyssna början</b>.</small>")
                    with gr.Row():
                        sg_lang = gr.Dropdown(list(music.LANGUAGES), value="Svenska", label="Språk")
                        sg_secs = gr.Slider(0, 240, value=0, step=5, label="Längd (sek, 0 = efter texten)")
                    sg_quality = gr.Radio(list(music.QUALITIES), value=list(music.QUALITIES)[0], label="Kvalitet")
                    with gr.Row():
                        sg_var = gr.Radio(VARIANT_CHOICES, value=VARIANT_CHOICES[1], label="Antal versioner (välj den bästa)")
                        sg_master = gr.Checkbox(True, label="🎚️ Automatisk mastering (-14 LUFS, som Spotify/YouTube)")
                    with gr.Accordion("Mer", open=False):
                        sg_bpm = gr.Number(0, precision=0, label="Tempo i bpm (0 = AI:n väljer)")
                        sg_seed = gr.Number(-1, precision=0, label="Frö (-1 = slumpa)")
                        gr.Markdown("<small>Samma frö + samma text + samma kvalitet = samma låt. Jämför modellerna "
                                    "genom att göra samma låt med samma frö i <i>Snabb</i> och <i>Normal</i>. "
                                    "XL-modellerna laddas ner första gången de används (10–20 GB).</small>")
                    with gr.Row():
                        sg_prev = gr.Button("🎧 Provlyssna början (ca 30 sek)")
                        sg_btn = gr.Button("🎤 Skapa låt", variant="primary")
                with gr.Column():
                    sg_outs, sg_rows, sg_uses, sg_keeps = [], [], [], []
                    for _i in range(MAX_VERSIONS):
                        sg_outs.append(gr.Audio(label=f"Version {_i + 1}", type="filepath", visible=_i == 0))
                        with gr.Row(visible=_i == 0) as _r:
                            sg_uses.append(gr.Button(f"✔ Använd version {_i + 1}", size="sm"))
                            sg_keeps.append(gr.Button("🎲 Behåll fröet", size="sm"))
                        sg_rows.append(_r)
                    sg_info = gr.Markdown()
                    with gr.Accordion("🎚️ Före mastering (version 1)", open=False):
                        sg_before = gr.Audio(label="Utan mastering", type="filepath", visible=False)
                    gr.Markdown("<small>Vald version används i <b>Redigera låt</b> och <b>Musikvideo</b>. "
                                "Alla låtar sparas i <b>📚 Mina låtar</b>.</small>")
                    sg_files = gr.File(label="Ladda ner (WAV och MP3)", file_count="multiple")
                    gr.Markdown("<small>💡 Märk upp texten med <b>[verse]</b>, <b>[chorus]</b>, <b>[bridge]</b> för "
                                "tydligare låtform. ACE-Step har MIT-licens – du får använda låtarna kommersiellt, "
                                "men se till att texten är din egen.</small>")
        with gr.Tab("📚 Mina låtar") as lib_tab:
            gr.Markdown("Alla låtar du gjort, nyast först – med text, stil, inställningar och frö. "
                        "Sparas i resultatmappen (i Google Drive om den är ansluten).")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        lib_dd = gr.Dropdown([], label="Låt", scale=4)
                        lib_ref = gr.Button("🔄 Uppdatera", size="sm", scale=1)
                    lib_audio = gr.Audio(label="Lyssna", type="filepath")
                    with gr.Row():
                        lib_load_btn = gr.Button("↩️ Ladda in inställningarna (för att göra om den)")
                        lib_use_btn = gr.Button("✔ Använd i Redigera och Musikvideo")
                    lib_files = gr.File(label="Ladda ner", file_count="multiple")
                with gr.Column():
                    lib_info = gr.Markdown()
        with gr.Tab("✂️ Redigera låt"):
            gr.Markdown("Gör om **en del** av en låt (resten behålls), eller gör en **ny version i en annan stil**. "
                        "Använd en låt du gjort här, eller ladda upp en egen.")
            with gr.Row():
                with gr.Column():
                    ed_src = gr.Audio(label="Låt att ändra", type="filepath")
                    ed_mode = gr.Radio(["Gör om en del", "Ny stil (hela låten)"], value="Gör om en del", label="Vad vill du göra?")
                    with gr.Row():
                        ed_start = gr.Number(30, label="Från (sek)")
                        ed_end = gr.Number(45, label="Till (sek)")
                    ed_desc = gr.Textbox(label="Beskrivning", lines=2,
                                         placeholder="Gör om en del: t.ex. 'gitarrsolo'. Ny stil: t.ex. 'akustisk version med piano och stråkar'")
                    ed_lyrics = gr.Textbox(label="Låttext (samma som låten, ändra rader du vill byta)", lines=6)
                    with gr.Row():
                        ed_lang = gr.Dropdown(list(music.LANGUAGES), value="Svenska", label="Språk")
                        ed_str = gr.Slider(0.1, 1.0, value=0.5, step=0.05,
                                           label="Ny stil: hur mycket av originalet som behålls")
                    ed_seed = gr.Number(-1, precision=0, label="Frö (-1 = slumpa)")
                    gr.Markdown("<small>Mastering och uttal följer inställningarna i <b>🎤 Låt med sång</b>.</small>")
                    ed_btn = gr.Button("✂️ Ändra låten", variant="primary")
                with gr.Column():
                    ed_out = gr.Audio(label="Resultat", type="filepath")
                    ed_info = gr.Markdown()
                    ed_files = gr.File(label="Ladda ner (WAV och MP3)", file_count="multiple")
        with gr.Tab("🎬 Musikvideo"):
            gr.Markdown("Gör en musikvideo av en låt: **bilder som följer texten**, mjuka kamerarörelser och "
                        "**texten i takt med sången**. Ca 3–6 min för en låt (längre med rörliga scener).")
            with gr.Row():
                with gr.Column():
                    mv_song = gr.Audio(label="Låt", type="filepath")
                    mv_lyrics = gr.Textbox(label="Låttext (för undertexter och bildernas motiv)", lines=6)
                    mv_desc = gr.Textbox(label="Vad ska synas? (miljö, person, känsla)", lines=2,
                                         placeholder="ett gäng vänner vid en sjö i Sverige en sommarkväll, gyllene ljus")
                    with gr.Row():
                        mv_style = gr.Dropdown(list(musicvideo.STYLES), value="Filmisk", label="Bildstil")
                        mv_aspect = gr.Radio(["9:16", "16:9", "1:1"], value="9:16", label="Format")
                    with gr.Row():
                        mv_lang = gr.Dropdown(list(music.LANGUAGES), value="Svenska", label="Språk i sången")
                        mv_subs = gr.Checkbox(True, label="Text i takt med sången")
                        mv_move = gr.Checkbox(False, label="Rörliga scener (LTX, +1–2 min per scen)")
                    mv_face = gr.Image(label="Samma person i bilderna (valfritt)", type="filepath", height=180)
                    mv_btn = gr.Button("🎬 Gör musikvideo", variant="primary")
                mv_out = gr.Video(label="Musikvideo")
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
                    am_files = gr.File(label="Ladda ner (WAV och MP3)", file_count="multiple")
                    gr.Markdown("<small>MusicGen-modellerna har licensen CC-BY-NC 4.0: fritt för egna och "
                                "icke-kommersiella filmer, men inte för att sälja. Vill du sälja: använd "
                                "<b>Låt med sång</b> och välj språk <i>Instrumental</i> (ACE-Step, MIT).</small>")
        with gr.Tab("Enkel musik (kod)"):
            gr.Markdown("Snabb bakgrundsmusik som skapas direkt med kod – ingen AI, inga licensfrågor.")
            with gr.Row():
                with gr.Column():
                    m_secs = gr.Slider(5, 180, value=30, step=1, label="Längd (sek)")
                    m_mood = gr.Dropdown(list(music.MOODS), value=list(music.MOODS)[0], label="Stämning")
                    m_nat = gr.Checkbox(True, label="Naturljud")
                    m_btn = gr.Button("Skapa musik", variant="primary")
                with gr.Column():
                    m_out = gr.Audio(label="Resultat", type="filepath")
                    m_files = gr.File(label="Ladda ner (WAV och MP3)", file_count="multiple")

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
    m_btn.click(do_music_mp3, [m_secs, m_mood, m_nat], [m_out, m_files])
    sg_in = [sg_desc, sg_lyrics, sg_write, sg_lang, sg_secs, sg_bpm, sg_seed, sg_var, sg_quality, sg_master, sg_pron]
    sg_res = [*sg_outs, *sg_rows, sg_info, sg_lyrics, sg_write, cur_song, sg_files, ed_src, mv_song, sg_before, lib_dd]
    _copy = dict(fn=lambda l, lang: (l, l, lang, lang), inputs=[sg_lyrics, sg_lang],
                 outputs=[ed_lyrics, mv_lyrics, ed_lang, mv_lang])
    sg_btn.click(do_song, sg_in, sg_res).then(**_copy)
    sg_prev.click(do_song_preview, sg_in, sg_res)
    for _i in range(MAX_VERSIONS):
        sg_uses[_i].click(lambda c, _n=_i + 1: pick_song(c, _n), cur_song, [ed_src, mv_song])
        sg_keeps[_i].click(lambda c, _n=_i + 1: keep_seed(c, _n), cur_song, [sg_seed, sg_var])
    ed_btn.click(do_edit, [ed_src, ed_mode, ed_start, ed_end, ed_desc, ed_lyrics, ed_lang, ed_str, ed_seed,
                           sg_master, sg_pron], [ed_out, ed_info, ed_files, mv_song, lib_dd])
    lib_tab.select(lib_refresh, None, lib_dd)
    lib_ref.click(lib_refresh, None, lib_dd)
    lib_dd.change(lib_show, lib_dd, [lib_audio, lib_info, lib_files])
    lib_load_btn.click(lib_load, lib_dd, [sg_desc, sg_lyrics, sg_lang, sg_secs, sg_bpm, sg_seed, sg_quality, sg_pron,
                                          sg_var, sg_write])
    lib_use_btn.click(lib_use, lib_dd, [ed_src, mv_song, ed_lyrics, mv_lyrics])
    app.load(load_song_settings, None, [sg_lang, sg_quality, sg_var, sg_master, sg_pron])
    mv_btn.click(do_video, [mv_song, mv_lyrics, mv_desc, mv_style, mv_aspect, mv_lang, mv_subs, mv_move, mv_face],
                 mv_out)
    am_btn.click(do_music_ai, [am_desc, am_style, am_secs, am_size, am_seed, am_guid, am_nat],
                 [am_out, am_info, am_files])
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
    music.install_in_background()   # ACE-Step (låtar med sång) installeras medan du gör annat
    app.queue(default_concurrency_limit=1).launch(share=True, allowed_paths=[str(OUT), "/content", "/kaggle/working"], debug=True,
                                                 theme=gr.themes.Soft(primary_hue="green"))
