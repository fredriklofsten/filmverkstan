# 🎬 Filmverkstan

Gratis AI-filmstudio som körs på Googles grafikkort i Colab.

**[▶ Öppna Filmverkstan Studio i Colab](https://colab.research.google.com/github/fredriklofsten/filmverkstan/blob/main/Filmverkstan_Studio.ipynb)**

| Flik | Vad den gör | AI-modell |
|---|---|---|
| 1. Bild | Fotorealistiska bilder från en beskrivning | RealVisXL V5 Lightning |
| 2. Rörlig scen | Gör en bild till en 2–6 sekunders videoscen | LTX-Video 0.9.5 |
| 3. Pratklipp | Får en person på en bild att prata | SadTalker + GFPGAN |
| 4. Röst | Svensk röst – naturlig eller snabb, kan härma en inspelning | Chatterbox Multilingual / Piper |
| 5. Musik | Hela låtar med sång (svenska m.fl.), AI-musik i valfri stil, eller enkel bakgrundsmusik | ACE-Step 1.5 / MusicGen / kod |
| 6. Bygg film | Sätter ihop allt med röst, text och musik | ffmpeg |

Beskrivningar kan skrivas på svenska – de översätts automatiskt (Helsinki-NLP opus-mt-sv-en).

## Struktur
- `Filmverkstan_Studio.ipynb` – startar allt i Colab (hämtar alltid senaste koden härifrån)
- `studio/app.py` – gränssnittet (Gradio)
- `studio/setup_env.py` – installation, inkl. rättningar så att SadTalker/basicsr fungerar med nya Python
- `studio/workers/` – varje AI-modell körs i en egen process så att minnet frigörs efteråt

## Licenser
Modellerna har egna licenser (LTX-Video: Lightricks open weights-licens, RealVisXL: OpenRAIL++, SadTalker: Apache 2.0, Chatterbox: MIT, MusicGen: CC-BY-NC 4.0, ACE-Step 1.5: MIT, Piper-röst: se rhasspy/piper-voices). **MusicGen: CC-BY-NC 4.0 – inte för kommersiellt bruk.** Läs dem innan du använder resultat kommersiellt.

## Kaggle (2 grafikkort, ca 30 h/vecka)
1. Skapa ett gratis konto på kaggle.com och verifiera telefonnumret (Settings → Phone verification).
2. **Create → New Notebook → File → Import Notebook → GitHub** och klistra in `fredriklofsten/filmverkstan`, välj `Filmverkstan_Kaggle.ipynb`.
3. **Session options:** Accelerator **GPU T4 x2**, Internet **On**. Tryck ▶.

## Nytt i v2.1
- **⚡ Snabbfilm:** skriv en idé → studion skriver manus (Qwen3 4B), som du kan ändra, och gör sedan bilder, rörliga scener, pratklipp, röst, musik och färdig film automatiskt.
- **👤 Samma person i alla bilder:** välj ett ansikte (IP-Adapter Plus Face) så återkommer samma person i varje bild.
- **Bättre läppsynk:** SadTalker ger huvudrörelser, LatentSync 1.5 ritar om munnen efter ljudet. Misslyckas den används SadTalker-klippet.
- **Längre och mjukare scener:** upp till 10 sekunder (byggs i delar), 60 bilder/s och 1,5× skarpare bild.

## Nytt i v2.2
- **Wan 2.2 (5B)** som extra videomodell: snyggare rörelser, ca 3× långsammare än LTX. Välj under *Videomodell*.
- **Ljudeffekter med MMAudio:** fåglar, steg, vind och vatten som följer det som händer i bilden. Ligger kvar under rösten och trycks ner när någon pratar.
- **Undertexter i takt med talet (Whisper):** korta textrader som visas exakt när orden sägs – även i pratklippen.
- Rättat: musiken tystnade när sista repliken var slut.


## Nytt i v2.3
- **🤖 AI-musik med MusicGen** (flik 5): beskriv musiken på svenska eller välj en av 12 färdiga stilar (lo-fi, episk orkester, synthwave, nordisk folkmusik …). Tre kvalitetsnivåer, upp till 3 minuter långa låtar (byggs i delar som fortsätter varandra), frö för att få samma låt igen och valfria naturljud.
- **AI-musik i filmerna:** välj en stil som börjar med 🤖 under *Musik* i Snabbfilm eller Bygg film. Musiken läggs lägre under rösten. Om något går fel används den enkla musiken i stället.
- Observera: MusicGen har licensen CC-BY-NC 4.0 – fritt för egna projekt, men inte för att sälja.

## Nytt i v2.4
- **🎤 Låt med sång (ACE-Step 1.5)** – flik 5: hela låtar med sång på svenska, engelska, norska m.fl., eller instrumentalt. Skriv egen låttext (gärna med `[verse]`, `[chorus]`, `[bridge]`) eller lämna den tom så skriver AI:n texten. Texten hamnar i rutan så att du kan ändra den och skapa låten igen.
- ACE-Step har **MIT-licens** – låtarna får användas kommersiellt (till skillnad från MusicGen).
- Installeras automatiskt första gången (ca 10–15 min, ca 20 GB disk) i en egen Python-miljö. Versionen är låst till en testad version.

## Nytt i v2.5
- **Snabbare låtar:** ACE-Step ligger kvar i minnet mellan låtarna (på Kaggle på grafikkort 2). Den stängs automatiskt när videomodellerna behöver kortet och efter 10 minuter utan låtar.
- **ACE-Step installeras redan när studion startar** (om det finns plats på disken), så första låten slipper vänta.
- **2 versioner per låt** – lyssna på båda och välj den bästa med *✔ Använd version*.
- **✂️ Redigera låt:** gör om en del av en låt (t.ex. sekund 30–45) eller gör en ny version i en annan stil.
- **🎬 Musikvideo:** bilder som följer låttexten, mjuka kamerarörelser (eller rörliga scener) och texten i takt med sången.
- **Filmmusiken (🤖) görs nu med ACE-Step** – instrumentalt och med MIT-licens, så filmerna får säljas. MusicGen och kodmusiken finns kvar som reserv.
- **MP3** (320 kbit/s) sparas bredvid varje WAV-fil.

## Nytt i v2.6 (Musikverkstan-uppgraderingen, del 1)
- **🎚️ Automatisk mastering:** varje låt får jämn ljudstyrka (-14 LUFS, samma som Spotify och YouTube), en lätt kompressor och en limiter (max -1 dB). Kan stängas av; lyssna på *Före mastering* för att jämföra. Mastrad fil sparas som 24-bit WAV.
- **Upp till 4 versioner** av samma låt. Varje version visar sitt frö – *🎲 Behåll fröet* skriver in det så att du kan göra om favoriten.
- **Kvalitetsreglage:** *Snabb* (2B turbo), *Normal* (nya **XL-modellen, 4B**, bättre ljud) och *Hög kvalitet* (XL sft, 50 steg). XL laddas ner första gången den används (10–20 GB). Senast valda inställningar sparas.
- **🗣️ Uttal:** skriv `ord = så som det ska sjungas` om AI:n uttalar svenska ord fel. Bara det som skickas till AI:n ändras.
- **🎧 Provlyssna början:** gör ca 30 sekunder av början av texten för att höra uttalet innan hela låten görs.
- **📚 Mina låtar:** alla låtar sparas med text, stil, inställningar och frö. Lyssna, ladda ner, ladda in inställningarna igen eller skicka låten till Redigera/Musikvideo.
