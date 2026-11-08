"""Översätter svenska beskrivningar till engelska (AI-modellerna förstår engelska bäst). Körs på processorn."""
import re

_m = None


def _looks_swedish(t):
    return bool(re.search(r"[åäöÅÄÖ]", t)) or bool(re.search(
        r"\b(och|som|med|den|det|en|ett|han|hon|går|sitter|står|långsamt|kameran|skogen|gamle|gubbe)\b", t.lower()))


def to_english(text):
    global _m
    t = text.strip()
    if not t or not _looks_swedish(t):
        return t
    try:
        from transformers import MarianMTModel, MarianTokenizer
        if _m is None:
            n = "Helsinki-NLP/opus-mt-sv-en"
            _m = (MarianTokenizer.from_pretrained(n), MarianMTModel.from_pretrained(n))
        tok, model = _m
        parts = [p for p in re.split(r"(?<=[.!?])\s+", t) if p]
        out = model.generate(**tok(parts, return_tensors="pt", padding=True), max_new_tokens=200)
        return " ".join(tok.batch_decode(out, skip_special_tokens=True))
    except Exception as e:
        print("Översättning misslyckades, använder texten som den är:", e)
        return t
