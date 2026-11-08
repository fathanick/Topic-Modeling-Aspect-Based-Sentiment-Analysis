"""Tests for the sentence-level ABSA parser and its grounding guarantees."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sentiment_analysis"))
from sentence_absa import ASPECTS, SYSTEM_PROMPT, build_messages, parse_response  # noqa: E402

fails = []


def ck(label, cond, detail=""):
    print(f"{'OK  ' if cond else 'FAIL'} {label}{' | ' + detail if detail and not cond else ''}")
    if not cond:
        fails.append(label)


# ---------- 1. multi-aspect, mixed polarity in one sentence ----------
s = "Pantainya bersih banget, tapi parkirnya sempit dan jauh."
raw = json.dumps({"aspects": [
    {"aspect": "cleanliness", "keywords": "bersih", "sentiment": "pos", "span": "bersih banget"},
    {"aspect": "facilities", "keywords": "parkir", "sentiment": "neg", "span": "parkirnya sempit dan jauh"},
]})
r = parse_response(raw, s)
ck("multi-aspect kept", len(r["aspects"]) == 2, str(r))
ck("polarities differ", {a["sentiment"] for a in r["aspects"]} == {"pos", "neg"})
ck("status ok", r["status"] == "ok", r["status"])

# ---------- 2. three aspects, two polarities (the review-180 case) ----------
s2 = "Hanya saja air pantai tempat bermain kanonya kotor, sisi sebelahnya pantainya jernih namun ombaknya keras."
raw2 = json.dumps({"aspects": [
    {"aspect": "cleanliness", "keywords": "kotor", "sentiment": "neg", "span": "air pantai tempat bermain kanonya kotor"},
    {"aspect": "scenery", "keywords": "jernih", "sentiment": "pos", "span": "sisi sebelahnya pantainya jernih"},
    {"aspect": "safety", "keywords": "ombak", "sentiment": "neg", "span": "ombaknya keras"},
]})
r2 = parse_response(raw2, s2)
ck("three aspects kept", len(r2["aspects"]) == 3, str(r2))
ck("review-180 reproduced", sum(a["sentiment"] == "neg" for a in r2["aspects"]) == 2)

# ---------- 3. hallucinated span is rejected ----------
raw3 = json.dumps({"aspects": [
    {"aspect": "cleanliness", "keywords": "bersih", "sentiment": "pos", "span": "sangat bersih sekali"},
]})
r3 = parse_response(raw3, s)
ck("non-verbatim span rejected", r3["aspects"] == [] and r3["status"] == "rejected", str(r3))

# ---------- 4. paraphrased keyword rejected ----------
raw4 = json.dumps({"aspects": [
    {"aspect": "facilities", "keywords": "tempat parkir", "sentiment": "neg", "span": "parkirnya sempit"},
]})
r4 = parse_response(raw4, s)
ck("paraphrased keyword rejected", r4["aspects"] == [], str(r4))

# ---------- 5. invalid aspect key rejected ----------
raw5 = json.dumps({"aspects": [
    {"aspect": "vibes", "keywords": "parkir", "sentiment": "neg", "span": "parkirnya sempit"},
]})
r5 = parse_response(raw5, s)
ck("unknown aspect rejected", r5["aspects"] == [], str(r5))

# ---------- 6. invalid sentiment rejected ----------
raw6 = json.dumps({"aspects": [
    {"aspect": "facilities", "keywords": "parkir", "sentiment": "bad", "span": "parkirnya sempit"},
]})
r6 = parse_response(raw6, s)
ck("invalid sentiment rejected", r6["aspects"] == [], str(r6))

# ---------- 7. partial: one good, one bad ----------
raw7 = json.dumps({"aspects": [
    {"aspect": "cleanliness", "keywords": "bersih", "sentiment": "pos", "span": "bersih banget"},
    {"aspect": "facilities", "keywords": "PARAPHRASE", "sentiment": "neg", "span": "parkirnya sempit"},
]})
r7 = parse_response(raw7, s)
ck("partial keeps the valid one", len(r7["aspects"]) == 1 and r7["status"] == "partial", str(r7))

# ---------- 8. empty aspects ----------
r8 = parse_response('{"aspects": []}', "Selamat pagi semua.")
ck("empty is valid", r8["aspects"] == [] and r8["status"] == "ok", str(r8))

# ---------- 9. code fence tolerated ----------
fenced = "```json\n" + json.dumps({"aspects": [
    {"aspect": "cleanliness", "keywords": "bersih", "sentiment": "pos", "span": "bersih banget"}]}) + "\n```"
r9 = parse_response(fenced, s)
ck("code fence parsed", len(r9["aspects"]) == 1, str(r9))

# ---------- 10. garbage handled ----------
r10 = parse_response("I cannot help with that.", s)
ck("no json -> badjson", r10["status"] == "badjson", str(r10["status"]))
r11 = parse_response("", s)
ck("empty -> empty", r11["status"] == "empty")

# ---------- 10b. duplicated answer must not become "Extra data" ----------
two = json.dumps({"aspects": [
    {"aspect": "cleanliness", "keywords": "bersih", "sentiment": "pos", "span": "bersih banget"}]})
r10b = parse_response(two + "\n\n" + two, s)
ck("repeated object parsed, not rejected", len(r10b["aspects"]) == 1 and r10b["status"] == "ok",
   str(r10b["status"]))
r10c = parse_response("Here you go:\n" + two + "\nHope this helps {ok}.", s)
ck("prose around json tolerated", len(r10c["aspects"]) == 1, str(r10c["status"]))
r10d = parse_response('{"aspects": []}\n{"aspects": []}', "Selamat pagi.")
ck("two empty objects tolerated", r10d["status"] == "ok", str(r10d["status"]))

# ---------- 11. near-verbatim copy is recovered to the source slice ----------
raw12 = json.dumps({"aspects": [
    {"aspect": "cleanliness", "keywords": "Bersih", "sentiment": "pos", "span": "Pantainya bersih banget"},
]})
r12 = parse_response(raw12, s)
ck("case drift recovered", len(r12["aspects"]) == 1, str(r12))
ck("recovered span is exact substring",
   r12["aspects"] and r12["aspects"][0]["span"] in s, str(r12["aspects"]))
ck("recovered span keeps source casing",
   r12["aspects"] and r12["aspects"][0]["span"] == "Pantainya bersih banget", str(r12["aspects"]))

raw13 = json.dumps({"aspects": [
    {"aspect": "facilities", "keywords": "parkirnya", "sentiment": "neg", "span": "parkirnya  sempit"},
]})
r13 = parse_response(raw13, s)
ck("whitespace drift recovered", len(r13["aspects"]) == 1, str(r13))

# ---------- 12. every worked example in the prompt must pass the parser ----------
import re as _re
pairs = _re.findall(r'Sentence: "(.*?)"\n(\{.*?\})\n', SYSTEM_PROMPT, _re.DOTALL)
ck("prompt teaches by worked example", len(pairs) >= 4, str(len(pairs)))
for i, (ex_s, ex_j) in enumerate(pairs):
    ex_r = parse_response(ex_j, ex_s)
    ck(f"example {i+1} parses cleanly", ex_r["status"] == "ok", f"{ex_r['status']} {ex_r['error']}")
    for a in ex_r["aspects"]:
        ck(f"example {i+1} {a['aspect']} span is verbatim", a["span"] in ex_s)
        ck(f"example {i+1} {a['aspect']} keyword is verbatim", a["keywords"] in ex_s)

# ---------- 13. prompt contract ----------
msgs = build_messages(s)
ck("prompt has 2 roles", [m["role"] for m in msgs] == ["system", "user"])
ck("sentence in user msg", s in msgs[1]["content"])
ck("all 12 aspects listed", all(a in msgs[0]["content"] for a in ASPECTS))
ck("glossary shown to model", "kebersihan:" in msgs[0]["content"])
ck("verbatim rule stated", "VERBATIM" in msgs[0]["content"])
ck("no star rating leaked into prompt", "star" not in msgs[1]["content"].lower())

# ---------- 14. dedupe keeps opinions, drops fragmentation ----------
from sentence_absa import dedupe_aspects  # noqa: E402

frag = [
    {"aspect": "atmosphere", "keywords": "Suasananya tenang", "sentiment": "pos", "span": "Suasananya tenang"},
    {"aspect": "atmosphere", "keywords": "luas", "sentiment": "pos", "span": "luas"},
    {"aspect": "atmosphere", "keywords": "enak buat jalan santai", "sentiment": "pos", "span": "enak buat jalan santai"},
]
dd = dedupe_aspects(frag)
ck("fragmented aspect collapsed to one", len(dd) == 1, str(len(dd)))
ck("longest span kept", dd[0]["span"] == "enak buat jalan santai", dd[0]["span"])

mixed = [
    {"aspect": "facilities", "keywords": "Toiletnya", "sentiment": "pos", "span": "Toiletnya bersih"},
    {"aspect": "facilities", "keywords": "musalanya", "sentiment": "neg", "span": "musalanya bau"},
]
ck("opposite polarity within an aspect survives", len(dedupe_aspects(mixed)) == 2)

ck("dedupe is order-stable for distinct aspects",
   len(dedupe_aspects(mixed + frag)) == 3)

print("\n" + ("ALL PARSER CHECKS PASSED" if not fails else f"FAILED: {fails}"))
raise SystemExit(1 if fails else 0)
