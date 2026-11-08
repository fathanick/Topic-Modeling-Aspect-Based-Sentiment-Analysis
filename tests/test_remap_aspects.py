"""Self-check for the aspect alias resolver.

Run: ./venv/bin/python tests/test_remap_aspects.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sentiment_analysis"))

from remap_aspects import ASPECT_ALIASES, UNMAPPED, canonical  # noqa: E402
from sentence_absa import ASPECTS, dedupe_aspects, parse_response  # noqa: E402

S = "Pelayanan petugasnya ramah dan penginapannya bersih."

# 1. every alias lands on a real glossary key
bad = {k: v for k, v in ASPECT_ALIASES.items() if v not in ASPECTS}
assert not bad, f"alias points at unknown aspect: {bad}"

# 2. no key is both mapped and deliberately unmapped
assert not (set(ASPECT_ALIASES) & UNMAPPED), set(ASPECT_ALIASES) & UNMAPPED

# 3. glossary keys pass through untouched
for k in ASPECTS:
    assert canonical(k) == k, k
    assert canonical(k.upper()) == k, k

# 4. a synonym is accepted through the resolver and rejected without it
raw = json.dumps({"aspects": [
    {"aspect": "services", "keywords": "petugasnya ramah",
     "sentiment": "pos", "span": "Pelayanan petugasnya ramah"},
]})
strict = parse_response(raw, S)
lenient = parse_response(raw, S, resolver=canonical)
assert len(strict["aspects"]) == 0, strict
assert len(lenient["aspects"]) == 1, lenient
assert lenient["aspects"][0]["aspect"] == "service"

# 5. aliasing must NOT weaken grounding: a paraphrased span is still rejected
raw_bad = json.dumps({"aspects": [
    {"aspect": "services", "keywords": "pelayanan bagus",
     "sentiment": "pos", "span": "pelayanan bagus"},
]})
assert parse_response(raw_bad, S, resolver=canonical)["aspects"] == []

# 6. unmapped keys stay unmapped
for k in UNMAPPED:
    assert canonical(k) == "", k

# 7. aliasing never invents an aspect the model did not emit
raw_empty = json.dumps({"aspects": []})
assert parse_response(raw_empty, S, resolver=canonical)["aspects"] == []

# 8. two synonyms of the same aspect collapse under dedupe instead of doubling
raw_dup = json.dumps({"aspects": [
    {"aspect": "accommodation", "keywords": "penginapannya",
     "sentiment": "pos", "span": "penginapannya bersih"},
    {"aspect": "facilities", "keywords": "penginapannya",
     "sentiment": "pos", "span": "penginapannya bersih"},
]})
got = dedupe_aspects(parse_response(raw_dup, S, resolver=canonical)["aspects"])
assert len(got) == 1, got
assert got[0]["aspect"] == "facilities"

print(f"OK   {len(ASPECT_ALIASES)} aliases all resolve to a glossary key")
print(f"OK   {len(UNMAPPED)} keys deliberately left unmapped")
print("OK   strict parser still rejects synonyms; resolver accepts them")
print("OK   grounding unchanged: paraphrased span rejected either way")
print("OK   aliasing cannot invent an aspect")
print("OK   two synonyms collapse to one mention")
print("\nALL REMAP CHECKS PASSED")
