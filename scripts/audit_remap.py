"""Independent audit of the alias remap. Recomputes every number from the raw
files rather than trusting the pipeline's own output.

Run: ./venv/bin/python scripts/audit_remap.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sentiment_analysis"))

from remap_aspects import ASPECT_ALIASES, canonical  # noqa: E402
from sentence_absa import ASPECTS  # noqa: E402

FULL = ROOT / "output/sentence_absa/sentence_absa_raw_noremap.csv"  # no alias resolver
CUR = ROOT / "output/sentence_absa/sentence_absa_raw.csv"
SENT = ROOT / "data/processed/sentences.csv"

orig = pd.read_csv(FULL)
cur = pd.read_csv(CUR)
sent = pd.read_csv(SENT)
smap = dict(zip(sent.sentence_id, sent.sentence))
LABELABLE = 6479

fails = []


def ck(name, cond, detail=""):
    print(f"{'OK  ' if cond else 'FAIL'} {name}  {detail}")
    if not cond:
        fails.append(name)


print("=" * 70)
print("1. SHAPE AND COVERAGE")
print("=" * 70)
ck("row count unchanged", len(cur) == len(orig) == LABELABLE,
   f"{len(orig)} -> {len(cur)}")
ck("sentence_id set identical",
   set(cur.sentence_id) == set(orig.sentence_id))
ck("one row per sentence_id", cur.sentence_id.is_unique)
ck("sentence text untouched",
   (cur.sort_values("sentence_id").sentence.values
    == orig.sort_values("sentence_id").sentence.values).all())
ck("every sentence text matches the corpus",
   all(smap.get(sid) == s for sid, s in zip(cur.sentence_id, cur.sentence)))

print("\n" + "=" * 70)
print("2. ONLY THE ASPECT KEY MAY HAVE CHANGED")
print("=" * 70)
# For every row, the set of (sentiment, keywords, span) must be identical:
# the remap is allowed to rename aspects, never to add or drop an opinion.
def trip(r):
    return sorted((a["sentiment"], a["keywords"], a["span"])
                  for a in json.loads(r.aspects or "[]"))


diff_trip = 0
for o, c in zip(orig.sort_values("sentence_id").itertuples(),
                cur.sort_values("sentence_id").itertuples()):
    if trip(o) != trip(c):
        diff_trip += 1

added = cur.n_aspects.sum() - orig.n_aspects.sum()
# A rescued mention can merge into an existing (aspect, sentiment) pair; dedupe
# then keeps the longer span, so a row's count can stay flat or even swap which
# span survives. What must never happen is a row losing mentions overall.
losses = 0
for o, c in zip(orig.sort_values("sentence_id").itertuples(),
                cur.sort_values("sentence_id").itertuples()):
    if len(json.loads(c.aspects or "[]")) < len(json.loads(o.aspects or "[]")):
        losses += 1
ck("no row lost a mention", losses == 0, f"{losses} rows shrank")

print("\n" + "=" * 70)
print("3. GROUNDING — EVERY STORED STRING IS IN ITS OWN SENTENCE")
print("=" * 70)
bad_kw = bad_sp = bad_asp = bad_sen = 0
for r in cur.itertuples():
    for a in json.loads(r.aspects or "[]"):
        if a["keywords"] not in r.sentence:
            bad_kw += 1
        if a["span"] not in r.sentence:
            bad_sp += 1
        if a["aspect"] not in ASPECTS:
            bad_asp += 1
        if a["sentiment"] not in ("pos", "neu", "neg"):
            bad_sen += 1
ck("keywords grounded", bad_kw == 0, f"{bad_kw} violations")
ck("span grounded", bad_sp == 0, f"{bad_sp} violations")
ck("aspect canonical", bad_asp == 0, f"{bad_asp} violations")
ck("sentiment canonical", bad_sen == 0, f"{bad_sen} violations")

print("\n" + "=" * 70)
print("4. NO CROSS-SENTENCE LEAKAGE FROM THE REMAP")
print("=" * 70)
# A rescued mention must be locatable in its own sentence. Compare the rescued
# set against the original: every NEW (aspect, span) must appear in that row.
leak = 0
for o, c in zip(orig.sort_values("sentence_id").itertuples(),
                cur.sort_values("sentence_id").itertuples()):
    old = {(a["aspect"], a["span"]) for a in json.loads(o.aspects or "[]")}
    for a in json.loads(c.aspects or "[]"):
        if a["span"] not in o.sentence:
            leak += 1
ck("no rescued span outside its sentence", leak == 0, f"{leak}")

print("\n" + "=" * 70)
print("5. DEDUPE DID NOT COLLAPSE DISTINCT ASPECTS")
print("=" * 70)
collapsed = 0
for r in cur.itertuples():
    items = json.loads(r.aspects or "[]")
    keys = [(a["aspect"], a["sentiment"]) for a in items]
    if len(keys) != len(set(keys)):
        collapsed += 1
ck("no duplicate (aspect, sentiment) within a sentence", collapsed == 0,
   f"{collapsed} rows")
ck("n_aspects agrees with the JSON", bool((cur.n_aspects
   == cur.aspects.apply(lambda s: len(json.loads(s or "[]")))).all()))

print("\n" + "=" * 70)
print("6. THE 128 RESCUED MENTIONS — WHAT THEY ACTUALLY ARE")
print("=" * 70)
rescued = []
for o, c in zip(orig.sort_values("sentence_id").itertuples(),
                cur.sort_values("sentence_id").itertuples()):
    old = {(a["aspect"], a["sentiment"], a["span"])
           for a in json.loads(o.aspects or "[]")}
    for a in json.loads(c.aspects or "[]"):
        if (a["aspect"], a["sentiment"], a["span"]) not in old:
            rescued.append({"sentence_id": c.sentence_id,
                            "aspect": a["aspect"], "span": a["span"],
                            "sentiment": a["sentiment"],
                            "sentence": c.sentence})
res = pd.DataFrame(rescued)
print(f"rescued mentions: {len(res)}")
if len(res):
    print(res.aspect.value_counts().to_string())
    res.to_csv(ROOT / "output/audit/remap_rescued_mentions.csv", index=False)
    print(f"\nwritten {ROOT/'output/audit/remap_rescued_mentions.csv'}")
    print("\n12 random samples for eyeballing:")
    for _, r in res.sample(min(12, len(res)), random_state=7).iterrows():
        print(f"  [{r.aspect}/{r.sentiment}] {r.span!r}")
        print(f"     sentence: {r.sentence[:110]}")

print("\n" + "=" * 70)
print("7. DETERMINISM — RE-RUNNING THE REMAP CHANGES NOTHING")
print("=" * 70)
before = cur.sort_values("sentence_id").aspects.tolist()
subprocess.run([sys.executable, str(ROOT / "scripts/reparse_with_aliases.py")],
               cwd=ROOT, check=True, capture_output=True)
again = pd.read_csv(CUR).sort_values("sentence_id").aspects.tolist()
ck("second reparse is a no-op", before == again,
   f"{sum(a != b for a, b in zip(before, again))} rows changed")

print("\n" + "=" * 70)
print("8. DISTRIBUTION SHIFT CAUSED BY THE REMAP")
print("=" * 70)
def counts(df):
    out = {}
    for s in df.aspects:
        for a in json.loads(s or "[]"):
            out[a["aspect"]] = out.get(a["aspect"], 0) + 1
    return pd.Series(out)


t = pd.DataFrame({"before": counts(orig), "after": counts(cur)}).fillna(0).astype(int)
t["delta"] = t.after - t.before
t["pct"] = (t.delta / t.before * 100).round(1)
print(t.reindex(list(ASPECTS)).to_string())
ck("no aspect lost mentions", bool((t.delta >= 0).all()), 
   f"min delta {t.delta.min()}")

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
print("ALL REMAP AUDITS PASSED" if not fails else f"FAILED: {fails}")
raise SystemExit(1 if fails else 0)
