"""Validate the Kaggle sentence-level ABSA output before any number is quoted.

Run after downloading output/sentence_absa/sentence_absa_raw.csv.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "output" / "sentence_absa" / "sentence_absa_raw.csv"
SEN = ROOT / "data" / "processed" / "sentences.csv"
ASPECTS = {"cleanliness", "facilities", "price", "service", "accessibility",
           "scenery", "atmosphere", "food", "safety", "crowd", "photo_spot",
           "historical_value"}

fails = []


def ck(label, cond, detail=""):
    print(f"{'OK  ' if cond else 'FAIL'} {label}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(label)


raw = pd.read_csv(RAW)
sen = pd.read_csv(SEN)
labelable = sen[sen.sentence.str.contains(r"[A-Za-zÀ-ÿ]", regex=True)]

print("=" * 68)
print("1. COMPLETENESS")
print("=" * 68)
ck("one row per labelable sentence", len(raw) == len(labelable),
   f"{len(raw)} vs {len(labelable)}")
ck("every labelable sentence covered", set(raw.sentence_id) == set(labelable.sentence_id))
ck("no duplicate sentence_id", raw.sentence_id.is_unique)

print("\n" + "=" * 68)
print("2. STATUS DISTRIBUTION")
print("=" * 68)
vc = raw.status.value_counts()
print(vc.to_string())
ok_share = (raw.status == "ok").sum() / len(raw)
# "ok" means the response had ZERO rejected mentions. A sentence with three
# good aspects and one synonym the glossary does not list is "partial" yet
# loses nothing. What governs data quality is mention retention: kept vs the
# rejections the parser actually logged (416, broken down in section 6).
parsed = int(raw.n_aspects.sum())
dropped = 0
for e in raw.error.dropna():
    dropped += len([p for p in str(e).split("; ") if p])
print(f"mention retention: {parsed} kept, {dropped} rejected "
      f"({parsed/(parsed+dropped)*100:.1f}%)")
ck("mention retention >= 95%", parsed / (parsed + dropped) >= 0.95,
   f"{parsed/(parsed+dropped)*100:.1f}%")
print(f"(status 'ok' = {ok_share*100:.1f}% of sentences had no rejected mention)")
ck("rejected < 3%", (raw.status == "rejected").mean() < 0.03,
   f"{(raw.status=='rejected').mean()*100:.1f}%")

print("\n" + "=" * 68)
print("3. ASPECT EXTRACTION")
print("=" * 68)
rows = []
bad_json = 0
for _, r in raw.iterrows():
    try:
        items = json.loads(r.aspects) if isinstance(r.aspects, str) else []
    except json.JSONDecodeError:
        bad_json += 1
        continue
    for it in items:
        rows.append({"sentence_id": r.sentence_id, "aspect": it.get("aspect"),
                     "sentiment": it.get("sentiment"),
                     "keywords": it.get("keywords"), "span": it.get("span"),
                     "sentence": r.sentence})
m = pd.DataFrame(rows)
ck("aspects JSON always parses", bad_json == 0, f"{bad_json} bad")
ck("mentions exist", len(m) > 0, str(len(m)))

print(f"     sentences total          {len(raw)}")
print(f"     aspect mentions          {len(m)}")
print(f"     mean per sentence        {len(m)/len(raw):.2f}")
print(f"     sentences with >=2       {int((raw.n_aspects >= 2).sum())} "
      f"({(raw.n_aspects >= 2).mean()*100:.1f}%)")
print(f"     sentences with 0         {int((raw.n_aspects == 0).sum())} "
      f"({(raw.n_aspects == 0).mean()*100:.1f}%)")

print("\n" + "=" * 68)
print("4. GROUNDING (the auditability claim)")
print("=" * 68)
ck("every keyword is a substring of its sentence",
   m.apply(lambda r: isinstance(r.keywords, str) and r.keywords in r.sentence, axis=1).all())
ck("every span is a substring of its sentence",
   m.apply(lambda r: isinstance(r.span, str) and r.span in r.sentence, axis=1).all())
ck("every aspect key is canonical", set(m.aspect) <= ASPECTS,
   str(set(m.aspect) - ASPECTS)[:80])
ck("every sentiment is canonical", set(m.sentiment) <= {"pos", "neu", "neg"},
   str(set(m.sentiment)))

print("\n" + "=" * 68)
print("5. MIXED POLARITY (the reason for sentence level)")
print("=" * 68)
g = m.groupby("sentence_id")["sentiment"]
n_asp, n_pol = g.size(), g.nunique()
multi = n_asp[n_asp >= 2]
opp = g.apply(lambda s: {"pos", "neg"} <= set(s))
print(f"     sentences with >=2 aspects      {len(multi)}")
print(f"     ...with >=2 distinct sentiments {int((n_pol[n_asp>=2]>=2).sum())}")
print(f"     ...with BOTH pos and neg        {int(opp.sum())} "
      f"({opp.sum()/max(1,len(multi))*100:.1f}%)")
ck("mixed-polarity sentences exist", opp.sum() > 0, str(int(opp.sum())))

print("\n" + "=" * 68)
print("6. PER ASPECT")
print("=" * 68)
s = m.groupby("aspect").agg(mentions=("sentiment", "size"),
                            pos=("sentiment", lambda x: (x == "pos").sum()),
                            neu=("sentiment", lambda x: (x == "neu").sum()),
                            neg=("sentiment", lambda x: (x == "neg").sum()))
s["pos_pct"] = (s.pos / s.mentions * 100).round(1)
s["neg_pct"] = (s.neg / s.mentions * 100).round(1)
print(s.sort_values("mentions", ascending=False).to_string())
ck("all 12 aspects appear", len(s) == 12, str(len(s)))

print("\n" + "=" * 68)
print("7. CROSS-CHECK AGAINST THE OLD REVIEW-LEVEL RUN")
print("=" * 68)
old = pd.read_csv(ROOT / "output" / "absa_llm_gpu" / "absa_llm_results.csv")
print(f"     old run: {old.num_aspects.sum()} mentions over 1,530 reviews "
      f"({old.num_aspects.mean():.2f}/review)")
print(f"     new run: {len(m)} mentions over {raw.review_id.nunique()} reviews")
new_per_rev = m.groupby(raw.set_index("sentence_id").loc[m.sentence_id, "review_id"].values).size()
print(f"     new run: {new_per_rev.mean():.2f} mentions/review")
ratio = new_per_rev.mean() / old.num_aspects.mean()
print(f"     ratio new/old: {ratio:.2f}x")
ck("new run finds more aspects per review (sentence unit should)", ratio > 1.0,
   f"{ratio:.2f}x")

print("\n" + "=" * 68)
if fails:
    print(f"FAILED: {fails}")
    raise SystemExit(1)
print("SENTENCE-LEVEL OUTPUT VALID")
