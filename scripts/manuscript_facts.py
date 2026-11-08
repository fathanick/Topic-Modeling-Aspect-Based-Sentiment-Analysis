"""Collect every number the revised manuscript needs, from verified outputs.

No number here is typed by hand: each is recomputed from the artifacts that the
verifiers already pass on. Prints a dated fact sheet for the revision.

Run: ./venv/bin/python scripts/manuscript_facts.py
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "output"


def hdr(t):
    print("\n" + "=" * 62)
    print(t)
    print("=" * 62)


# ---------------------------------------------------------------- corpus
hdr("CORPUS")
rev = pd.read_csv(ROOT / "data/processed/reviews_clean.csv") if (ROOT / "data/processed/reviews_clean.csv").exists() else None
sents = pd.read_csv(ROOT / "data/processed/sentences.csv")
print(f"kalimat            {len(sents)}")
print(f"review             {sents.review_id.nunique()}")
print(f"destinasi          {sents.destination.nunique()}")
if rev is not None:
    print(f"reviews_clean cols {list(rev.columns)[:8]}")

# stars from the review-level file used earlier
for cand in ["data/processed/reviews_with_stars.csv", "data/processed/reviews_clean.csv"]:
    p = ROOT / cand
    if p.exists():
        d = pd.read_csv(p)
        if "stars" in d.columns:
            print(f"stars: mean {d.stars.mean():.2f} | 5star {100*(d.stars==5).mean():.1f}%  ({cand})")
            break

# ---------------------------------------------------------------- topics
hdr("TOPIC MODEL")
tm = json.loads((O / "topic_modeling_final/topic_coherence.json").read_text())
print(json.dumps(tm, indent=2, ensure_ascii=False)[:900])
ts = pd.read_csv(O / "topic_modeling_final/topics_summary.csv")
print(f"\ntopics_summary cols {list(ts.columns)}")
print(f"n_topics {len(ts)}")
print(ts.head(12).to_string())
man = json.loads((O / "topic_modeling_final/model_manifest.json").read_text())
print("\nmanifest:", json.dumps(man, ensure_ascii=False)[:700])

# ---------------------------------------------------------------- mentions
hdr("SENTENCE-LEVEL MENTIONS")
km = pd.read_csv(O / "sentence_absa/kalimat_aspek_sentimen.csv")
raw = pd.read_csv(O / "sentence_absa/sentence_absa_raw.csv")
print(f"kalimat dilabeli   {len(raw)}")
print(f"kalimat >=1 aspek  {int((raw.n_aspects>0).sum())}")
print(f"kalimat nol aspek  {int((raw.n_aspects==0).sum())}")
print(f"mention            {len(km)}")
print(f"mention/kalimat    {len(km)/len(raw):.2f}")
print(f"mention/review     {len(km)/raw.review_id.nunique():.2f}")
print(f"review             {raw.review_id.nunique()}")

hdr("SENTIMENT DISTRIBUTION (mentions)")
vc = km.sentiment.value_counts()
for k, v in vc.items():
    print(f"  {k:5s} {v:6d}  {100*v/len(km):5.1f}%")

hdr("ASPECT x SENTIMENT")
p = (km.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0))
p["total"] = p.sum(1)
for c in ["pos", "neu", "neg"]:
    if c not in p:
        p[c] = 0
p["pos_pct"] = 100 * p["pos"] / p["total"]
p["neg_pct"] = 100 * p["neg"] / p["total"]
p["neu_pct"] = 100 * p["neu"] / p["total"]
p["score"] = (p["pos"] - p["neg"]) / p["total"]
p = p.sort_values("total", ascending=False)
cols = ["total", "pos", "neu", "neg", "pos_pct", "neg_pct", "neu_pct", "score"]
print(p[cols].round(2).to_string())

hdr("MULTI-ASPECT / MIXED POLARITY")
per = km.groupby("sentence_id").agg(n_aspect=("aspect", "size"),
                                    kinds=("sentiment", lambda s: len(set(s))))
print(f"kalimat >=2 aspek  {int((per.n_aspect>=2).sum())}  ({100*(per.n_aspect>=2).mean():.1f}% of {len(raw)})")
print(f"  positif DAN negatif {int((per.kinds>=2).sum())}")

hdr("NO-ASPECT SENTENCES")
al = pd.read_csv(O / "audit/aspectless_sentences.csv")
na = pd.read_csv(O / "sentence_absa/no_aspect_sentiment.csv")
print(al.cls.value_counts().to_string())
print(al.reason.value_counts().to_string())
print(na.sentence_sentiment.value_counts().to_string())

hdr("DESTINATIONS")
print(f"destinasi di korpus {sents.destination.nunique()}")
sb = pd.read_csv(O / "topic_modeling_final/topic_share_by_destination.csv")
print(f"topic_share_by_destination rows {len(sb)}")
xd = pd.read_csv(O / "topic_modeling_final/topic_x_destination_pct.csv")
print(f"topic x destination {xd.shape}")
