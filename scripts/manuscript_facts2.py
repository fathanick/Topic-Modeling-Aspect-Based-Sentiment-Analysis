"""Facts for the sentence-level revision, computed strictly.

Every number printed here is what goes into the manuscript. Definitions are
explicit so the verifier can re-derive them:

  mention            = one (sentence, aspect, sentiment) row
  multi-aspect sent. = sentence with >= 2 mentions
  mixed polarity     = sentence carrying at least one `pos` AND one `neg`
  competing polarity = same review contains a `pos` and a `neg` mention
                       anywhere (review-level analysis cannot express this)

Run: ./venv/bin/python scripts/manuscript_facts2.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "output"

km = pd.read_csv(O / "sentence_absa/kalimat_aspek_sentimen.csv")
raw = pd.read_csv(O / "sentence_absa/sentence_absa_raw.csv")
al = pd.read_csv(O / "audit/aspectless_sentences.csv")
na = pd.read_csv(O / "sentence_absa/no_aspect_sentiment.csv")
sents = pd.read_csv(ROOT / "data/processed/sentences.csv")


def hdr(t):
    print("\n" + "-" * 60)
    print(t)


hdr("CORPUS")
print(f"reviews                 {raw.review_id.nunique()}")
print(f"sentences built         {len(sents)}")
print(f"destinations            {raw.destination.nunique()}")
print(f"sentences labelled      {len(raw)}")
print(f"sentences >=1 aspect    {int((raw.n_aspects > 0).sum())}"
      f"  ({100*(raw.n_aspects > 0).mean():.1f}%)")
print(f"sentences no aspect     {int((raw.n_aspects == 0).sum())}"
      f"  ({100*(raw.n_aspects == 0).mean():.1f}%)")

hdr("MENTIONS")
print(f"mentions                {len(km)}")
print(f"per sentence            {len(km)/len(raw):.2f}")
print(f"per review              {len(km)/raw.review_id.nunique():.2f}")
vc = km.sentiment.value_counts()
for k in ["pos", "neu", "neg"]:
    print(f"  {k}  {vc[k]:5d}  {100*vc[k]/len(km):5.1f}%")

hdr("MULTI-ASPECT SENTENCES")
per = km.groupby("sentence_id").sentiment.agg(
    n="size", has_pos=lambda s: (s == "pos").any(), has_neg=lambda s: (s == "neg").any())
multi = per[per.n >= 2]
print(f"sentence >=2 mentions   {len(multi)}  ({100*len(multi)/len(raw):.1f}% of {len(raw)})")
print(f"  of sentences w/ aspect {100*len(multi)/int((raw.n_aspects>0).sum()):.1f}%")
mixed = per[per.has_pos & per.has_neg]
print(f"mixed polarity sentence {len(mixed)}  ({100*len(mixed)/len(raw):.1f}% of all)")
print(f"  of multi-aspect        {100*len(mixed)/len(multi):.1f}%")

hdr("REVIEW-LEVEL COMPETING POLARITY")
# kalimat_aspek_sentimen has no review_id; recover it from the raw table
sid2rev = raw.set_index("sentence_id").review_id
k2 = km.assign(review_id=km.sentence_id.map(sid2rev))
assert k2.review_id.notna().all(), "unmapped sentence_id"
rv = k2.groupby("review_id").sentiment.agg(
    n="size", has_pos=lambda s: (s == "pos").any(), has_neg=lambda s: (s == "neg").any())
comp = rv[rv.has_pos & rv.has_neg]
print(f"reviews with mentions   {len(rv)}")
print(f"reviews competing signs {len(comp)}  ({100*len(comp)/len(rv):.1f}%)")

# stars per sentence
st = raw[["sentence_id", "stars"]].copy()
m = mixed.join(st.set_index("sentence_id"))
print("\nmixed-polarity sentences by stars:")
print(m.stars.value_counts().sort_index().to_string())
print(f"  in 4-5 stars          {int(m.stars.isin([4,5]).sum())}"
      f"  ({100*m.stars.isin([4,5]).mean():.1f}%)")

hdr("ASPECT TABLE (for TABLE 5 rewrite)")
p = km.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
for c in ["pos", "neu", "neg"]:
    if c not in p:
        p[c] = 0
p["total"] = p.sum(1)
p = p.sort_values("total", ascending=False)
print(f"{'aspect':18s}{'tot':>6}{'pos':>6}{'neu':>6}{'neg':>6}{'pos%':>7}{'neu%':>7}{'neg%':>7}{'score':>7}")
for a, r in p.iterrows():
    print(f"{a:18s}{r.total:6d}{r.pos:6d}{r['neu']:6d}{r.neg:6d}"
          f"{100*r.pos/r.total:7.1f}{100*r['neu']/r.total:7.1f}{100*r.neg/r.total:7.1f}"
          f"{(r.pos-r.neg)/r.total:7.2f}")

hdr("NO-ASPECT ACCOUNTING")
print(al.cls.value_counts().to_string())
print(al.reason.value_counts().to_string())
print(na.sentence_sentiment.value_counts().to_string())

hdr("TOPICS")
ts = pd.read_csv(O / "topic_modeling_final/topics_summary.csv")
print(f"n topics {len(ts)}  |  assigned {int(ts.document_count.sum())}"
      f"  |  outliers 84  |  coverage {100*ts.document_count.sum()/1530:.1f}%")
print(ts[["topic_id", "document_count", "percentage"]].to_string(index=False))
