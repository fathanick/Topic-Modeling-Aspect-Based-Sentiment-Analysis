"""Confidence intervals for the adjudication proportions, and the disagreement audit."""
import math
import pandas as pd

s = pd.read_csv("data/annotation/adjudication_completed.csv")
k = pd.read_csv("data/annotation/adjudication_key_DO_NOT_SHARE.csv").drop(columns=["aspect"])
d = s.merge(k, on="item_id")
BIN = {"yes": "complaint", "no": "no_complaint", "absent": "no_complaint", "unclear": None}


def wilson(k_, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k_ / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))


def verdict(row, col):
    v = BIN.get(row[col])
    if v is None:
        return "undecided"
    if v == "complaint":
        return "aspect-level" if row["_llm_label"] == "negative" else "review-level"
    return "review-level" if row["_llm_label"] == "negative" else "aspect-level"


print("=== proportion supporting the aspect-level labeller ===")
for col, name in (("contains_complaint_annotator1", "annotator 1"),
                  ("contains_complaint_annotator2", "annotator 2")):
    d["v"] = d.apply(lambda r: verdict(r, col), axis=1)
    dec = d[d.v != "undecided"]
    kk = (dec.v == "aspect-level").sum()
    lo, hi = wilson(kk, len(dec))
    print(f"  {name}: {kk}/{len(dec)} = {kk/len(dec)*100:.1f}%  Wilson 95% CI [{lo*100:.1f}, {hi*100:.1f}]")

both = d[(d.contains_complaint_annotator1 == d.contains_complaint_annotator2) &
         (d.contains_complaint_annotator1 != "unclear")].copy()
both["v"] = both.apply(lambda r: verdict(r, "contains_complaint_annotator1"), axis=1)
kk = (both.v == "aspect-level").sum()
lo, hi = wilson(kk, len(both))
print(f"  consensus only: {kk}/{len(both)} = {kk/len(both)*100:.1f}%  Wilson 95% CI [{lo*100:.1f}, {hi*100:.1f}]")

print("\n=== how often does a conflict change the direction? ===")
conf = d[d.contains_complaint_annotator1 != d.contains_complaint_annotator2].copy()
conf["v1"] = conf.apply(lambda r: verdict(r, "contains_complaint_annotator1"), axis=1)
conf["v2"] = conf.apply(lambda r: verdict(r, "contains_complaint_annotator2"), axis=1)
flip = ((conf.v1 != conf.v2) & conf.v1.isin(["aspect-level", "review-level"])
        & conf.v2.isin(["aspect-level", "review-level"]))
print(f"  conflicts: {len(conf)}/60 ({len(conf)/60*100:.0f}%)")
print(f"  conflicts that point to opposite labellers: {flip.sum()}")

print("\n=== disagreement structure (the substantive finding) ===")
cats = pd.crosstab(conf.contains_complaint_annotator1, conf.contains_complaint_annotator2)
# classify each conflict
def kind(r):
    a, b = r.contains_complaint_annotator1, r.contains_complaint_annotator2
    if "absent" in (a, b):
        return "mention boundary"
    if "unclear" in (a, b):
        return "evaluation vs description"
    return "complaint vs no-complaint"
conf["kind"] = conf.apply(kind, axis=1)
print(conf.kind.value_counts().to_string())
print(f"\n  mention boundary: {(conf.kind=='mention boundary').sum()}/{len(conf)} = "
      f"{(conf.kind=='mention boundary').sum()/len(conf)*100:.0f}% of all conflicts")
