"""Verify every number in DRAFT_v3_abstract_intro_results.md."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASP = ["cleanliness", "facilities", "price", "scenery", "atmosphere", "accessibility",
       "crowd", "safety", "service", "food", "historical_value", "photo_spot"]
fail = []


def ck(label, got, want, tol=0.05):
    ok = abs(got - want) <= tol
    if not ok:
        fail.append(label)
    print(f"{'OK  ' if ok else 'FAIL'} {label}: {got} vs {want}")


ab = pd.read_csv(ROOT / "output/absa_llm_gpu/absa_llm_results.csv")
div = pd.read_csv(ROOT / "output/divergence/aspect_disagreement.csv")
het = pd.read_csv(ROOT / "output/audit/within_review_heterogeneity.csv")

# corpus
ck("reviews", len(ab), 1530, 0)
counts = [sum(1 for a in ASP if isinstance(r[f"{a}_sentiment"], str)
              and r[f"{a}_sentiment"].strip()) for _, r in ab.iterrows()]
ck("mentions", sum(counts), 5691, 0)
ck("mean aspects", round(sum(counts) / len(counts), 2), 3.72, 0.01)
ck("multi-aspect", sum(1 for x in counts if x > 1), 1288, 0)
ck("multi pct", round(sum(1 for x in counts if x > 1) / len(ab) * 100, 1), 84.2)
ck("zero-aspect reviews", sum(1 for x in counts if x == 0), 29, 0)

m = het[het.n >= 2]
ck("mixed polarity", int(m.mixed_polarity.sum()), 375, 0)
ck("mixed pct", round(m.mixed_polarity.mean() * 100, 1), 29.1)
ck("uniform", int(m.all_same.sum()), 609, 0)
ck("uniform pct", round(m.all_same.mean() * 100, 1), 47.3)
ck("majority share mean", round(m.majority_share.mean(), 2), 0.81, 0.01)
ck("majority share median", round(m.majority_share.median(), 2), 0.83, 0.01)

# ceiling
ck("comparable", len(div), 3638, 0)
g = div.groupby("review_id")["llm"]
share = g.agg(lambda s: s.value_counts().max() / len(s))
n = g.size()
ceil = (share * n).sum() / n.sum() * 100
ck("ceiling", round(ceil, 1), 83.6, 0.1)
ck("ceiling matches", round((share * n).sum()), 3041, 0)
ck("observed", round((~div.disagree).mean() * 100, 1), 77.2)
ck("gap", round(ceil - (~div.disagree).mean() * 100, 1), 6.4, 0.15)
ck("unrepresentable", round(100 - ceil, 1), 16.4, 0.15)

# coverage
labelled = len(div)
ck("silence count", 5691 - labelled, 2053, 0)
ck("silence pct", round((5691 - labelled) / 5691 * 100, 1), 36.1)
svc = div[div.aspect == "service"]
ck("service comparable", len(svc), 123, 0)
ck("service silence pct", round((414 - 123) / 414 * 100, 1), 70.3)

# disagreement by aspect
for a, want in (("accessibility", 42.1), ("crowd", 41.8), ("price", 41.3),
                ("safety", 39.4), ("scenery", 6.6), ("service", 14.6)):
    got = round(div[div.aspect == a].disagree.mean() * 100, 1)
    ck(f"disagree {a}", got, want, 0.15)

# bands
for band, want in ((1, 30.2), (2, 50.0), (3, 49.5), (4, 36.4), (5, 18.7)):
    got = round(div[div.stars == band].disagree.mean() * 100, 1)
    ck(f"band {band}", got, want, 0.15)

# complaint markers
ck("marker disagree", round(div[div.disagree].evidence_has_neg_marker.mean() * 100, 1), 39.5, 0.15)
ck("marker agree", round(div[~div.disagree].evidence_has_neg_marker.mean() * 100, 1), 21.8, 0.15)
ck("evidence len disagree", round(div[div.disagree].evidence_len.mean(), 1), 47.6, 0.1)
ck("evidence len agree", round(div[~div.disagree].evidence_len.mean(), 1), 40.0, 0.1)

# ranking
rk = pd.read_csv(ROOT / "output/divergence/rank_inversions.csv")
ck("rank inversions", len(rk), 12, 0)

# scores
pos = div[div.llm == "positive"].shape[0] / len(div) * 100
ck("llm positive pct", round(pos, 1), 72.4, 0.2)

print("\nALL OK" if not fail else f"\nFAILED: {fail}")
raise SystemExit(1 if fail else 0)
