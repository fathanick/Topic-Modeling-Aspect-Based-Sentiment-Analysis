"""Verify every number in RESEARCH_DIRECTION_v3.md against the data."""
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASP = ["cleanliness", "facilities", "price", "scenery", "atmosphere", "accessibility",
       "crowd", "safety", "service", "food", "historical_value", "photo_spot"]
fail = []


def ck(label, got, want, tol=0.05):
    ok = abs(got - want) <= tol
    print(f"{'OK ' if ok else 'FAIL'} {label}: {got} vs {want}")
    if not ok:
        fail.append(label)


ab = pd.read_csv(ROOT / "output/absa_llm_gpu/absa_llm_results.csv")
counts = [sum(1 for a in ASP if isinstance(r[f"{a}_sentiment"], str)
              and r[f"{a}_sentiment"].strip()) for _, r in ab.iterrows()]
ck("reviews", len(ab), 1530, 0)
ck("mentions", sum(counts), 5691, 0)
ck("% with >1 aspect", round(sum(1 for x in counts if x > 1) / len(ab) * 100, 1), 84.2)
ck("mean aspects", round(sum(counts) / len(counts), 2), 3.72, 0.01)

het = pd.read_csv(ROOT / "output/audit/within_review_heterogeneity.csv")
multi = het[het.n >= 2]
ck("reviews >=2 aspects", len(multi), 1288, 0)
ck("mixed polarity", int(multi.mixed_polarity.sum()), 375, 0)
ck("mixed %", round(multi.mixed_polarity.mean() * 100, 1), 29.1)
ck("all same", int(multi.all_same.sum()), 609, 0)
ck("all same %", round(multi.all_same.mean() * 100, 1), 47.3)
ck("median majority share", round(multi.majority_share.median(), 2), 0.83, 0.01)
ck("mean majority share", round(multi.majority_share.mean(), 2), 0.81, 0.01)

div = pd.read_csv(ROOT / "output/divergence/aspect_disagreement.csv")
g = div.groupby("review_id")["llm"]
share = g.agg(lambda s: s.value_counts().max() / len(s))
n = g.size()
ceiling = (share * n).sum() / n.sum() * 100
ck("ceiling", round(ceiling, 1), 83.6, 0.1)
ck("observed agreement", round((~div.disagree).mean() * 100, 1), 77.2)
ck("gap", round(ceiling - (~div.disagree).mean() * 100, 1), 6.4, 0.15)
ck("comparable labels", len(div), 3638, 0)

# sentence level
def sentences(text):
    return [(m.start(), m.end()) for m in re.finditer(r"[^.!?\n]+[.!?]*", text)
            if text[m.start():m.end()].strip()]


tot_sent, located, unloc = 0, 0, 0
per_sent, sent_multi, sent_opp = [], 0, 0
for _, r in ab.iterrows():
    text = str(r.original_text)
    sents = sentences(text)
    tot_sent += len(sents)
    bucket = {i: [] for i in range(len(sents))}
    for a in ASP:
        lab, ev = r.get(f"{a}_sentiment"), r.get(f"{a}_evidence")
        if not isinstance(lab, str) or not lab.strip():
            continue
        if not isinstance(ev, str) or not ev.strip():
            continue
        i = text.find(ev.strip())
        if i < 0:
            unloc += 1
            continue
        j = i + len(ev.strip())
        for k, (s, e) in enumerate(sents):
            if i >= s and j <= e + 1:
                bucket[k].append((a, lab.strip()))
                located += 1
                break
    for k, items in bucket.items():
        if items:
            per_sent.append(len(items))
        if len({a for a, _ in items}) >= 2:
            sent_multi += 1
            labs = {l for _, l in items}
            if "positive" in labs and "negative" in labs:
                sent_opp += 1

ck("total sentences", tot_sent, 5862, 0)
ck("sentences with >=1 aspect", len(per_sent), 2530, 0)
ck("spans located", located, 3435, 0)
ck("spans unlocatable", unloc, 1029, 0)
ck("sentences >=2 aspects", sent_multi, 628, 0)
ck("sentences opposite polarity", sent_opp, 116, 0)
ck("% >=2 aspects", round(sent_multi / len(per_sent) * 100, 1), 24.8)
ck("% opposite among multi", round(sent_opp / sent_multi * 100, 1), 18.5)
ck("mean aspects/sentence", round(sum(per_sent) / len(per_sent), 2), 1.36, 0.01)

print("\nALL OK" if not fail else f"\nFAILED: {fail}")
raise SystemExit(1 if fail else 0)
