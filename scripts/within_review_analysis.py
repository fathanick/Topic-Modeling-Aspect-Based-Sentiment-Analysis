"""Is a review-level label even capable of representing its aspects?

Measures the ceiling on any review-level labeller: if each review must carry one
label, how much aspect-level information is destroyed by construction?
"""
import re
import pandas as pd

ASP = ["cleanliness", "facilities", "price", "scenery", "atmosphere", "accessibility",
       "crowd", "safety", "service", "food", "historical_value", "photo_spot"]
POL = {"positive": 1, "neutral": 0, "negative": -1}

ab = pd.read_csv("output/absa_llm_gpu/absa_llm_results.csv")

rows = []
for _, r in ab.iterrows():
    present = {}
    for a in ASP:
        v = r.get(f"{a}_sentiment")
        if isinstance(v, str) and v.strip():
            present[a] = (v.strip(), r.get(f"{a}_evidence"))
    rows.append(present)

rec = []
for p, r in zip(rows, ab.itertuples()):
    labs = [v[0] for v in p.values()]
    n = len(labs)
    npos, nneu, nneg = labs.count("positive"), labs.count("neutral"), labs.count("negative")
    rec.append({
        "review_id": r.review_id, "n": n, "npos": npos, "nneu": nneu, "nneg": nneg,
        "mixed_polarity": npos > 0 and nneg > 0,
        "all_same": len(set(labs)) == 1 if n else False,
        "majority_share": max(npos, nneu, nneg) / n if n else None,
    })
t = pd.DataFrame(rec)
multi = t[t.n >= 2]

print("=== reviews with >=2 aspects ===")
print(f"n = {len(multi)}  ({len(multi)/len(t)*100:.1f}% of corpus)")
print(f"contain BOTH positive and negative aspects: {multi.mixed_polarity.sum()} "
      f"= {multi.mixed_polarity.mean()*100:.1f}%")
print(f"all aspects same polarity: {multi.all_same.sum()} = {multi.all_same.mean()*100:.1f}%")
print(f"\nmajority share of aspects within review:")
print(multi.majority_share.describe().to_string())

print("\n=== by aspect count ===")
g = multi.groupby("n").agg(reviews=("review_id", "size"),
                           mixed=("mixed_polarity", "sum"),
                           mixed_pct=("mixed_polarity", "mean"),
                           avg_majority=("majority_share", "mean"))
g["mixed_pct"] = (g.mixed_pct * 100).round(1)
g["avg_majority"] = g.avg_majority.round(3)
print(g.to_string())

# ceiling: best possible accuracy of ONE review-level label predicting each aspect
print("\n=== ceiling on any review-level labeller ===")
print("A single review label predicts each mention by its majority aspect polarity.")
ceiling_num = (t[["npos", "nneu", "nneg"]].max(axis=1) * (t.n > 0)).sum()
total = t.n.sum()
print(f"  mentions: {total}")
print(f"  correct under majority-per-review ceiling: {ceiling_num} = {ceiling_num/total*100:.1f}%")

# how much of that is because most mentions are positive anyway
print(f"  mentioning 'positive' throughout: {t.npos.sum()/total*100:.1f}%")
print(f"  ceiling if every mention predicted positive: {t.npos.sum()/total*100:.1f}%")

t.to_csv("output/audit/within_review_heterogeneity.csv", index=False)
print("\nwritten output/audit/within_review_heterogeneity.csv")
