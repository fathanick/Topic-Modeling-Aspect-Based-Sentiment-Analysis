"""Review-level baseline on the CURRENT pipeline, for the sign-reversal claim.

The manuscript compares sentence-level scores against "review-level
aggregation". That baseline must come from the same model and corpus as the
sentence-level run (Qwen2.5-7B, 1,530 reviews) — not from the old Llama/Groq
output, otherwise the comparison confounds unit of analysis with model change.

`absa_llm_gpu/absa_llm_results.csv` is that baseline: review-level output from
the same GPU pipeline. This script recomputes the per-aspect scores from it and
prints them next to the sentence-level scores.

Run: ./venv/bin/python scripts/review_level_baseline.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "output"

ASPECTS = ["cleanliness", "facilities", "price", "service", "accessibility",
           "scenery", "atmosphere", "food", "safety", "crowd", "photo_spot",
           "historical_value"]

rv = pd.read_csv(O / "absa_llm_gpu/absa_llm_results.csv")
print(f"review-level rows: {len(rv)}")
print("status:", rv.analysis_status.value_counts().to_dict())

rows = []
for a in ASPECTS:
    col = f"{a}_sentiment"
    if col not in rv:
        print("  missing", col)
        continue
    s = rv[col].dropna().astype(str).str.lower()
    n = len(s)
    p = int((s == "positive").sum())
    q = int((s == "negative").sum())
    u = int((s == "neutral").sum())
    rows.append({"aspect": a, "total": n, "pos": p, "neg": q, "neu": u,
                 "score": (p - q) / n if n else 0.0})
rvl = pd.DataFrame(rows).set_index("aspect")
print("\n=== review-level (Qwen2.5-7B, same pipeline) ===")
print(rvl[["total", "pos", "neg", "neu", "score"]].round(3).to_string())

km = pd.read_csv(O / "sentence_absa/kalimat_aspek_sentimen.csv")
agg = km.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
agg["total"] = agg.sum(axis=1)
agg["score"] = (agg.pos - agg.neg) / agg.total
print("\n=== sentence-level ===")
print(agg[["total", "pos", "neg", "neu", "score"]].round(3).to_string())

cmp = pd.DataFrame({
    "review_total": rvl.total, "review_score": rvl.score.round(3),
    "sent_total": agg.total, "sent_score": agg.score.round(3),
})
cmp["delta"] = (cmp.sent_score - cmp.review_score).round(3)
cmp["sign_flip"] = ((cmp.review_score > 0) & (cmp.sent_score < 0))
print("\n=== comparison ===")
print(cmp.to_string())
print(f"\nmentions review-level {int(rvl.total.sum()):,}  "
      f"| per review {rvl.total.sum()/len(rv):.2f}")
print(f"mentions sentence-level {int(agg.total.sum()):,}  "
      f"| per review {agg.total.sum()/1529:.2f}")
print("sign flips:", list(cmp[cmp.sign_flip].index))
print("\noverall review-level polarity:")
oc = rv.overall_sentiment.dropna().astype(str).str.lower().value_counts()
print(oc.to_string())
