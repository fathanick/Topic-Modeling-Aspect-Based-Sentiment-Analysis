"""Full results tables needed by §4.1, §4.3, §4.4 — every number in one place."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASP = ["scenery","facilities","price","accessibility","crowd","atmosphere",
       "service","food","historical_value","cleanliness","safety","photo_spot"]

ab = pd.read_csv(ROOT / "output/absa_llm_gpu/absa_llm_results.csv")
ib = pd.read_csv(ROOT / "output/absa_indobertweet/absa_indobertweet_results.csv")
ibs = pd.read_csv(ROOT / "output/absa_indobertweet/absa_indobertweet_summary.csv")
ds = pd.read_csv(ROOT / "output/topic_x_aspect/destination_aspect.csv")
dc = pd.read_csv(ROOT / "output/topic_x_aspect/topic_aspect_counts.csv")
dm = pd.read_csv(ROOT / "output/topic_x_aspect/topic_aspect_matrix.csv")
ts = pd.read_csv(ROOT / "output/topic_modeling_final/topics_summary.csv")
mani = json.load(open(ROOT / "output/topic_modeling_final/model_manifest.json"))
coh = json.load(open(ROOT / "output/topic_modeling_final/topic_coherence.json"))

print("========== 4.1: LABELLER B — ASPECT SUMMARY ==========")
rows = []
for a in ASP:
    s = ab[f"{a}_sentiment"]
    v = s[s.isin(("positive","negative","neutral"))]
    p, n, u = (v=="positive").sum(), (v=="negative").sum(), (v=="neutral").sum()
    rows.append({"aspect": a, "mentions": len(v), "pos%": round(p/len(v)*100,1),
                 "neu%": round(u/len(v)*100,1), "neg%": round(n/len(v)*100,1),
                 "score": round((p-n)/len(v),3)})
print(pd.DataFrame(rows).sort_values("mentions", ascending=False).to_string(index=False))

print("\n========== 4.1: LABELLER A — ASPECT SUMMARY ==========")
print(ibs[["aspect","total_mentions","positive_pct","negative_pct","sentiment_score"]].to_string(index=False))

print("\n========== 4.1: SIDE BY SIDE ==========")
m = ibs[["aspect","total_mentions","sentiment_score"]].merge(
    pd.DataFrame(rows)[["aspect","mentions","score"]], on="aspect")
m.columns = ["aspect","A_mentions","A_score","B_mentions","B_score"]
m["delta"] = (m.B_score - m.A_score).round(3)
print(m.sort_values("delta").to_string(index=False))
print(f"\nA spread: {m.A_score.max()-m.A_score.min():.3f} | B spread: {m.B_score.max()-m.B_score.min():.3f}")
print(f"Spearman rho: {m.A_score.corr(m.B_score, method='spearman'):.3f}")

print("\n========== 4.3: DESTINATION x ASPECT ==========")
print("columns:", list(ds.columns))
print(ds.head(3).to_string(index=False))

print("\n========== 4.4: TOPIC x ASPECT ==========")
print("counts shape:", dc.shape, "| matrix shape:", dm.shape)
print(dm.round(2).to_string(index=False))
print("\ncounts:")
print(dc.to_string(index=False))

print("\n========== 4.4: TOPICS ==========")
print(ts.to_string(index=False))
print(f"\nc_v={coh['c_v']} c_npmi={coh['c_npmi']} topics={mani['n_topics']} "
      f"outlier={mani['outlier_pct']}% coverage={mani['coverage_pct']}%")
