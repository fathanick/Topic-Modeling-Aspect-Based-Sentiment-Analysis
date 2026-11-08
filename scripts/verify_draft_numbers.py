"""Verify every number cited in docs/DRAFT_abstract_intro_divergence.md."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "output/divergence/aspect_disagreement.csv")
ab = pd.read_csv(ROOT / "output/absa_llm_gpu/absa_llm_results.csv")
ib = pd.read_csv(ROOT / "output/absa_indobertweet/absa_indobertweet_results.csv")
F = []

def ck(label, got, want, tol=0.15):
    ok = abs(got - want) <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {label}: got {got} want {want}")
    if not ok: F.append(label)

print("--- aggregate ---")
ck("matched labels", len(d), 3638, 0)
ck("disagree pct", round(d.disagree.mean()*100, 1), 22.8, 0.05)
ck("llm-neg ind-pos pct", round(d.llm_negative_ind_positive.mean()*100, 1), 8.9, 0.05)
ck("llm-neg ind-pos count", int(d.llm_negative_ind_positive.sum()), 324, 0)

print("\n--- label distributions ---")
ASP = ["scenery","facilities","price","accessibility","crowd","atmosphere",
       "service","food","historical_value","cleanliness","safety","photo_spot"]
LBL = ("positive","negative","neutral")
alc = pd.concat([ab[f"{a}_sentiment"] for a in ASP])
ibc = pd.concat([ib[f"{a}_sentiment"] for a in ASP])
ck("llm total mentions", int(alc.isin(LBL).sum()), 5691, 0)
ck("llm positive pct", round((alc=="positive").sum()/alc.isin(LBL).sum()*100,1), 70.3, 0.05)
ck("llm neutral pct", round((alc=="neutral").sum()/alc.isin(LBL).sum()*100,1), 16.4, 0.05)
ck("llm negative pct", round((alc=="negative").sum()/alc.isin(LBL).sum()*100,1), 13.4, 0.05)
ck("ib total mentions", int(ibc.isin(LBL).sum()), 5413, 0)
ck("ib positive pct", round((ibc=="positive").sum()/ibc.isin(LBL).sum()*100,1), 88.2, 0.05)
ck("ib negative pct", round((ibc=="negative").sum()/ibc.isin(LBL).sum()*100,1), 7.8, 0.05)
ck("ib neutral pct", round((ibc=="neutral").sum()/ibc.isin(LBL).sum()*100,1), 4.1, 0.05)
ck("silence pct", round((1-3638/5691)*100,1), 36.1, 0.05)

print("\n--- per-aspect disagreement ---")
want = {"accessibility":42.1,"crowd":41.8,"price":41.3,"safety":39.4,"facilities":19.6,
        "food":19.4,"cleanliness":17.8,"historical_value":16.4,"photo_spot":15.1,
        "service":14.6,"atmosphere":11.1,"scenery":6.6}
for a,w in want.items():
    g = round(d[d.aspect==a].disagree.mean()*100,1)
    ck(f"disagree {a}", g, w, 0.05)

print("\n--- per-star disagreement ---")
want_s = {1:30.2, 2:50.0, 3:49.5, 4:36.4, 5:18.7}
for s,w in want_s.items():
    g = round(d[d.stars==s].disagree.mean()*100,1)
    ck(f"disagree stars={s}", g, w, 0.05)

print("\n--- evidence characteristics ---")
ag = d[~d.disagree]; di = d[d.disagree]
ck("agree n", len(ag), 2808, 0)
ck("disagree n", len(di), 830, 0)
ck("agree mean len", round(ag.evidence_len.mean(),1), 40.0, 0.15)
ck("disagree mean len", round(di.evidence_len.mean(),1), 47.6, 0.15)

print("\n--- five star share ---")
ck("five-star pct", round((ab.stars==5).mean()*100,1), 78.6, 0.15)
ck("mean stars", round(ab.stars.mean(),2), 4.7, 0.05)
ck("destinations", ab.destination.nunique(), 34, 0)
ck("reviews", len(ab), 1530, 0)

print()
if F:
    raise SystemExit(f"FAILED {len(F)}: {F}")
print("ALL DRAFT NUMBERS VERIFIED")
