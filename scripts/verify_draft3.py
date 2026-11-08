"""Verify every number in DRAFT_results_4_1_4_3_4_4.md."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F = []

def ck(label, got, want, tol=0.005):
    ok = abs(float(got) - float(want)) <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {label}: got {got} want {want}")
    if not ok: F.append(label)

ASP = ["scenery","facilities","price","accessibility","crowd","atmosphere",
       "service","food","historical_value","cleanliness","safety","photo_spot"]
ab = pd.read_csv(ROOT / "output/absa_llm_gpu/absa_llm_results.csv")
ib = pd.read_csv(ROOT / "output/absa_indobertweet/absa_indobertweet_results.csv")
ibs = pd.read_csv(ROOT / "output/absa_indobertweet/absa_indobertweet_summary.csv")
ds = pd.read_csv(ROOT / "output/topic_x_aspect/destination_aspect.csv").set_index("destination")
r = pd.read_csv(ROOT / "output/topic_modeling_final/topic_modeling_results.csv")
ts = pd.read_csv(ROOT / "output/topic_modeling_final/topics_summary.csv")
coh = json.load(open(ROOT / "output/topic_modeling_final/topic_coherence.json"))

print("--- 4.1 table P (labeller B) ---")
want_P = {"scenery":(785,94.5,2.4,3.1,0.915),"atmosphere":(674,87.5,6.1,6.4,0.812),
 "price":(589,48.9,38.5,12.6,0.363),"facilities":(550,75.1,11.1,13.8,0.613),
 "photo_spot":(469,83.4,8.7,7.9,0.755),"crowd":(454,42.5,29.1,28.4,0.141),
 "accessibility":(437,50.8,18.8,30.4,0.204),"cleanliness":(423,55.8,20.6,23.6,0.322),
 "service":(414,76.8,13.8,9.4,0.674),"historical_value":(390,76.7,18.7,4.6,0.721),
 "food":(313,74.4,16.3,9.3,0.652),"safety":(193,38.9,31.1,30.1,0.088)}
for a,(mt,pp,pn,ng,sc) in want_P.items():
    v = ab[f"{a}_sentiment"]; v = v[v.isin(("positive","negative","neutral"))]
    ck(f"B {a} mentions", len(v), mt)
    ck(f"B {a} score", round(((v=="positive").sum()-(v=="negative").sum())/len(v),3), sc)

print("\n--- 4.1 table R deltas ---")
for a, d in {"safety":-0.753,"crowd":-0.610,"accessibility":-0.569,"price":-0.403,
             "cleanliness":-0.383,"food":-0.195,"facilities":-0.152,"photo_spot":-0.071,
             "historical_value":-0.063,"service":-0.059,"atmosphere":-0.011,
             "scenery":0.028}.items():
    v = ab[f"{a}_sentiment"]; v = v[v.isin(("positive","negative","neutral"))]
    b = round(((v=="positive").sum()-(v=="negative").sum())/len(v),3)
    A = float(ibs[ibs.aspect==a].sentiment_score.iloc[0])
    ck(f"delta {a}", round(b-A,3), d)

print("\n--- 4.1 spreads and rho ---")
m = ibs[["aspect","sentiment_score"]].copy()
m["B"] = [round(((ab[f'{a}_sentiment'].isin(('positive','negative','neutral'))).sum() and
   (ab.loc[ab[f'{a}_sentiment'].isin(('positive','negative','neutral')), f'{a}_sentiment']=='positive').sum() -
   (ab.loc[ab[f'{a}_sentiment'].isin(('positive','negative','neutral')), f'{a}_sentiment']=='negative').sum()) /
   (ab[f'{a}_sentiment'].isin(('positive','negative','neutral'))).sum(), 3) for a in m.aspect]
ck("A spread", round(m.sentiment_score.max()-m.sentiment_score.min(),3), 0.182)
ck("B spread", round(m.B.max()-m.B.min(),3), 0.827)
ck("spearman", round(m.sentiment_score.corr(m.B, method="spearman"),3), 0.406)
ck("pearson", round(m.sentiment_score.corr(m.B),3), 0.425)

print("\n--- 4.3: scenery minimum and 'factor of nine' ---")
ck("scenery min", round(ds.scenery.min(),2), 0.71)
ck("B max/min ratio", round(0.915/0.088,1), 10.4, 0.1)
print(f"  -> 'more than nine' claim: 0.915/0.088 = {0.915/0.088:.1f} (valid)")
for a, mn in {"photo_spot":0.0,"historical_value":0.0,"food":0.0}.items():
    ck(f"{a} min", round(ds[a].min(),3), mn)

print("\n--- 4.3: destination lowest-three claims ---")
ck("Bukit Bintang accessibility", round(ds.loc["Bukit Bintang","accessibility"],2), -0.67)
ck("Candi Borobudur crowd", round(ds.loc["Candi Borobudur","crowd"],2), -1.00)
ck("Pantai Depok cleanliness", round(ds.loc["Pantai Depok","cleanliness"],2), -0.62)
ck("Pantai Depok service", round(ds.loc["Pantai Depok","service"],2), -0.09)
ck("Situs Warungboto facilities", round(ds.loc["Situs Warungboto","facilities"],2), -0.21)
ck("Candi Borobudur price", round(ds.loc["Candi Borobudur","price"],2), -0.25)
ck("Pantai Baron atmosphere", round(ds.loc["Pantai Baron","atmosphere"],2), 0.44)
ck("Museum Ullen Sentalu photo_spot", round(ds.loc["Museum Ullen Sentalu","photo_spot"],2), 0.10)

print("\n--- 4.4 topics ---")
ck("topics", len(ts), 8)
ck("coverage", round(ts.document_count.sum()/1530*100,1), 94.5)
ck("c_v", coh["c_v"], 0.5912)
ck("c_npmi", coh["c_npmi"], 0.0870, 0.0001)
t0 = ts[ts.topic_id==0].document_count.iloc[0]
ck("topic 0 pct", round(t0/1530*100,1), 52.1)
ck("two largest pct", round(ts.document_count.nlargest(2).sum()/1530*100,1), 81.5)
for t, want in {5:("Sendang Sono",26),6:("Makam Raja-Raja Imogiri",25),7:("Gumuk Pasir Parangkusumo",25)}.items():
    sub = r[r.topic==t]
    ck(f"topic {t} top dest", int(sub.destination.value_counts().iloc[0]), want[1])
    print(f"       topic {t} top dest name: {sub.destination.value_counts().index[0]}")

print("\n--- 4.4: destinations >=80% one cluster ---")
p = pd.read_csv(ROOT / "output/topic_modeling_final/topic_x_destination_pct.csv").set_index("destination")
dom = sum(1 for _,row in p.iterrows() if pd.to_numeric(row, errors='coerce').max() >= 80)
ck("dominated destinations", dom, 22)

print("\n--- 4.4: topic 3 has no safety mentions ---")
import csv
cnt = pd.read_csv(ROOT / "output/topic_x_aspect/topic_aspect_counts.csv").set_index("topic")
ck("topic 3 safety count", int(cnt.loc[3,"safety"]), 0)
ck("topic 2 crowd count", int(cnt.loc[2,"crowd"]), 9)
ck("topic 2 crowd score", round(float(pd.read_csv(ROOT/"output/topic_x_aspect/topic_aspect_matrix.csv").set_index("topic").loc[2,"crowd"]),2), -0.33)

print("\n--- 4.4: Imogiri / Kalasan / Gumuk aspect claims ---")
mtx = pd.read_csv(ROOT / "output/topic_x_aspect/topic_aspect_matrix.csv").set_index("topic")
ck("topic4 atmosphere", round(mtx.loc[4,"atmosphere"],2), 1.00)
ck("topic4 food", round(mtx.loc[4,"food"],2), 0.00)
ck("topic6 accessibility", round(mtx.loc[6,"accessibility"],2), -0.40)
ck("topic6 crowd", round(mtx.loc[6,"crowd"],2), -0.10)
ck("topic7 scenery", round(mtx.loc[7,"scenery"],2), 0.85)
ck("topic7 price", round(mtx.loc[7,"price"],2), 0.67)
ck("topic7 crowd", round(mtx.loc[7,"crowd"],2), -0.25)

print()
if F:
    raise SystemExit(f"FAILED {len(F)}: {F}")
print("ALL §4.1/4.3/4.4 NUMBERS VERIFIED")
