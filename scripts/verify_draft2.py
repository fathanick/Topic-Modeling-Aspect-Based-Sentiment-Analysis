"""Verify numbers in DRAFT_lit_method_discussion.md that were not in the first draft."""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F = []

def ck(label, got, want, tol=0.05):
    ok = abs(float(got) - float(want)) <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {label}: got {got} want {want}")
    if not ok: F.append(label)

print("--- coverage / mentions ---")
ab = pd.read_csv(ROOT / "output/absa_llm_gpu/absa_llm_results.csv")
d = pd.read_csv(ROOT / "output/divergence/aspect_disagreement.csv")
ck("matched pct of 5691", round(len(d)/5691*100, 1), 63.9, 0.05)
ck("mentions per review", round(5691/1530, 2), 3.72, 0.01)
ck("mean stars", round(ab.stars.mean(), 2), 4.68, 0.01)

print("\n--- evidence grounding gap ---")
ASP = ["scenery","facilities","price","accessibility","crowd","atmosphere",
       "service","food","historical_value","cleanliness","safety","photo_spot"]
tot = noev = 0
for a in ASP:
    sc, ec = f"{a}_sentiment", f"{a}_evidence"
    m = ab[sc].isin(("positive","negative","neutral"))
    tot += int(m.sum())
    ev = ab.loc[m, ec]
    noev += int(ev.isna().sum())
print(f"  mentions {tot}, without evidence {noev} ({noev/tot*100:.1f}%)")
ck("mentions without evidence pct", round(noev/tot*100, 1), 18.1, 0.05)

print("\n--- topic clusters ---")
t = pd.read_csv(ROOT / "output/topic_modeling_final/topics_summary.csv")
top2 = t.document_count.nlargest(2).sum() / 1530 * 100
ck("two largest clusters pct of corpus", round(top2, 1), 81.5, 0.05)
print(f"  cluster shares: {list((t.document_count/1530*100).round(1))}")

p = pd.read_csv(ROOT / "output/topic_modeling_final/topic_x_destination_pct.csv").set_index("destination")
dom = sum(1 for _, r in p.iterrows() if pd.to_numeric(r, errors="coerce").max() >= 80)
ck("destinations >=80% one cluster", dom, 22, 0)

print("\n--- beaches in corpus ---")
beach = [x for x in ab.destination.unique() if x.lower().startswith("pantai")]
print(f"  beach destinations: {len(beach)} of {ab.destination.nunique()}")
ck("beach destinations", len(beach), 18, 0)

print("\n--- per-destination review counts ---")
vc = ab.destination.value_counts()
ck("destinations with 50", int((vc==50).sum()), 29, 0)
ck("destinations with 40", int((vc==40).sum()), 1, 0)
ck("destinations with 10", int((vc==10).sum()), 4, 0)

print("\n--- stopwords ---")
src = (ROOT / "src/preprocessing/preprocessing.py").read_text()
m = re.search(r"INDONESIAN_STOP_WORDS\s*=\s*\{(.*?)\n\}", src, re.S)
if m:
    words = re.findall(r"'([^']+)'", m.group(1))
    ck("unique Indonesian stopwords", len(set(words)), 255, 3)

print()
if F:
    raise SystemExit(f"FAILED {len(F)}: {F}")
print("ALL NEW DRAFT NUMBERS VERIFIED")
