"""Verify every number quoted in docs/AUDIT_alias_and_unlabelled.md.

Run: ./venv/bin/python scripts/verify_audit_doc.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_unlabelled_sentences import classify  # noqa: E402

raw = pd.read_csv(ROOT / "output/sentence_absa/sentence_absa_raw.csv")
sus = pd.read_csv(ROOT / "output/audit/unlabelled_suspects.csv")
fz = pd.read_csv(ROOT / "output/audit/fuzzy_recoveries.csv")
rt = pd.read_csv(Path("/tmp/kerr/retry/sentence_absa_raw.csv"))
loc = pd.read_csv(ROOT / "output/sentence_absa_retry_local/sentence_absa_raw.csv")
oka = pd.read_csv(ROOT / "output/audit/ollama_vs_kaggle.csv")

mentions = int(raw.n_aspects.sum())
labelled = int((raw.n_aspects > 0).sum())
zero = int((raw.n_aspects == 0).sum())

fails = []


def eq(name, got, want):
    ok = got == want
    print(f"{'OK  ' if ok else 'FAIL'} {name}: {got} (doc says {want})")
    if not ok:
        fails.append(name)


def near(name, got, want, tol=0.05):
    ok = abs(got - want) <= tol
    print(f"{'OK  ' if ok else 'FAIL'} {name}: {got} (doc says {want})")
    if not ok:
        fails.append(name)


eq("kalimat dilabeli", len(raw), 6479)
eq("kalimat >=1 aspek", labelled, 5170)
near("persen >=1 aspek", round(labelled / len(raw) * 100, 1), 79.8)
eq("mention total", mentions, 8412)
near("mention/kalimat", round(mentions / len(raw), 2), 1.30)

per_rev = raw.groupby("review_id").n_aspects.sum()
near("mention/review", round(per_rev.mean(), 2), 5.50)
near("rasio vs 3.72", round(per_rev.mean() / 3.72, 2), 1.48)

multi = raw[raw.n_aspects >= 2]
eq("kalimat >=2 aspek", len(multi), 2144)
near("persen >=2 aspek", round(len(multi) / len(raw) * 100, 1), 33.1)

opp = 0
for s in multi.aspects:
    sent = {a["sentiment"] for a in json.loads(s or "[]")}
    if {"pos", "neg"} <= sent:
        opp += 1
eq("positif DAN negatif", opp, 411)
near("persen dr yg >=2 aspek", round(opp / len(multi) * 100, 1), 19.2)

bad = 0
for r in raw.itertuples():
    for a in json.loads(r.aspects or "[]"):
        if a["keywords"] not in r.sentence or a["span"] not in r.sentence:
            bad += 1
eq("grounding violations", bad, 0)

rej = sum(len([p for p in str(e).split("; ") if p])
          for e in raw.error.dropna())
near("mention retention %", round(mentions / (mentions + rej) * 100, 1), 94.7)

# --- opsi 2: komposisi 1.309 kalimat tanpa aspek ---
eq("kalimat tanpa aspek", zero, 1309)
cls = raw[raw.n_aspects == 0].sentence.apply(classify).value_counts()
eq("A terlalu pendek", int(cls["A too short"]), 159)
eq("B pertanyaan", int(cls["B question"]), 16)
eq("C deskriptif", int(cls["C descriptive"]), 841)
eq("D evaluative cue", int(cls["D evaluative cue"]), 109)
eq("E aspect keyword", int(cls["E aspect keyword"]), 184)
eq("jumlah kategori", int(cls.sum()), 1309)
near("A persen", round(cls["A too short"] / zero * 100, 1), 12.1)
near("B persen", round(cls["B question"] / zero * 100, 1), 1.2)
near("C persen", round(cls["C descriptive"] / zero * 100, 1), 64.2)
near("D persen", round(cls["D evaluative cue"] / zero * 100, 1), 8.3)
near("E persen", round(cls["E aspect keyword"] / zero * 100, 1), 14.1)

# --- re-run determinisme: the correction that killed the 238 claim ---
eq("retry rows", len(rt), 238)
eq("retry mentions", int(rt.n_aspects.sum()), 0)
v6 = raw.set_index("sentence_id")
identical = sum(1 for r in rt.itertuples() if str(v6.loc[r.sentence_id, "raw"]).strip()
                == str(r.raw).strip())
eq("retry raw identik dgn v6", identical, 238)
# unlabelled_suspects.csv is a PRE-fuzzy snapshot: it still contains the 45
# mentions that the tolerant matcher later recovered, so its 302 rows and the
# post-recovery 1.309 are measured at different points. Compare like with like:
# D+E on the CURRENT zero set must be 293 (302 minus 9 recovered sentences).
znow = raw[raw.n_aspects == 0].copy()
znow["cls"] = znow.sentence.apply(classify)
de_now = int(znow.cls.isin(["D evaluative cue", "E aspect keyword"]).sum())
eq("D+E pada set nol saat ini", de_now, 293)
eq("D+E pre-fuzzy di suspects file",
   int(sus.status.isin(["ok", "rejected"]).sum()), 302)
eq("kalimat pulih = selisih", 302 - de_now, 9)

# --- fuzzy ---
eq("fuzzy recoveries", len(fz), 45)
eq("fuzzy: span", int((fz.field == "span").sum()), 27)
eq("fuzzy: keywords", int((fz.field == "keywords").sum()), 18)
near("fuzzy sim median", round(fz.sim.median(), 3), 0.963)
eq("fuzzy sim < 0.85", int((fz.sim < 0.85).sum()), 0)

# --- bias ---
raw["nw"] = raw.sentence.str.split().str.len()
near("kata dilabeli", round(raw[raw.n_aspects > 0].nw.mean(), 1), 14.2)
near("kata tanpa aspek", round(raw[raw.n_aspects == 0].nw.mean(), 1), 8.9)
near("rasio kata", round(raw[raw.n_aspects == 0].nw.mean()
                         / raw[raw.n_aspects > 0].nw.mean(), 2), 0.63)
for s, want in {1: 21.6, 2: 14.8, 3: 23.0, 4: 18.7, 5: 20.4}.items():
    sub = raw[raw.stars == s]
    near(f"skip {s} bintang %", round((sub.n_aspects == 0).mean() * 100, 1), want)

# --- ollama, documented as NOT merged ---
eq("ollama compared", len(oka), 40)
eq("ollama identical", int(oka.identical.sum()), 24)
near("ollama identical %", round(oka.identical.mean() * 100, 1), 60.0)
near("ollama mean jaccard", round(oka.jaccard.mean(), 3), 0.738)
eq("ollama mentions (63 sentences)", int((loc.n_aspects > 0).sum()), 63)
eq("ollama total mentions", int(loc.n_aspects.sum()), 76)

print("\n" + ("ALL DOC NUMBERS VERIFIED" if not fails else f"FAILED: {fails}"))
raise SystemExit(1 if fails else 0)
