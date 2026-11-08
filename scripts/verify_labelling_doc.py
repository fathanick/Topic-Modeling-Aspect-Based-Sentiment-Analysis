"""Verify every number quoted in docs/LABELLING_aspectless_sentences.md.

Run: ./venv/bin/python scripts/verify_labelling_doc.py
"""
import collections
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "sentiment_analysis"))

from audit_unlabelled_sentences import classify  # noqa: E402
from sentence_absa import SYSTEM_PROMPT  # noqa: E402

raw = pd.read_csv(ROOT / "output/sentence_absa/sentence_absa_raw.csv")
al = pd.read_csv(ROOT / "output/audit/aspectless_sentences.csv")

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


mentions = int(raw.n_aspects.sum())
eq("kalimat", len(raw), 6479)
eq("mention", mentions, 8412)

z = raw[raw.n_aspects == 0].copy()
z["cls"] = z.sentence.apply(classify)
per_rev = raw.groupby("review_id").n_aspects.sum()
rev = raw.groupby("review_id").agg(nsent=("sentence", "size"),
                                   tot=("n_aspects", "sum"),
                                   stars=("stars", "first"))
z["rev_total"] = z.review_id.map(per_rev)

eq("kalimat tanpa aspek", len(z), 1309)
in_asp = int((z.rev_total > 0).sum())
solo = int((z.rev_total == 0).sum())
eq("di review beraspek", in_asp, 1248)
eq("di review tanpa aspek", solo, 61)
near("persen di review beraspek", round(in_asp / len(z) * 100, 1), 95.3)
near("persen solo", round(solo / len(z) * 100, 1), 4.7)

zero_rev = rev[rev.tot == 0]
eq("review tanpa aspek", len(zero_rev), 41)
eq("total review", len(rev), 1529)
near("persen review tanpa aspek", round(len(zero_rev) / len(rev) * 100, 1), 2.7)
for s, want in {4: 8, 5: 33}.items():
    eq(f"review nol-aspek {s} bintang", int((zero_rev.stars == s).sum()), want)

# --- D-class star spread ---
d = z[z.cls == "D evaluative cue"]
eq("kelas D", len(d), 109)
for s, want in {1: 4, 2: 2, 3: 2, 4: 21, 5: 80}.items():
    eq(f"D {s} bintang", int((d.stars == s).sum()), want)

# --- prompt schema: sentiment exists only inside aspect objects ---
eq('"sentiment" hanya di objek aspect', '"sentiment"' in SYSTEM_PROMPT, True)
top_level = re.search(r'\{\s*"sentiment"\s*:', SYSTEM_PROMPT) is not None
eq("tidak ada sentiment di level atas", top_level, False)

# --- ceiling numbers quoted ---
div = pd.read_csv(ROOT / "output/divergence/aspect_disagreement.csv")
g = div.groupby("review_id")["llm"]
maj_share = g.agg(lambda s: s.value_counts().max() / len(s))
n_asp = g.size()
ceiling = (maj_share * n_asp).sum() / n_asp.sum() * 100
observed = (~div.disagree).mean() * 100
near("CEILING = 83.6%", round(ceiling, 1), 83.6)
near("observed 77.2%", round(observed, 1), 77.2)
near("gap 6.4 pp", round(ceiling - observed, 1), 6.4)

# --- resolution column ---
eq("baris aspectless_sentences", len(al), 1309)
eq("no_aspect_no_mention", int((al.resolution == "no_aspect_no_mention").sum()), 1248)
eq("review_has_no_aspect_either",
   int((al.resolution == "review_has_no_aspect_either").sum()), 61)
eq("kolom sentiment TIDAK ada", "sentiment" in al.columns, False)
eq("kolom aspect TIDAK ada", "aspect" in al.columns, False)

# --- mixed polarity, the argument at risk ---
multi = raw[raw.n_aspects >= 2]
eq("kalimat >=2 aspek", len(multi), 2144)
opp = 0
for s in multi.aspects:
    if {"pos", "neg"} <= {a["sentiment"] for a in json.loads(s or "[]")}:
        opp += 1
eq("positif DAN negatif", opp, 411)
near("persen mixed", round(opp / len(multi) * 100, 1), 19.2)

# star spread of the mixed-polarity sentences (the argument at risk)
import collections
star_mix = collections.Counter()
for r in multi.itertuples():
    if {"pos", "neg"} <= {a["sentiment"] for a in json.loads(r.aspects or "[]")}:
        star_mix[int(r.stars)] += 1
for s, want in {1: 12, 2: 5, 3: 12, 4: 106, 5: 276}.items():
    eq(f"mixed {s} bintang", star_mix[s], want)
eq("mixed di bintang 4-5", star_mix[4] + star_mix[5], 382)
near("persen mixed di 4-5", round((star_mix[4] + star_mix[5]) / opp * 100, 1), 92.9)

# --- section 3: rescued no-aspect sentiment ---
na = pd.read_csv(ROOT / "output/sentence_absa/no_aspect_sentiment.csv")
eq("no_aspect_sentiment baris", len(na), 104)
for lab, want in {"pos": 60, "neu": 27, "neg": 14, "mixed": 3}.items():
    eq(f"sentimen {lab}", int((na.sentence_sentiment == lab).sum()), want)
eq("reason rejected", int((al.reason == "rejected_aspect_outside_glossary").sum()), 104)
eq("reason empty", int((al.reason == "model_returned_empty_list").sum()), 1205)
eq("reason lengkap", int(al.reason.notna().sum()), 1309)

asp = collections.Counter()
for v in na.model_aspect:
    for a in str(v).split("; "):
        asp[a] += 1
eq("overall 27", asp["overall"], 27)
eq("historical_value 10", asp["historical_value"], 10)
ck_names = ["Parah bgt sih.", "Sepi di hari libur nasional."]
for t in ck_names:
    hit = na[na.sentence.str.strip() == t]
    eq(f"negatif ada: {t[:26]!r}", len(hit) == 1 and hit.iloc[0].sentence_sentiment == "neg", True)
hit = na[na.sentence.str.contains("tidak akan menyarankan sama sekali", na=False)]
eq("negatif 'tidak akan menyarankan'", int((hit.sentence_sentiment == "neg").sum()), 1)
hit = na[na.sentence.str.contains("tidak merekomendasikan", na=False)]
eq("negatif 'tidak merekomendasikan'", int((hit.sentence_sentiment == "neg").sum()), 1)

# the rescued table must not be a mention table
km2 = pd.read_csv(ROOT / "output/sentence_absa/kalimat_aspek_sentimen.csv")
eq("tanpa overlap dgn mention", len(set(na.sentence_id) & set(km2.sentence_id)), 0)
eq("mention tetap 8412", len(km2), 8412)

print("\n" + ("ALL LABELLING-DOC NUMBERS VERIFIED" if not fails else f"FAILED: {fails}"))
raise SystemExit(1 if fails else 0)
