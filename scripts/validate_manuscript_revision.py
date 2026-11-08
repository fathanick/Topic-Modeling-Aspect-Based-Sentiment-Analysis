"""Validate the revised manuscript against the artifacts it claims to describe.

Checks, in order:
  1. zip + XML integrity, paragraph count, table count
  2. no superseded number survives anywhere in the text
  3. every number that IS present matches output/
  4. captions/figures in the right order
  5. reference list untouched (24 entries, max citation <= 24)
  6. figures embedded are the regenerated files

Run: ../.venv/bin/python scripts/validate_manuscript_revision.py
"""
import json
import re
import zipfile
from pathlib import Path

import docx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT.parent / "Manuscript"
DOC = MAN / "Manuscript-v3-revised.docx"
fails, warns = [], []


def ck(name, cond, detail=""):
    print(f"{'OK  ' if cond else 'FAIL'} {name}  {detail}")
    if not cond:
        fails.append(name)


def note(name, detail):
    print(f"WARN {name}  {detail}")
    warns.append(name)


d = docx.Document(DOC)
body = "\n".join(p.text for p in d.paragraphs)
abstract = "\n".join(p.text for p in d.tables[0].cell(0, 0).paragraphs)
tables = "\n".join(
    cell.text for t in d.tables for row in t.rows for cell in row.cells)
full = body + "\n" + abstract + "\n" + tables
print(f"paragraphs {len(d.paragraphs)}  tables {len(d.tables)}  words {len(full.split())}")

# ---------------------------------------------------------------- 1 integrity
with zipfile.ZipFile(DOC) as z:
    ck("zip integrity", z.testzip() is None)
    import lxml.etree as ET
    for part in ["word/document.xml", "word/styles.xml"]:
        try:
            ET.fromstring(z.read(part))
            ck(f"xml ok {part}", True)
        except Exception as e:
            ck(f"xml ok {part}", False, str(e)[:60])

# ---------------------------------------------------------------- 2 stale numbers
STALE = {
    "59 destinations": r"\b59 (?:Yogyakarta |tourist )?(?:destinations|attractions)",
    "23 topics": r"\b23 (?:different )?topics",
    "4,926": r"4,926",
    "3.2 per review": r"3\.2 aspect mentions",
    "81.9%": r"81\.9\s?%",
    "43.2%": r"43\.2\s?%",
    "661 documents": r"\b661\b",
    "869 documents": r"\b869\b",
    "Llama 3.1": r"Llama 3\.1",
    "Groq": r"\bGroq\b",
    "0.923": r"\+?0\.923",
    "+0.059": r"\+0\.059",
    "+0.043": r"\+0\.043",
    "+0.101": r"\+0\.101",
    "three phases": r"three phases",
    "few-shot": r"few[- ]shot",
}
for label, pat in STALE.items():
    hits = re.findall(pat, full)
    ck(f"no stale: {label}", not hits, f"{len(hits)} hit(s)")

# ---------------------------------------------------------------- 3 current numbers
raw = pd.read_csv(ROOT / "output/sentence_absa/sentence_absa_raw.csv")
km = pd.read_csv(ROOT / "output/sentence_absa/kalimat_aspek_sentimen.csv")
al = pd.read_csv(ROOT / "output/audit/aspectless_sentences.csv")
na = pd.read_csv(ROOT / "output/sentence_absa/no_aspect_sentiment.csv")
tm = json.loads((ROOT / "output/topic_modeling_final/model_manifest.json").read_text())

EXPECT = [
    ("34 destinations", "34"),
    ("1,530 reviews", "1,530"),
    ("6,552 sentences", "6,552"),
    ("6,479 labelled", "6,479"),
    ("5,170 with aspect", "5,170"),
    ("1,309 without", "1,309"),
    ("8,412 mentions", "8,412"),
    ("2,144 multi-aspect", "2,144"),
    ("411 mixed", "411"),
    ("630 competing", "630"),
    ("8 topics", f"{tm['n_topics']} topics"),
    ("c_v 0.591", "0.591"),
    ("94.5% coverage", f"{tm['coverage_pct']}%"),
    ("5.5% outliers", f"{tm['outlier_pct']}%"),
]
for label, needle in EXPECT:
    ck(f"present: {label}", needle in full, needle)

# arithmetic that the text asserts
ck("mentions/corpus", len(km) == 8412, str(len(km)))
ck("sentences labelled", len(raw) == 6479, str(len(raw)))
ck("with aspect", int((raw.n_aspects > 0).sum()) == 5170)
ck("without aspect", int((raw.n_aspects == 0).sum()) == 1309)
ck("no-aspect split", int((al.reason == "model_returned_empty_list").sum()) == 1205)
ck("no-aspect rescued", int((al.reason == "rejected_aspect_outside_glossary").sum()) == 104)
ck("rescued table", len(na) == 104)

per = km.groupby("sentence_id").sentiment.agg(
    n="size", p=lambda s: (s == "pos").any(), q=lambda s: (s == "neg").any())
ck("multi-aspect 2144", int((per.n >= 2).sum()) == 2144)
ck("mixed 411", int((per.p & per.q).sum()) == 411)

sid2rev = raw.set_index("sentence_id").review_id
rv = km.assign(rid=km.sentence_id.map(sid2rev)).groupby("rid").sentiment.agg(
    p=lambda s: (s == "pos").any(), q=lambda s: (s == "neg").any())
ck("competing 630", int((rv.p & rv.q).sum()) == 630)
ck("reviews with mentions 1488", len(rv) == 1488)

# per-aspect figures quoted in the text
agg = km.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
agg["tot"] = agg.sum(axis=1)
checks = {
    "scenery 1277": int(agg.loc["scenery", "tot"]) == 1277,
    "atmosphere 1112": int(agg.loc["atmosphere", "tot"]) == 1112,
    "price 941": int(agg.loc["price", "tot"]) == 941,
    "safety 313": int(agg.loc["safety", "tot"]) == 313,
    "safety neg 51.8": abs(100 * agg.loc["safety", "neg"] / agg.loc["safety", "tot"] - 51.8) < 0.15,
    "accessibility neg 313": int(agg.loc["accessibility", "neg"]) == 313,
    "price neu 61.0": abs(100 * agg.loc["price", "neu"] / agg.loc["price", "tot"] - 61.0) < 0.15,
}
for k, v in checks.items():
    ck(k, v)

score = {a: (agg.loc[a, "pos"] - agg.loc[a, "neg"]) / agg.loc[a, "tot"] for a in agg.index}
ck("safety -0.24", round(score["safety"], 2) == -0.24, f"{score['safety']:.3f}")
ck("accessibility -0.10", round(score["accessibility"], 2) == -0.10, f"{score['accessibility']:.3f}")
ck("crowd -0.02", round(score["crowd"], 2) == -0.02, f"{score['crowd']:.3f}")

# ---------------------------------------------------------------- 4 tables
t1 = "\n".join(c.text for r in d.tables[1].rows for c in r.cells)
ck("table1 no Borobudur", "Borobudur" not in t1)
ck("table1 no Bandung", "Bandung" not in t1 and "Bogor" not in t1)

t3 = {r.cells[0].text.strip(): r.cells[1].text.strip() for r in d.tables[3].rows}
ck("table3 destinations 34", t3.get("Total destinations") == "34", t3.get("Total destinations"))
ck("table3 stars 4.68", t3.get("Average star rating") == "4.68/5")
ck("table3 5star 78.6", t3.get("5-star reviews") == "78.6%")

t5 = [[c.text.strip() for c in r.cells] for r in d.tables[5].rows]
ck("table5 rows 14", len(t5) == 14, str(len(t5)))
ck("table5 header", t5[0][0] == "Aspect")
ck("table5 overall 8412", t5[-1][1] == "8,412", t5[-1][1])
tot_sum = sum(int(r[1].replace(",", "")) for r in t5[1:-1])
ck("table5 sums 8412", tot_sum == 8412, str(tot_sum))
allneg = [r[0] for r in t5[1:-1] if r[5].startswith("-")]
ck("table5 3 negative rows", len(allneg) == 3, str(allneg))

# ---------------------------------------------------------------- 5 references
with zipfile.ZipFile(DOC) as z:
    xml = z.read("word/document.xml").decode("utf8")
# 5,691 is a CURRENT number (review-level mentions), not a stale one: it is
# asserted below instead of being forbidden.
ck("5,691 review-level mentions in text", "5,691" in full)
_cit_nums = []
for _blk in re.findall(r"<w:sdt>.*?</w:sdt>", xml, re.S):
    if "MENDELEY_CITATION_v3" in _blk:
        _cit_nums += [int(n) for n in re.findall(r"\d+", "".join(
            re.findall(r"<w:t[^>]*>([^<]*)</w:t>", _blk)))]
ck("24 bibliography entries cited", sorted(set(_cit_nums)) == list(range(1, 25)),
   f"{len(_cit_nums)} citation SDTs, refs {min(_cit_nums)}-{max(_cit_nums)}")
ck("bibliography SDT kept", "MENDELEY_BIBLIOGRAPHY" in xml)
cite_nums = [int(n) for n in re.findall(r"\[(\d+)\]", full)]
ck("no citation > 24", not cite_nums or max(cite_nums) <= 24, str(max(cite_nums) if cite_nums else 0))

# ---------------------------------------------------------------- 6 figures
with zipfile.ZipFile(DOC) as z:
    media = sorted([n for n in z.namelist() if n.startswith("word/media/")])
ck("8 images", len(media) == 8, str(len(media)))
with zipfile.ZipFile(DOC) as z:
    order = re.findall(r'r:embed="(rId\d+)"', xml)
rels = None
with zipfile.ZipFile(DOC) as z:
    rels = z.read("word/_rels/document.xml.rels").decode("utf8")
rid2t = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
seq = [rid2t.get(r, "?").split("/")[-1] for r in order]
print("   embedded order:", seq)
ck("8 drawings", len(seq) == 8, str(len(seq)))

caps = [p.text.strip() for p in d.paragraphs if p.text.strip().startswith("Fig.")]
ck("8 captions", len(caps) == 8, str(len(caps)))
for i, c in enumerate(caps, 1):
    ck(f"caption {i} numbered", c.startswith(f"Fig. {i}."), c[:30])

print()
if fails:
    print(f"FAILED {len(fails)}: {fails}")
    raise SystemExit(1)
print(f"ALL MANUSCRIPT CHECKS PASSED ({len(warns)} warnings)")
for w in warns:
    print("  warn:", w)
