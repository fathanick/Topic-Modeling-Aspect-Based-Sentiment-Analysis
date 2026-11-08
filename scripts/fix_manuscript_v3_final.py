"""Fix the stale numbers/claims that survived the DOCX rebuild (v3-revised).

Paragraph-level edits through python-docx runs: a replacement may span several
runs, in which case the first affected run receives the whole new text and the
others are emptied. Runs are used rather than raw XML so Mendeley citation SDT
fields stay untouched; each target is asserted not to straddle one.

Run: ../.venv/bin/python scripts/fix_manuscript_v3_final.py
"""
import re
import shutil
from pathlib import Path

import docx

MAN = Path(__file__).resolve().parents[2] / "Manuscript"
DOC = MAN / "Manuscript-v3-revised.docx"
BAK = MAN / "Manuscript-v3-revised.pre-fix.docx"

REPL = [
    # Review-level scores of accessibility / crowd / safety are +0.204 / +0.141 /
    # +0.088, not +0.059 / +0.043 / +0.101 (review_level_baseline.py).
    ("(+0.059, +0.043 and +0.101)", "(+0.204, +0.141 and +0.088)"),
    # 70.3/13.4 is the review-level split; the sentence-level split is
    # 45.8 pos / 16.4 neg / 37.8 neu over 8,412 mentions (Table 5).
    ("At sentence level the same corpus yields 45.8% positive and 37.8% negative over 8,412 mentions.",
     "At sentence level the same corpus yields 45.8% positive, 16.4% negative and 37.8% neutral over 8,412 mentions."),
    # BERTopic yields four groupings; "five thematic pillars" was the old foil.
    ("The five thematic pillars identified by BERTopic coastal and marine experiences, culinary tourism, "
     "cultural heritage, photography-oriented activities, and urban landmarks accurately reflect",
     "The four thematic groupings identified by BERTopic (coastal and marine experiences, urban and "
     "photography-oriented activities, cultural heritage, and pilgrimage) accurately reflect"),
    # Figures the manuscript never contained.
    ("evidenced by the similarity heatmap and dendrogram,",
     "evidenced by a c_v of 0.5912,"),
    # Facilities scores +0.156 (fourth-highest volume, 900), not +0.817 / second.
    ("Among the six excellent-rated aspects, Facilities (+0.817) is noteworthy for receiving the "
     "second-highest mention volume of any aspect despite recording the lowest score within the "
     "excellent range.",
     "Among the six excellent-rated aspects, Facilities (+0.156) is noteworthy for receiving the "
     "fourth-highest mention volume of any aspect (900) while recording one of the lowest positive scores."),
    # Figure numbering: fig6 = topic x destination, fig7 = aspects.
    ("Figure 7 reports the mention volume of each aspect split by polarity (left) and the resulting "
     "net score (right).",
     "Figure 6 reports the share of each topic within the destinations, and Figure 7 the mention "
     "volume of each aspect split by polarity (left) with the resulting net score (right)."),
    ("Figure 6 shows both levels.", "Figure 7 shows both levels."),
    ("Figure 7 shows how aspect composition varies across the 14 most-reviewed destinations.",
     "Figure 6 shows how topic composition varies across the 16 most-reviewed destinations."),
    # The sentence claimed "temple destinations concentrate on historical value
    # (0.26 of mentions at Prambanan)" — true (0.257) — but the rest of the list
    # was invented. Replaced by measured aspect shares, which is what Table 5
    # and the per-destination table support.
    ("The variation is large enough that a single national or regional average would describe none "
     "of them: temple destinations concentrate on historical value (0.26 of mentions at Prambanan), "
     "coastal destinations on atmosphere and food, and urban destinations on scenery and price.",
     "The variation is large enough that a single national or regional average would describe none "
     "of them: at Candi Prambanan 25.7% of aspect mentions concern historical value, whereas across "
     "the five most-reviewed beaches scenery leads at 21.2%; at Malioboro atmosphere and crowd level "
     "account for 25.4% and 19.0%, and at Tugu Jogja atmosphere and photo spots for 24.7% and 19.2%."),
    # min_cluster_size / min_topic_size are 15 in the final model (manifest).
    ("min_cluster_size=10", "min_cluster_size=15"),
    ("min_topic_size=10", "min_topic_size=15"),
]


def replace_in_paragraph(p, old, new):
    runs = p.runs
    text = "".join(r.text for r in runs)
    if old not in text:
        return 0
    start = text.index(old)
    end = start + len(old)
    # locate run boundaries
    bounds, pos = [], 0
    for r in runs:
        bounds.append((pos, pos + len(r.text)))
        pos += len(r.text)
    touched = [i for i, (a, b) in enumerate(bounds) if a < end and b > start]
    if not touched:
        return 0
    first = touched[0]
    # citation SDTs live between runs; refuse edits that would jump over one
    span_xml = p._p.xml
    if span_xml.count("<w:sdt>") != 0 and len(touched) > 1:
        seg = runs[first]._r
        # conservative: require the whole paragraph's SDT count to be 0
        if "<w:sdt>" in p._p.xml:
            raise SystemExit(f"refusing cross-run edit in paragraph with citation field: {old[:40]}")
    a, b = bounds[first]
    la, lb = bounds[touched[-1]]
    head = runs[first].text[: start - a]
    tail = runs[touched[-1]].text[end - la:] if end > la else ""
    runs[first].text = head + new + tail
    for i in touched[1:]:
        runs[i].text = ""
    return 1


def main():
    if not BAK.exists():
        shutil.copy2(DOC, BAK)
    d = docx.Document(DOC)
    counts = {}
    for old, new in REPL:
        n = sum(replace_in_paragraph(p, old, new) for p in d.paragraphs)
        if n == 0:  # abstract sits inside a table cell
            for t in d.tables:
                for row in t.rows:
                    for c in row.cells:
                        n += sum(replace_in_paragraph(p, old, new) for p in c.paragraphs)
        counts[old[:45]] = n
        if n == 0:
            raise SystemExit(f"NOT APPLIED: {old[:70]!r}")
        if n > 1:
            raise SystemExit(f"AMBIGUOUS ({n} hits): {old[:70]!r}")
    d.save(DOC)
    for k, v in counts.items():
        print(f"  {v}  {k}")
    print("saved", DOC.name)


if __name__ == "__main__":
    main()
