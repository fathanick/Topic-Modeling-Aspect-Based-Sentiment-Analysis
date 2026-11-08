"""Second pass: cross-reference and claim fixes the first pass missed."""
import docx
from pathlib import Path

MAN = Path(__file__).resolve().parents[2] / "Manuscript"
DOC = MAN / "Manuscript-v3-revised.docx"

EDITS = [
    # 4.4 never existed; the topic/aspect comparison is 4.1 vs 4.2.
    (102, "Topic coverage of 94.5% means the comparison between topics and aspects in Section 4.4 "
          "rests on almost the whole corpus.",
          "Topic coverage of 94.5% means the topic and aspect views described in Sections 4.1 and 4.2 "
          "rest on almost the whole corpus."),
    # manifest: selected_by = "manual choice for corpus coverage, not the sweep argmax";
    # the argmax (min_cluster_size=25, leaf, c_v 0.6161) left 57.2% outliers.
    (102, "which was optimised over ten candidate configurations using coherence rather than left at its default",
          "which was tuned over ten candidate configurations and set to the coverage-oriented configuration "
          "reported here rather than to the coherence maximum, which left 57.2% of reviews unassigned"),
    (105, "TABLE ", "Table "),
]


def replace_in_paragraph(p, old, new):
    runs = p.runs
    # preferred path: the whole match sits inside a single run, so the SDT
    # fields elsewhere in the paragraph are never touched.
    for r in runs:
        if old in r.text:
            r.text = r.text.replace(old, new, 1)
            return 1
    text = "".join(r.text for r in runs)
    if old not in text:
        return 0
    if "<w:sdt>" in p._p.xml:
        raise SystemExit(f"match straddles a citation field: {old[:60]}")
    start, end = text.index(old), text.index(old) + len(old)
    bounds, pos = [], 0
    for r in runs:
        bounds.append((pos, pos + len(r.text)))
        pos += len(r.text)
    touched = [i for i, (a, b) in enumerate(bounds) if a < end and b > start]
    first, last = touched[0], touched[-1]
    head = runs[first].text[: start - bounds[first][0]]
    tail = runs[last].text[end - bounds[last][0]:]
    runs[first].text = head + new + tail
    for i in touched[1:]:
        runs[i].text = ""
    return 1


d = docx.Document(str(DOC))
for idx, old, new in EDITS:
    n = replace_in_paragraph(d.paragraphs[idx], old, new)
    print(f"  P{idx}: {n}")
    assert n == 1, (idx, old[:50])
d.save(str(DOC))
d2 = docx.Document(str(DOC))
print(d2.paragraphs[102].text[-450:])
print()
print(d2.paragraphs[105].text[:170])
