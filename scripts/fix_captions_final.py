"""Final pass: put every figure caption next to its drawing and fix numbering.

After the rebuild the drawings are all present but two captions drifted and the
old "Fig . Top keywords by aspect" caption survived at the end. This walks the
body in document order, and for each drawing assigns a caption number in
sequence, moving the existing caption paragraph directly behind the drawing.

Run after build_manuscript_revision.py:
  ../.venv/bin/python scripts/fix_captions_final.py
"""
import copy
import re
from pathlib import Path

import docx
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT.parent / "Manuscript" / "Manuscript-v3-revised.docx"

W = qn("w:p")
DRAW = qn("w:drawing")

CAPTIONS = [
    "Fig. 1. Research workflow: reviews, sentence splitting, aspect extraction "
    "and sentence-level analysis.",
    "Fig. 2. BERTopic pipeline. Parameters are those of the final model "
    "(min_cluster_size=15, eom).",
    "Fig. 3. Prompt structure for aspect extraction. Every keyword must be "
    "copied verbatim from the sentence.",
    "Fig. 4. Topic composition: 8 topics, 94.5% of reviews assigned, 84 outliers "
    "(5.5%).",
    "Fig. 5. Most representative terms for each of the eight topics.",
    "Fig. 6. Share of each topic within the 16 most-reviewed destinations.",
    "Fig. 7. Aspect mentions by polarity (left) and net sentiment score per "
    "aspect (right).",
]

doc = docx.Document(DOC)
body = doc.element.body

# collect paragraphs in body order
paras = body.findall(f".//{W}")

# a caption paragraph = any paragraph whose text starts with Fig / 'Fig .'
def is_caption(p):
    txt = "".join(t.text or "" for t in p.iter(qn("w:t"))).strip()
    return bool(re.match(r"^Fig\.?\s*\d*\s*\.", txt))


def has_drawing(p):
    return p.find(f".//{DRAW}") is not None


# drop every existing caption paragraph from the tree, remembering one as a
# formatting template
template = None
removed = 0
for p in list(body.iter(W)):
    if is_caption(p):
        if template is None:
            template = copy.deepcopy(p)
        p.getparent().remove(p)
        removed += 1
print(f"removed {removed} caption paragraphs, kept one as template")
assert template is not None

# rebuild captions from the template
def make_caption(text):
    p = copy.deepcopy(template)
    ts = list(p.iter(qn("w:t")))
    if ts:
        ts[0].text = text
        for extra in ts[1:]:
            extra.text = ""
    else:
        r = p.makeelement(qn("w:r"), {})
        t = r.makeelement(qn("w:t"), {})
        t.text = text
        r.append(t)
        p.append(r)
    return p


# walk the body in order; after each drawing (which sits inside a run inside a
# paragraph) attach its caption paragraph
drawings = list(body.iter(DRAW))
print(f"found {len(drawings)} drawings")
assert len(drawings) == len(CAPTIONS), f"{len(drawings)} drawings vs {len(CAPTIONS)} captions"

for i, d in enumerate(drawings):
    # paragraph that owns this drawing
    p = d.getparent()
    while p is not None and p.tag != W:
        p = p.getparent()
    assert p is not None
    cap = make_caption(CAPTIONS[i])
    p.addnext(cap)

doc.save(DOC)
print("captions rebuilt and attached")

# verify
d2 = docx.Document(DOC)
seq = []
for p in d2.paragraphs:
    t = p.text.strip()
    if t.startswith("Fig.") or t.startswith("Fig "):
        seq.append(t[:60])
print(f"\ncaptions in order ({len(seq)}):")
for s in seq:
    print("  ", s)
