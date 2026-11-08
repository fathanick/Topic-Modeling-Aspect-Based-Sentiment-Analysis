"""Dump the revised manuscript as a structure-tagged text file for LaTeX porting.

Emits a line per block:  <KIND>|<STYLE>|<text>
  H1/H2/H3  heading by style name
  FIG       paragraph that contains an inline drawing
  CAP       caption paragraph
  TBL       table marker (followed by TSV rows until ENDTBL)
  REF       bibliography entry (from inside the w:sdt content control)
  P         body paragraph
  ABS/META  abstract, keywords, title
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import docx
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT.parent / "Manuscript" / "Manuscript-v3-revised.docx"
OUT = Path("/tmp/ms_blocks.txt")

doc = docx.Document(str(DOC))
body = doc.element.body
W_P = qn("w:p")
W_T = qn("w:t")
DRAW = qn("w:drawing")

lines = []


def ptext(p_el):
    return "".join(t.text or "" for t in p_el.iter(W_T))


# ---- bibliography lives in the last w:sdt; pull it out first
bib_entries = []
for sdt in body.iter(qn("w:sdt")):
    txt = "".join(t.text or "" for t in sdt.iter(W_T))
    if len(txt) > 1000 and txt.lstrip().startswith("[1]"):
        # entries are separated by a paragraph break; iterate w:p inside
        for p in sdt.iter(W_P):
            t = ptext(p).strip()
            if re.match(r"^\[\d+\]", t):
                bib_entries.append(t)
        break
print(f"bib entries: {len(bib_entries)}", file=sys.stderr)

# ---- walk top-level body children in order
def walk(el):
    for child in el:
        if child.tag == W_P:
            yield child
        elif child.tag == qn("w:tbl"):
            yield child


for child in walk(body):
    if child.tag == qn("w:tbl"):
        rows = []
        for tr in child.iter(qn("w:tr")):
            cells = []
            for tc in tr.iter(qn("w:tc")):
                cells.append(ptext(tc).strip().replace("\n", " "))
            rows.append(cells)
        lines.append(("TBL", "", ""))
        for r in rows:
            lines.append(("ROW", "", "\t".join(r)))
        lines.append(("ENDTBL", "", ""))
        continue

    p = Paragraph(child, doc)
    style = p.style.name if p.style is not None else ""
    txt = p.text.strip()
    has_draw = child.find(f".//{DRAW}") is not None
    if has_draw:
        lines.append(("FIG", style, txt))
        continue
    if not txt:
        continue

    sl = style.lower()
    if sl.startswith("heading 1") or sl == "title":
        kind = "H1"
    elif sl.startswith("heading 2"):
        kind = "H2"
    elif sl.startswith("heading 3"):
        kind = "H3"
    elif sl == "caption":
        kind = "CAP"
    elif txt.upper() == txt and len(txt) < 80 and re.match(r"^[A-Z0-9 ,&/.\-]+$", txt):
        kind = "H1"
    elif re.match(r"^\[\d+\]", txt):
        kind = "REF"
    else:
        kind = "P"
    lines.append((kind, style, txt))

with OUT.open("w") as f:
    for k, s, t in lines:
        f.write(f"{k}|{s}|{t}\n")
    f.write("=== BIBLIOGRAPHY ===\n")
    for b in bib_entries:
        f.write(f"BIB|{b}\n")

print(f"wrote {OUT} ({len(lines)} blocks, {len(bib_entries)} bib entries)", file=sys.stderr)
print(json.dumps({"blocks": len(lines), "bib": len(bib_entries)}))
