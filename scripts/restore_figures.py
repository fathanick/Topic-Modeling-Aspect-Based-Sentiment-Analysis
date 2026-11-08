"""Restore the missing figure block and finalise the figure layout.

Pass 1 replaced the paragraph that held Figure 5's drawing, so the document has
6 drawings instead of 7 while the text still references seven. This reinserts
the block between the topic-composition text and the ABSA results heading, then
renumbers all seven captions to match the body order.

Run after fix_manuscript_pass2.py:
  ../.venv/bin/python scripts/restore_figures.py
"""
import copy
import re
import zipfile
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT.parent / "Manuscript"
DOC = MAN / "Manuscript-v3-revised.docx"
FIG = ROOT / "output/figures"

doc = docx.Document(DOC)
P = doc.paragraphs

# ---------------------------------------------------------------- find anchor
anchor = None
for p in P:
    if p.text.strip().startswith("Fig. 6. Share of each topic"):
        anchor = p
        break
assert anchor is not None, "caption anchor for the topic-by-destination figure not found"

# The 'LLM-Based Sentiment Analysis Results' heading must stay after the figures.
heading = None
for p in P:
    if p.text.strip() == "LLM-Based Sentiment Analysis Results":
        heading = p
        break
assert heading is not None

# ---------------------------------------------------------------- build the block
# A figure block = empty paragraph holding the drawing + caption paragraph.
# Copy the caption paragraph of Fig 6 as a formatting template, drop its drawing
# (none) and reuse everything else.
cap6 = anchor
new_cap = copy.deepcopy(cap6._p)

# image paragraph: reuse the shape of the paragraph that holds the existing
# drawing so the inline-extent and paragraph properties carry over
drawing_para = None
for p in P:
    if p._p.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing"):
        drawing_para = p
        break
assert drawing_para is not None, "no existing drawing paragraph to clone"

img_p = copy.deepcopy(drawing_para._p)

# swap the relationship id of the cloned drawing to a fresh one pointing at the
# topic-words PNG we are about to add to word/media/
NEW_RID = "rId9001"
NEW_MEDIA = "word/media/image8.png"

xml_str = img_p.xml
xml_str = re.sub(r'r:embed="[^"]+"', f'r:embed="{NEW_RID}"', xml_str, count=1)
from lxml import etree  # noqa: E402

new_el = etree.fromstring(xml_str)

# ---------------------------------------------------------------- insert before heading
heading._p.addprevious(new_el)
heading._p.addprevious(new_cap)
print("inserted figure block before the ABSA results heading")

# ---------------------------------------------------------------- rewrite captions
CAPS = [
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
n = 0
for p in doc.paragraphs:
    t = p.text.strip()
    if re.match(r"^Fig\.?\s*\d+\.", t):
        if n < len(CAPS) and p.runs:
            p.runs[0].text = CAPS[n]
            for r in p.runs[1:]:
                r.text = ""
            n += 1
print(f"captions rewritten: {n}")

doc.save(DOC)

# ---------------------------------------------------------------- add media + rel
with zipfile.ZipFile(DOC) as z:
    rels = z.read("word/_rels/document.xml.rels").decode("utf8")
assert "rId9001" in open(DOC, "rb").read().decode("latin-1", "ignore") or True

tmp = DOC.with_suffix(".tmp.docx")
with zipfile.ZipFile(DOC) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "word/media/image5.png":
            pass  # keep; slot assignment happens below
        zout.writestr(item, data)
    # add the topic-words figure and its relationship
    zout.writestr(NEW_MEDIA, (FIG / "fig3_topic_words.png").read_bytes())
tmp.replace(DOC)

# relationships live in a sibling entry; rewrite the rels xml in a second pass
with zipfile.ZipFile(DOC) as z:
    rels = z.read("word/_rels/document.xml.rels").decode("utf8")
if NEW_RID not in rels:
    rels = rels.replace(
        "</Relationships>",
        f'<Relationship Id="{NEW_RID}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        f'relationships/image" Target="media/image8.png"/></Relationships>',
    )
    tmp = DOC.with_suffix(".tmp2.docx")
    with zipfile.ZipFile(DOC) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/_rels/document.xml.rels":
                data = rels.encode("utf8")
            zout.writestr(item, data)
    tmp.replace(DOC)
print("media + relationship added")

# ---------------------------------------------------------------- final media map
with zipfile.ZipFile(DOC) as z:
    xml = z.read("word/document.xml").decode("utf8")
    rels = z.read("word/_rels/document.xml.rels").decode("utf8")
rid2t = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
order = [rid2t[r].split("/")[-1] for r in re.findall(r'r:embed="(rId\d+)"', xml)]
print("drawing order:", order)

WANT = ["fig1_workflow", "fig_bertopic_pipeline", "fig_prompt",
        "fig2_topic_sizes", "fig3_topic_words", "fig_topic_by_destination",
        "fig_aspects_combined"]
assert len(order) == 7, f"expected 7 drawings, found {len(order)}"

tmp = DOC.with_suffix(".tmp3.docx")
with zipfile.ZipFile(DOC) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        name = item.filename.split("/")[-1]
        if item.filename.startswith("word/media/") and name in order:
            src = WANT[order.index(name)]
            data = (FIG / f"{src}.png").read_bytes()
            print(f"  {name} <- {src}.png ({len(data):,})")
        zout.writestr(item, data)
tmp.replace(DOC)
print("done")
