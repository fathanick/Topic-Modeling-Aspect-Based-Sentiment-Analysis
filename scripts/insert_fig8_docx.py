"""Insert the aspect x destination heatmap as Fig. 8, after the existing Fig. 7.

The image itself is produced by scripts/make_fig6_aspect_by_destination.py from
output/sentence_absa/kalimat_aspek_sentimen.csv (8,412 real mentions); this
script only places it in the DOCX and keeps the figure order monotonic.
"""
import copy
import docx
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT.parent / "Manuscript" / "Manuscript-v3-revised.docx"
IMG = ROOT / "output" / "figures" / "fig7_aspect_destination.png"
CAPTION = ("Fig. 8. Share of each destination's aspect mentions taken by the eight most "
           "frequent aspects, for the 14 most-reviewed destinations.")

d = docx.Document(str(DOC))
paras = d.paragraphs

img_tpl = next(p for p in paras if "graphic" in p._p.xml and not p.text.strip())
cap_tpl = next(p for p in paras if p.text.strip().startswith("Fig. 7."))
anchor = next(p for p in paras if p.text.startswith("Aspect composition also varies"))
assert cap_tpl._p.getnext() is not None

rId, _image = d.part.get_or_add_image(str(IMG))

new_p = copy.deepcopy(img_tpl._p)
for blip in new_p.iter(qn("a:blip")):
    blip.set(qn("r:embed"), rId)

new_cap = copy.deepcopy(cap_tpl._p)
for fld in list(new_cap.iter(qn("w:fldSimple"))):
    fld.getparent().remove(fld)
for fc in list(new_cap.iter(qn("w:instrText"))):
    fc.getparent().remove(fc)
cs = Paragraph(new_cap, cap_tpl._parent)
for r in list(cs.runs):
    r._r.getparent().remove(r._r)
cs.add_run(CAPTION)

# place the pair after the Fig. 7 caption so numbering stays 1..8
cap_tpl._p.addnext(new_p)
new_p.addnext(new_cap)

d.save(str(DOC))
print("Fig. 8 inserted after:", cap_tpl.text[:50])
