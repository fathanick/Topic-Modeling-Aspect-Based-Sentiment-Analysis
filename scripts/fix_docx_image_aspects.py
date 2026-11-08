"""Rescale every inline image in the DOCX to its true aspect ratio.

Word renders wp:extent and a:ext verbatim; if either disagrees with the PNG's
pixel aspect the picture is squashed. Seven of eight figures were distorted by
3-130%. This sets width to min(true width, text column) and derives height from
the pixel ratio, so nothing is stretched and nothing overflows the margin.

Run: ../.venv/bin/python scripts/fix_docx_image_aspects.py
"""
import re
import zipfile
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT.parent / "Manuscript" / "Manuscript-v3-revised.docx"
EMU = 914400

d = docx.Document(str(DOC))
s = d.sections[0]
text_w_emu = int(s.page_width - s.left_margin - s.right_margin)

with zipfile.ZipFile(DOC) as z:
    xml = z.read("word/document.xml").decode("utf8")
    rels = z.read("word/_rels/document.xml.rels").decode("utf8")
    media = {n: z.read(n) for n in z.namelist() if n.startswith("word/media/")}
rid2f = {r: "word/media/" + f
         for r, f in re.findall(r'Id="(rId\d+)"[^>]*Target="media/([^"]+)"', rels)}

for drawing in re.finditer(r"<w:drawing>.*?</w:drawing>", xml, re.S):
    blk = drawing.group(0)
    rid = re.search(r'r:embed="(rId\d+)"', blk)
    if not rid:
        continue
    png = media.get(rid2f.get(rid.group(1), ""), b"")
    if not png:
        continue
    pw, ph = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    true_ar = pw / ph
    cx = int(re.search(r'<wp:extent cx="(\d+)"', blk).group(1))
    w = min(cx, text_w_emu)
    h = int(round(w / true_ar))
    new = re.sub(r'<wp:extent cx="\d+" cy="\d+"/>', f'<wp:extent cx="{w}" cy="{h}"/>', blk, 1)
    new = re.sub(r'(<a:xfrm><a:off[^/]*/><a:ext cx=")\d+(" cy=")\d+("/>)',
                 rf"\g<1>{int(w / 635)}\g<2>{int(h / 635)}\g<3>", new, 1)
    xml = xml.replace(blk, new)
    print(f"{rid.group(1)}: {cx/EMU:.2f}in -> {w/EMU:.2f} x {h/EMU:.2f} in (ar {true_ar:.3f})")

tmp = DOC.with_suffix(".tmp.docx")
with zipfile.ZipFile(DOC) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = xml.encode("utf8") if item.filename == "word/document.xml" else zin.read(item.filename)
        zout.writestr(item, data)
tmp.replace(DOC)
print("saved", DOC.name)
