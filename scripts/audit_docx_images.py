"""DOCX image audit: aspect-ratio distortion and placed width vs text column.

Word shows an inline image stretched to the extent stored in wp:extent. If that
extent does not match the PNG's own pixel aspect ratio the picture is squashed.
Report each drawing with its ratio error; FAIL above 2%.
"""
import io
import re
import zipfile
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT.parent / "Manuscript" / "Manuscript-v3-revised.docx"
z = zipfile.ZipFile(DOC)
xml = z.read("word/document.xml").decode("utf8")
rels = z.read("word/_rels/document.xml.rels").decode("utf8")
rid2f = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="media/([^"]+)"', rels))

d = docx.Document(str(DOC))
s = d.sections[0]
text_w = (s.page_width - s.left_margin - s.right_margin) / 914400

rows, bad = [], 0
for m in re.finditer(r'<wp:extent cx="(\d+)" cy="(\d+)"/>(.{0,4000}?)r:embed="(rId\d+)"',
                     xml, re.S):
    cx, cy, rid = int(m.group(1)), int(m.group(2)), m.group(4)
    f = rid2f.get(rid)
    if not f:
        continue
    png = z.read("word/media/" + f)
    pw, ph = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    placed_ratio = cx / cy
    true_ratio = pw / ph
    err = abs(placed_ratio / true_ratio - 1) * 100
    w_in = cx / 914400
    rows.append((rid, f, pw, ph, round(w_in, 2), round(placed_ratio, 3),
                 round(true_ratio, 3), round(err, 1)))
    if err > 2:
        bad += 1

print(f"text column width: {text_w:.2f} in")
print("rid        file                px        placed_in  placed_ar  true_ar  err%")
for r in rows:
    print(f"{r[0]:10s} {r[1]:19s} {r[2]}x{r[3]:<6} {r[4]:>8} {r[5]:>9} {r[6]:>8} {r[7]:>6}")
print(f"\nfigures with >2% aspect distortion: {bad}/{len(rows)}")
