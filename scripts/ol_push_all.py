import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from ol_push import BASE, ROOT, upload  # noqa: E402

BST = "6aefb38f0e2efc1538fb0a56"
FIG = "6aefb3900e2efc1538fb0aa3"

jobs = [
    (BASE / "main.tex", "main.tex", ROOT, False),
    (BASE / "sn-bibliography.bib", "sn-bibliography.bib", ROOT, False),
    (BASE / "sn-jnl.cls", "sn-jnl.cls", ROOT, False),
    (BASE / "bst/sn-mathphys-num.bst", "sn-mathphys-num.bst", BST, False),
]
for p in sorted((BASE / "figures").glob("*.pdf")):
    jobs.append((p, p.name, FIG, True))

for local, remote, folder, binary in jobs:
    print(f"{remote:32s}", upload(local, remote, folder, binary))
