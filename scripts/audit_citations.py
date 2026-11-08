"""Audit reference list vs in-text citations for the LaTeX build.

Checks: every reference cited at least once, every citation resolves to a
reference, numbering follows first-citation order. Reads the built PDF so the
check reflects what a reviewer will see (grouped citations like [3, 4] count as
one callout with two refs).

Run: ../.venv/bin/python scripts/audit_citations.py [pdf]
"""
import re
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/jitt_build/main.pdf")

doc = fitz.open(PDF)
text = "".join(p.get_text() for p in doc)
ref_start = text.rfind("References")
body, refs = text[:ref_start], text[ref_start:]

calls = re.findall(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]", body)
first_seen, order = {}, []
for c in calls:
    nums = [int(x) for x in re.split(r"\s*,\s*", c)]
    for n in nums:
        if n not in first_seen:
            first_seen[n] = len(first_seen) + 1
            order.append(n)

listed = [int(x) for x in re.findall(r"\[(\d+)\]\s", refs)]
listed = sorted(set(listed))

print(f"citations: {len(calls)} callouts, {len(first_seen)} distinct refs")
print(f"reference list: {len(listed)} entries {listed[0]}..{listed[-1]}")
print(f"first-citation order: {order}")

problems = []
uncited = [n for n in listed if n not in first_seen]
if uncited:
    problems.append(f"references never cited: {uncited}")
dangling = [n for n in first_seen if n not in listed]
if dangling:
    problems.append(f"citations with no reference: {dangling}")
# acceptable when the bibliography lists cited refs in numeric order
if order != sorted(order):
    problems.append(f"numbering not in first-citation order: {order}")

for p in problems:
    print("FAIL", p)
print("\nRESULT:", "clean" if not problems else f"{len(problems)} problem(s)")
