"""Would local Ollama outputs be mergeable with the Kaggle NF4 outputs?

Kaggle ran Qwen2.5-7B-Instruct in 4-bit NF4. Ollama runs Q4_K_M. Same base
model, different quantisation, so the labels are not guaranteed identical.
Merging two backends into one dataset without measuring that is a silent
methodological break.

This replays 40 sentences that Kaggle ALREADY labelled and measures agreement.
High agreement = the local backend is a defensible stand-in for the retry set.

Run: ./venv/bin/python scripts/ollama_vs_kaggle_agreement.py
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sentiment_analysis"))

from sentence_absa import build_messages, dedupe_aspects, parse_response  # noqa: E402
from remap_aspects import canonical  # noqa: E402

MODEL = "qwen2.5:7b"
URL = "http://localhost:11434/api/chat"
N = 40


def ask(sentence):
    body = {
        "model": MODEL,
        "messages": build_messages(sentence),
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 3072, "num_predict": 512},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"]


def pairs(aspects):
    return {(a["aspect"], a["sentiment"]) for a in aspects}


raw = pd.read_csv(ROOT / "output/sentence_absa/sentence_absa_raw.csv")
# sample across the aspect-count range so it is not all trivial single-aspect rows
sample = raw.sample(N, random_state=42)

rows, t0 = [], time.time()
for r in sample.itertuples():
    kag = json.loads(r.aspects or "[]")
    res = parse_response(ask(r.sentence), r.sentence, resolver=canonical)
    loc = dedupe_aspects(res["aspects"])

    a, b = pairs(kag), pairs(loc)
    inter, union = len(a & b), len(a | b)
    rows.append({
        "sentence_id": r.sentence_id,
        "n_kaggle": len(a),
        "n_ollama": len(b),
        "jaccard": round(inter / union, 3) if union else 1.0,
        "identical": a == b,
        "both_empty": not a and not b,
        "kaggle_only": sorted(f"{x}/{y}" for x, y in a - b),
        "ollama_only": sorted(f"{x}/{y}" for x, y in b - a),
        "status": res["status"],
    })

df = pd.DataFrame(rows)
el = time.time() - t0

out = ROOT / "output/audit/ollama_vs_kaggle.csv"
df.to_csv(out, index=False)

print(f"sentences compared        {len(df)}")
print(f"elapsed                   {el:.1f}s ({el/len(df):.2f}s each)")
print(f"identical (aspect,sent)   {int(df.identical.sum())} ({df.identical.mean()*100:.1f}%)")
print(f"both empty                {int(df.both_empty.sum())}")
print(f"mean jaccard              {df.jaccard.mean():.3f}")
print(f"median jaccard            {df.jaccard.median():.3f}")
print(f"local parse status        {df.status.value_counts().to_dict()}")
print()
print("per-sentence disagreements (first 12):")
for r in df[~df.identical].head(12).itertuples():
    print(f"  jac={r.jaccard:.2f}  kaggle={r.kaggle_only}  ollama={r.ollama_only}")
print(f"\nwritten {out}")
