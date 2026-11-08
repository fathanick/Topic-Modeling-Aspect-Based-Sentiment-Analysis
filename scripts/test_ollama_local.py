"""Can Ollama run the retry set locally, and how fast?

Kaggle's weekly GPU quota is exhausted, so the 238 retry sentences are blocked
on an external service. This checks the local fallback: same prompt, same
parser, Metal GPU. Prints wall-clock per sentence so the 238-sentence run can
be estimated from measurement rather than guesswork.

Run: ./venv/bin/python scripts/test_ollama_local.py
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sentiment_analysis"))

from sentence_absa import SYSTEM_PROMPT, build_messages, parse_response  # noqa: E402

MODEL = "qwen2.5:7b"
URL = "http://localhost:11434/api/chat"


def ask(sentence, num_ctx=3072):
    body = {
        "model": MODEL,
        "messages": build_messages(sentence),
        "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": 512},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    return out["message"]["content"], time.time() - t0


import pandas as pd

df = pd.read_csv(ROOT / "data/processed/sentences_retry.csv").head(5)
times = []
for row in df.itertuples():
    raw, dt = ask(row.sentence)
    res = parse_response(raw, row.sentence)
    kept, dropped = res["aspects"], [p for p in res["error"].split("; ") if p]
    times.append(dt)
    print(f"[{dt:5.1f}s] status={res['status']} {row.sentence[:70]!r}")
    print(f"         aspects={len(kept)} dropped={len(dropped)}")
    for k in kept:
        print(f"         - {k['aspect']}/{k['sentiment']} kw={k['keywords']!r}")

print(f"\nmean {sum(times)/len(times):.1f}s  max {max(times):.1f}s")
print(f"estimate for 238 sentences: {238*sum(times)/len(times)/60:.1f} min")
