"""Label the 238 retry sentences locally, into a STAGING dir.

Written to output/sentence_absa_retry_local/, never merged into the Kaggle
output. Backend agreement on already-labelled sentences was 60% (mean Jaccard
0.74), so these labels are a candidate set to be confirmed by the Kaggle run,
not a replacement for it.

Run: ./venv/bin/python scripts/run_retry_local.py
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
OUT = ROOT / "output/sentence_absa_retry_local"


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
        o = json.loads(r.read())
    return o["message"]["content"], o.get("eval_count", 0)


df = pd.read_csv(ROOT / "data/processed/sentences_retry.csv")
OUT.mkdir(parents=True, exist_ok=True)
ckpt = OUT / "retry_local.jsonl"

done = set()
if ckpt.exists():
    for line in ckpt.read_text().splitlines():
        if line.strip():
            done.add(json.loads(line)["sentence_id"])
    print(f"resuming, {len(done)} already done")

t0 = time.time()
n_new = 0
with open(ckpt, "a") as fh:
    for row in df.itertuples():
        if row.sentence_id in done:
            continue
        raw, ntok = ask(row.sentence)
        res = parse_response(raw, row.sentence, resolver=canonical)
        res["aspects"] = dedupe_aspects(res["aspects"])
        rec = {
            "sentence_id": row.sentence_id,
            "review_id": row.review_id,
            "destination": row.destination,
            "stars": int(row.stars),
            "sentence": row.sentence,
            "n_aspects": len(res["aspects"]),
            "aspects": json.dumps(res["aspects"], ensure_ascii=False),
            "status": res["status"],
            "error": res["error"],
            "raw": raw[:600],
            "gen_tokens": ntok,
            "backend": "ollama-qwen2.5-7b-q4km",
        }
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        n_new += 1
        if n_new % 25 == 0:
            el = time.time() - t0
            print(f"  {n_new} new | {len(done)+n_new}/{len(df)} | "
                  f"{el/n_new:.2f}s each | eta {(len(df)-len(done)-n_new)*el/n_new/60:.1f}m")

el = time.time() - t0
print(f"\nnew rows {n_new} in {el:.1f}s ({el/max(n_new,1):.2f}s each)")

out = pd.DataFrame([json.loads(l) for l in ckpt.read_text().splitlines() if l.strip()])
out = out.drop_duplicates("sentence_id")
out.to_csv(OUT / "sentence_absa_raw.csv", index=False)

print(f"\nrows                    {len(out)}")
print(f"status                  {out.status.value_counts().to_dict()}")
print(f"sentences with >=1 asp  {int((out.n_aspects>0).sum())} "
      f"({(out.n_aspects>0).mean()*100:.1f}%)")
print(f"still zero aspects      {int((out.n_aspects==0).sum())}")
print(f"total mentions          {int(out.n_aspects.sum())}")
print(f"\nwritten {OUT}")
