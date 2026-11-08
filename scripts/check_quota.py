"""Check the Groq free-tier daily quota (TPD) before a long run.

Free tier: 200,000 tokens/day per model. A full 1,530-review run needs
~790,000 tokens, so it CANNOT finish in one day. Run this to see remaining
budget and the model's per-day limit.

Usage:  GROQ_API_KEY=... ./venv/bin/python scripts/check_quota.py
"""
import os
from groq import Groq

PROBE_TOKENS = 20  # a single tiny call reveals the remaining TPD in the error


def main():
    c = Groq(api_key=os.environ["GROQ_API_KEY"])
    for model in ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]:
        try:
            c.chat.completions.create(model=model,
                                      messages=[{"role": "user", "content": "ok"}],
                                      max_tokens=PROBE_TOKENS)
            print(f"{model:24} available (no quota error at probe time)")
        except Exception as e:
            msg = str(e)
            if "tokens per day" in msg:
                import re
                limit = re.search(r"\(TPD\): Limit (\d+), Used (\d+)", msg)
                eta = re.search(r"try again in ([^\'.]+)", msg)
                if limit:
                    tot, used = int(limit.group(1)), int(limit.group(2))
                    print(f"{model:24} TPD {used:,}/{tot:,} "
                          f"(remaining {tot-used:,}) reset in {eta.group(1) if eta else '?'}")
                else:
                    print(f"{model:24} TPD exhausted: {msg[:120]}")
            elif "tokens per minute" in msg:
                print(f"{model:24} TPM limited (daily budget likely OK)")
            else:
                print(f"{model:24} {type(e).__name__}: {msg[:100]}")


if __name__ == "__main__":
    main()
