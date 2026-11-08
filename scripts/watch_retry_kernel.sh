#!/usr/bin/env bash
# Poll the retry kernel to completion, then fetch its output.
#
# The 238-sentence retry is minutes of GPU work but sits behind install + model
# load, so the kernel can read RUNNING for ~10-15 min. Poll instead of blocking
# a foreground call.
set -uo pipefail

cd "/Users/bsi-5-2200085/Documents/MY PUBLICATIONS/TOPIC MODELING-SENTIMENT TOURISM/Topic-Modeling-Aspect-Based-Sentiment-Analysis"
export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token")"
KAGGLE="$HOME/kgtools/kgvenv/bin/kaggle"
LOG=/tmp/retry_kernel_status.log
: > "$LOG"

for i in $(seq 1 120); do
  st=$("$KAGGLE" kernels status fathanick/sentence-absa-tourism 2>/dev/null | grep -o 'KernelWorkerStatus\.[A-Z]*')
  echo "$(date '+%H:%M:%S') $st" >> "$LOG"
  case "$st" in
    *COMPLETE*) echo "COMPLETE after $i polls" >> "$LOG"; break ;;
    *ERROR*|*CANCEL*) echo "FAILED: $st" >> "$LOG"; break ;;
  esac
  sleep 60
done

if grep -q "COMPLETE" "$LOG"; then
  rm -rf /tmp/kretry
  "$KAGGLE" kernels output fathanick/sentence-absa-tourism -p /tmp/kretry >> "$LOG" 2>&1
  echo "--- downloaded ---" >> "$LOG"
  find /tmp/kretry -type f >> "$LOG" 2>/dev/null
fi

tail -12 "$LOG"
