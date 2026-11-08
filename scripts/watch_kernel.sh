#!/bin/bash
# Poll the Kaggle kernel every 5 min, append to a status log, stop when finished.
cd "/Users/bsi-5-2200085/Documents/MY PUBLICATIONS/TOPIC MODELING-SENTIMENT TOURISM/Topic-Modeling-Aspect-Based-Sentiment-Analysis" || exit 1
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
K=~/kgtools/kgvenv/bin/kaggle
LOG=/tmp/sentence_kernel_status.log
: > "$LOG"
for i in $(seq 1 130); do
  S=$($K kernels status fathanick/sentence-absa-tourism 2>&1 | tail -1)
  echo "$(date '+%H:%M:%S') $S" >> "$LOG"
  case "$S" in
    *COMPLETE*|*ERROR*|*CANCEL*) echo "DONE $S" >> "$LOG"; exit 0 ;;
  esac
  sleep 300
done
echo "TIMEOUT watcher" >> "$LOG"
