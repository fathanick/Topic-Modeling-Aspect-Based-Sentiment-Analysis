#!/usr/bin/env bash
# Push the retry kernel once the weekly GPU quota has reset.
#
# Kaggle's GPU allowance is 30h/week and resets at 00:00 UTC. The full 6.479-
# sentence run consumed 30,77h, so the retry can only start after the reset.
# Refuse to push while the quota still reads zero, otherwise the push fails and
# burns a kernel slot for nothing.
set -euo pipefail

cd "$(dirname "$0")/.."
export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token")"
KAGGLE="$HOME/kgtools/kgvenv/bin/kaggle"

echo "=== quota ==="
"$KAGGLE" quota

remaining=$("$KAGGLE" quota 2>/dev/null | awk '$1=="GPU"{print $3}' | tr -d 'h')
echo "gpu remaining: ${remaining}h"

if [ "${remaining%.*}" = "0" ]; then
  echo "REFUSED: GPU quota still exhausted; not pushing."
  exit 2
fi

echo "=== push ==="
"$KAGGLE" kernels push -p kaggle_kernel

echo "=== status ==="
sleep 20
"$KAGGLE" kernels status fathanick/sentence-absa-tourism
