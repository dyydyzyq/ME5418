#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$ROOT/demo_sac.py" "$@" &
sac_pid=$!


python3 "$ROOT/demo_ppo.py" "$@" &
ppo_pid=$!



trap 'kill $ppo_pid $sac_pid 2>/dev/null' INT TERM ERR EXIT

wait $ppo_pid
wait $sac_pid

trap - INT TERM ERR EXIT
