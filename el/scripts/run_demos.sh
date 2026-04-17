#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
STATE_DIR="${EL_STATE_DIR:-$PWD/.el_state_demos}"
rm -rf "$STATE_DIR"
export EL_STATE_DIR="$STATE_DIR"
el seed --from-bundled >/dev/null
el demo --all --offline --json
