#!/usr/bin/env bash
# Reproduce the el v0 demo suite end to end, from a clean checkout, with
# no network, no LLM, and no GPU required.
#
# Usage:
#   ./reproduce.sh
#
# This script:
#   1. Installs el in editable mode.
#   2. Seeds the skill registry from the committed seed corpus
#      (NL2Bash subset + tldr + synthesized man page summaries).
#   3. Runs the offline demo suite and checks exit codes.
#   4. Prints a one-line verdict per demo.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v el >/dev/null 2>&1; then
    echo "[reproduce] installing el (editable)"
    pip install -e . >/dev/null
fi

STATE_DIR="${EL_STATE_DIR:-$PWD/.el_state_repro}"
rm -rf "$STATE_DIR"
export EL_STATE_DIR="$STATE_DIR"

echo "[reproduce] seeding skill registry at $STATE_DIR"
el seed --from-bundled

echo "[reproduce] running offline demo suite"
el demo --all --offline --workspace --json > "$STATE_DIR/demo_report.json"

python - <<PY
import json, sys
report = json.load(open("$STATE_DIR/demo_report.json"))
total = len(report["demos"])
ok = sum(1 for d in report["demos"] if d["status"] == "ok")
skipped = sum(1 for d in report["demos"] if d["status"] == "skipped")
failed = total - ok - skipped
for d in report["demos"]:
    mark = {"ok": "ok", "skipped": "skip"}.get(d["status"], "FAIL")
    print(f"  [{mark}] {d['name']}  ({d['duration_ms']} ms)")
print(f"[reproduce] {ok}/{total} ok, {skipped} skipped (online), {failed} failed")
sys.exit(0 if failed == 0 else 1)
PY
