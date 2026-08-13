#!/usr/bin/env bash
# End-to-end smoke checks against a running backend on :8000.
# Expects: scripts/dev.sh running (or app deployed).
set -uo pipefail
B="http://localhost:8000"
pass=0; fail=0
chk() { # chk <label> <expected> <actual>
  if [ "$2" == "$3" ]; then pass=$((pass+1)); echo "PASS  $1"; else fail=$((fail+1)); echo "FAIL  $1 (expected=$2 actual=$3)"; fi
}
chk "health"      "ok" "$(curl -s $B/api/health | grep -o '"ok"')"
chk "root 200"    "200" "$(curl -s -o /dev/null -w '%{http_code}' $B/)"
chk "generate_204" "204" "$(curl -s -o /dev/null -w '%{http_code}' $B/generate_204)"
chk "spa fallback" "200" "$(curl -s -o /dev/null -w '%{http_code}' -H 'Accept: text/html' $B/dashboard)"
chk "missing asset 404" "404" "$(curl -s -o /dev/null -w '%{http_code}' $B/nonexistent.xyz)"
echo "---"
echo "passed=$pass failed=$fail"
[ $fail -eq 0 ] || exit 1