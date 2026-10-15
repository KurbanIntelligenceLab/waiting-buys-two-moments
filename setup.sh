#!/bin/sh
# Build the wheel, install it (not editable) into a fresh virtual environment,
# and check imports and module entry points from outside the source tree.
#
#   sh setup.sh [DATA_ROOT]
#
# DATA_ROOT (default: this checkout) is a source checkout containing the saved EM data; the EM claim verification runs against it
# through WBTM_ROOT. Uses constraints.txt for the dependency versions.
set -eu
SRC=$(cd "$(dirname "$0")" && pwd)
DATA=$(cd "${1:-$SRC}" && pwd)
PY=${PYTHON:-python3}
WORK=$(mktemp -d)
# setuptools writes build/ and wtm.egg-info/ into the source tree; remove them
# afterwards unless they were already there
CLEAN=""
[ -e "$SRC/build" ] || CLEAN="$CLEAN $SRC/build"
[ -e "$SRC/wtm.egg-info" ] || CLEAN="$CLEAN $SRC/wtm.egg-info"
trap 'rm -rf "$WORK" $CLEAN' EXIT

"$PY" -m pip wheel --no-deps -w "$WORK/dist" "$SRC" >/dev/null
WHEEL=$(ls "$WORK"/dist/wtm-*.whl)
echo "built $(basename "$WHEEL")"
"$PY" -m zipfile -l "$WHEEL" | awk '{print $1}' | grep -E '^(wtm|experiments)/' > "$WORK/members.txt"
for f in experiments/em_mechanism/verify_claims.py experiments/em_mechanism/snapshots.py \
         experiments/em_mechanism/PROTOCOL.md experiments/em_mechanism/MECHANISM_PROTOCOL.md \
         experiments/config/frozen_benchmark.yaml wtm/npmle.py; do
  grep -qx "$f" "$WORK/members.txt" || { echo "wheel lacks $f"; exit 1; }
done

"$PY" -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install -q -c "$SRC/constraints.txt" "$WHEEL"
cd "$WORK"                                    # outside the source tree
"$WORK/venv/bin/python" - <<'EOF'
import importlib, pathlib
mods = ["wtm", "wtm.npmle", "wtm.provenance", "experiments.scripts.benchmark",
        "experiments.audit.observation_window_audit", "experiments.em_mechanism.core",
        "experiments.em_mechanism.snapshots", "experiments.em_mechanism.verify_claims",
        "experiments.em_mechanism.compare_mechanism_outputs"]
for m in mods:
    mod = importlib.import_module(m)
    assert "site-packages" in mod.__file__, (m, mod.__file__)
print(f"imports OK from site-packages ({len(mods)} modules)")
EOF
# without a data root, the EM verification must fail loudly (exit 2)
set +e
"$WORK/venv/bin/python" -m experiments.em_mechanism.verify_claims >/dev/null 2>"$WORK/err.txt"
rc=$?
set -e
[ "$rc" -eq 2 ] && grep -q "EM STUDY OUTPUTS MISSING" "$WORK/err.txt" || { echo "expected exit 2 without data, got $rc"; cat "$WORK/err.txt"; exit 1; }
echo "verify_claims without WBTM_ROOT: exit 2 (missing data reported), as required"
WBTM_ROOT="$DATA" "$WORK/venv/bin/python" -m experiments.em_mechanism.verify_claims | tail -1
echo "WHEEL CHECK PASSED"
