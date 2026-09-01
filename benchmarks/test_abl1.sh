#!/usr/bin/env bash
# Reproduce the abl1 benchmark mock-up. Writes uncompressed result rows under
# benchmarks/results/test_2d and test_3d:
#   - 2D editing: all three arms (chemistree / generalist / naked).
#   - 3D scaffold decoration (guidance x arm) + understanding probes:
#     chemistree vs generalist only -- naked has no 3D access, so it is dropped.
# The uniform per-case timeout is overridable via TIMEOUT (default 300s).
#
# Requirements:
#   - the `chemistree` conda env with `poetry install`
#   - a `smina` env: conda create -n smina -c conda-forge smina
#   - the `claude` CLI on PATH
#
# Usage (from anywhere; ~25-30 min on haiku, best backgrounded):
#   bash benchmarks/test_abl1.sh
#   SMINA_BIN=/path/to/smina bash benchmarks/test_abl1.sh   # explicit smina
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"

# smina lives in its own env (its openbabel dep conflicts with the chemistree
# env's Python 3.14). Point SMINA_BIN at the binary, or have `smina` on PATH.
: "${SMINA_BIN:=$(conda run -n smina which smina 2>/dev/null)}"
export SMINA_BIN
[ -z "$SMINA_BIN" ] && echo "warning: SMINA_BIN unset; the 3D tracks need smina" >&2

run() { conda run -n chemistree poetry run python -m benchmarks.run "$@" 2>&1 | grep -v DEPRECATION; }

RESULTS_2D="$REPO/benchmarks/results/test_2d"
RESULTS_3D="$REPO/benchmarks/results/test_3d"
mkdir -p "$RESULTS_2D" "$RESULTS_3D"
rm -f "$RESULTS_2D"/*.jsonl "$RESULTS_3D"/*.jsonl

C="$REPO/benchmarks/cases"
# A uniform per-case timeout. A hard subprocess timeout kills the agent mid-run and
# yields no FINAL_SMILES, so keep it generous enough for the slow decoration cases.
TIMEOUT="${TIMEOUT:-300}"

# 2D editing -- all three arms (naked is a valid text baseline from the SMILES).
for arm in naked generalist chemistree; do
  echo "########## 2D editing: $arm ##########"
  run --items "$C/abl1_2d.jsonl" --arm "$arm" --timeout "$TIMEOUT" \
      --out "$RESULTS_2D/${arm}.jsonl"
done

# 3D probes + scaffold decoration -- chemistree vs generalist only. Naked has no 3D
# access (SMILES only), so its 3D result is noise and is dropped entirely.
for arm in generalist chemistree; do
  echo "########## 3D: $arm ##########"
  echo "--- understanding probes ---"
  run --items "$C/abl1_3d_probes.jsonl" --arm "$arm" --timeout "$TIMEOUT" \
      --out "$RESULTS_3D/probes_${arm}.jsonl"
  echo "--- scaffold decoration (guided) ---"
  run --items "$C/abl1_3d_decoration.jsonl" --arm "$arm" --timeout "$TIMEOUT" \
      --out "$RESULTS_3D/decorate_${arm}.jsonl"
  echo "--- scaffold decoration (no guidance / ablation) ---"
  run --items "$C/abl1_3d_decoration.jsonl" --arm "$arm" --timeout "$TIMEOUT" \
      --no-guidance --out "$RESULTS_3D/decorate_${arm}_noguid.jsonl"
done
echo "########## ALL DONE ##########"
