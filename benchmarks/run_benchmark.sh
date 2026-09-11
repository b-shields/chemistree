#!/usr/bin/env bash
# Full DUD-Z benchmark runner. For every target it launches the whole arm x track
# matrix at once (a per-target batch, run in parallel), then moves to the next target:
#   - 2D editing        : naked / generalist / chemistree
#   - 3D probes         : generalist / chemistree            (naked has no 3D access)
#   - 3D decoration     : generalist / chemistree, guided AND --no-guidance (ablation)
# = 9 runs per target. Each run does REPEAT samples of every case internally, writing one
# row per sample (tagged with a 'replicate' index) into its single out file -- so a whole
# target is 9 parallel jobs no matter the sample count. Result rows land uncompressed under
# benchmarks/results/full/<target>/, each run's stdout in a sibling .log.
#
# Defaults: n=3 samples on haiku (haiku is stochastic). Override with REPEAT / MODEL.
#
# The per-case timeout is a generous hang-guard (default 600s); efficiency is measured by
# cost/tokens, not wall time (tools run locally in ms; model + API round-trips dominate).
#
# Requirements: the `chemistree` conda env (poetry install); a `smina` env
# (conda create -n smina -c conda-forge smina); the `claude` CLI on PATH.
#
# Usage (from anywhere):
#   bash benchmarks/run_benchmark.sh                 # all 10 DUD-Z targets
#   bash benchmarks/run_benchmark.sh abl1            # just the abl1 smoke test
#   bash benchmarks/run_benchmark.sh egfr parp1      # a subset
#   REPEAT=1 bash benchmarks/run_benchmark.sh        # a quick single-sample pass
#   MODEL=sonnet bash benchmarks/run_benchmark.sh    # a stronger model
#   MAXPAR=6 bash benchmarks/run_benchmark.sh        # tighter concurrency cap (needs bash 4.3+)
# Env: MODEL(haiku) REPEAT(3) TIMEOUT(600) MAXPAR(10; 0=whole target batch at once) TRACE(0)
#      OUT(benchmarks/results/full) SMINA_BIN.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"

# smina lives in its own env (openbabel conflicts with the chemistree env's Python 3.14).
: "${SMINA_BIN:=$(conda run -n smina which smina 2>/dev/null)}"
export SMINA_BIN
[ -z "$SMINA_BIN" ] && echo "warning: SMINA_BIN unset; the 3D tracks need smina" >&2

MODEL="${MODEL:-haiku}"
REPEAT="${REPEAT:-3}"     # samples/case: haiku is stochastic, so n=3 by default
TIMEOUT="${TIMEOUT:-600}"
MAXPAR="${MAXPAR:-10}"    # 0 = launch a whole target's batch at once; >0 caps concurrency
OUT="${OUT:-$REPO/benchmarks/results/full}"
TRACE_FLAG=""; [ "${TRACE:-0}" = "1" ] && TRACE_FLAG="--trace"
C="$REPO/benchmarks/cases"

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && \
  TARGETS=(egfr aa2ar andr hivpr fa10 hdac8 parp1 hs90a ada nram abl1)

run() { conda run -n chemistree poetry run python -m benchmarks.run "$@" 2>&1 | grep -v DEPRECATION; }

# Cap concurrency when MAXPAR>0 (needs bash 4.3+ for `wait -n`); otherwise no-op, so a
# whole target batch launches together and we `wait` for it before the next target.
throttle() {
  [ "$MAXPAR" -gt 0 ] || return 0
  while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do wait -n; done
}

# launch <items> <arm> <outfile> [extra run.py flags...]
launch() {
  local items="$1" arm="$2" out="$3"; shift 3
  if [ ! -f "$items" ]; then echo "skip (missing): $items" >&2; return; fi
  if [ "${DRY:-0}" = "1" ]; then
    echo "  run --items $(basename "$items") --arm $arm --repeat $REPEAT $* --out $(basename "$out")"
    return
  fi
  throttle
  run --items "$items" --arm "$arm" --model "$MODEL" --timeout "$TIMEOUT" \
      --repeat "$REPEAT" $TRACE_FLAG "$@" --out "$out" > "$out.log" 2>&1 &
}

echo "targets: ${TARGETS[*]}  | model=$MODEL repeat=$REPEAT timeout=$TIMEOUT maxpar=$MAXPAR"
echo "results -> $OUT"

for target in "${TARGETS[@]}"; do
  D="$OUT/$target"
  rm -rf "$D"; mkdir -p "$D"
  echo "########## $target : launching batch ##########"
  # Each run.py handles the REPEAT samples internally, writing one row per sample to its
  # single out file (tagged with a 'replicate' index), so a whole target is 9 parallel
  # jobs regardless of REPEAT.
  # 2D editing -- all three arms (naked is a valid text baseline from the SMILES).
  for arm in naked generalist chemistree; do
    launch "$C/${target}_2d.jsonl" "$arm" "$D/2d_${arm}.jsonl"
  done
  # 3D probes -- chemistree vs generalist (naked has no 3D access).
  for arm in generalist chemistree; do
    launch "$C/${target}_3d_probes.jsonl" "$arm" "$D/probes_${arm}.jsonl"
  done
  # 3D decoration -- chemistree vs generalist, guided and no-guidance ablation.
  for arm in generalist chemistree; do
    launch "$C/${target}_3d_decoration.jsonl" "$arm" "$D/decorate_${arm}.jsonl"
    launch "$C/${target}_3d_decoration.jsonl" "$arm" \
           "$D/decorate_${arm}_noguid.jsonl" --no-guidance
  done
  wait   # finish this target's batch before starting the next
  echo "########## $target : done ##########"
done
echo "ALL TARGETS DONE -> $OUT"
