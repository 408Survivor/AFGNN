#!/bin/bash
# Phase B: multi-seed runs (43~46) for the top-2 Phase A configs.
# Each unit = one train + one test, auto-numbered into experiments/exp_N.
# Safe to re-run: a (config, seed) pair is skipped when some exp_N already has
# its run_info.txt (exact command line) AND test_test_results.json.
#
# Usage:  bash src/scripts/run_phase_b.sh
# Note: do NOT Ctrl+Z (suspend leaves a zombie process and a broken exp_N).
# Ctrl+C is fine; if a run is interrupted, delete the incomplete exp_N dir
# before re-running this script (see experiments/INDEX.md numbering rules).
set -e

cd /home/ltq/Code/AFGNN
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DVlog

CONFIGS=(
  experiments/configs/afgnn_face_enhanced_focal_adamw_cosine.yaml   # Phase A #1 (exp_7)
  experiments/configs/afgnn_face_enhanced_focal_a05_g15.yaml       # Phase A #2 (exp_6)
)
SEEDS=(43 44 45 46)

is_done() {
  local cfg="$1" seed="$2"
  for d in experiments/exp_*/; do
    [ -f "${d}run_info.txt" ] || continue
    if grep -q -- "--config ${cfg} --seed ${seed}\b" "${d}run_info.txt" \
       && [ -f "${d}test_test_results.json" ]; then
      return 0
    fi
  done
  return 1
}

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if is_done "${cfg}" "${seed}"; then
      echo "== SKIP  ${cfg} seed ${seed} (already completed)"
      continue
    fi
    echo ""
    echo "============================================================"
    echo "== RUN  ${cfg} seed ${seed}"
    echo "============================================================"
    python src/train.py --config "${cfg}" --seed "${seed}"
    exp_dir=$(ls -d experiments/exp_* | sort -t_ -k2 -n | tail -1)
    python src/test.py --exp_dir "${exp_dir}" --split test
    echo "== DONE ${cfg} seed ${seed} -> ${exp_dir}"
  done
done

echo ""
echo "Phase B finished. Tell Kimi to update INDEX.md / PROGRESS.md."
