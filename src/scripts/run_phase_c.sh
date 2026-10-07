#!/bin/bash
# Phase C: ablation reruns (seed 42) on the post-fix features.
# 16 ablation configs, each = one train + one test, auto-numbered into
# experiments/exp_N. Base of the ablation family is ablation_full.yaml
# (equivalent to afgnn_face_enhanced_focal.yaml, the exp_1 config).
#
# Safe to re-run: a (config, seed) pair is skipped when some exp_N already has
# its run_info.txt (exact command line) AND test_test_results.json.
#
# Usage:  bash src/scripts/run_phase_c.sh
# Note: do NOT Ctrl+Z. Ctrl+C is fine; if a run is interrupted, delete the
# incomplete exp_N dir before re-running (see experiments/INDEX.md rules).
set -e

cd /home/ltq/Code/AFGNN
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DVlog

SEED=42
CONFIGS=(
  ablation/configs/ablation_full.yaml
  ablation/configs/ablation_no_static.yaml
  ablation/configs/ablation_no_temporal.yaml
  ablation/configs/ablation_no_region.yaml
  ablation/configs/ablation_no_velocity.yaml
  ablation/configs/ablation_no_selfloop.yaml
  ablation/configs/ablation_random_graph.yaml
  ablation/configs/ablation_region_mask_00_face_contour.yaml
  ablation/configs/ablation_region_mask_01_left_eyebrow.yaml
  ablation/configs/ablation_region_mask_02_right_eyebrow.yaml
  ablation/configs/ablation_region_mask_03_nose_bridge.yaml
  ablation/configs/ablation_region_mask_04_nose_bottom.yaml
  ablation/configs/ablation_region_mask_05_left_eye.yaml
  ablation/configs/ablation_region_mask_06_right_eye.yaml
  ablation/configs/ablation_region_mask_07_outer_mouth.yaml
  ablation/configs/ablation_region_mask_08_inner_mouth.yaml
)

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
  if is_done "${cfg}" "${SEED}"; then
    echo "== SKIP  ${cfg} seed ${SEED} (already completed)"
    continue
  fi
  echo ""
  echo "============================================================"
  echo "== RUN  ${cfg} seed ${SEED}"
  echo "============================================================"
  python src/train.py --config "${cfg}" --seed "${SEED}"
  exp_dir=$(ls -d experiments/exp_* | sort -t_ -k2 -n | tail -1)
  python src/test.py --exp_dir "${exp_dir}" --split test
  echo "== DONE ${cfg} seed ${SEED} -> ${exp_dir}"
done

echo ""
echo "Phase C finished. Tell Kimi to update INDEX.md / PROGRESS.md."
