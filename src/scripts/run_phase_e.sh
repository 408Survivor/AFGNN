#!/bin/bash
# Phase E: camera-ready 补实验（修订清单 revision_log.md 第 C 节）
#   1. ablation_dynamic_k1 x seeds 42~46          (5 runs; tab:face_ablation dynamic 行)
#   2. ablation_dynamic_k2_motion x seeds 42~46   (5 runs; tab:face_ablation motion 行)
#   3. afgnn_focal_adamw_cosine_audio16 x 42~46   (5 runs; tab:design_ablation 16 帧行)
#   4. afgnn_focal_adamw_cosine_audio64 x 42~46   (5 runs; tab:design_ablation 64 帧行)
#   5. afgnn_focal_adamw_cosine_concat x 42~46    (5 runs; tab:fusion_ablation concat 多seed)
#   6. afgnn_face_enhanced (weighted BCE) x 43~46 (4 runs; seed 42 = exp_2 已有)
# Total: 29 train+test units, auto-numbered into experiments/exp_N (exp_104 起).
#
# Safe to re-run: a (config, seed) pair is skipped when some exp_N already has
# its run_info.txt (exact command line) AND test_test_results.json.
#
# Usage:  bash src/scripts/run_phase_e.sh
# Note: do NOT Ctrl+Z. Ctrl+C is fine; if a run is interrupted, delete the
# incomplete exp_N dir before re-running (see experiments/INDEX.md rules).
set -e

cd /home/ltq/Code/AFGNN
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DVlog

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

run_one() {
  local cfg="$1" seed="$2"
  if is_done "${cfg}" "${seed}"; then
    echo "== SKIP  ${cfg} seed ${seed} (already completed)"
    return 0
  fi
  echo ""
  echo "============================================================"
  echo "== RUN  ${cfg} seed ${seed}"
  echo "============================================================"
  python src/train.py --config "${cfg}" --seed "${seed}"
  exp_dir=$(ls -d experiments/exp_* | sort -t_ -k2 -n | tail -1)
  python src/test.py --exp_dir "${exp_dir}" --split test
  echo "== DONE ${cfg} seed ${seed} -> ${exp_dir}"
}

for cfg in ablation/configs/ablation_dynamic_k1.yaml \
           ablation/configs/ablation_dynamic_k2_motion.yaml \
           experiments/configs/afgnn_focal_adamw_cosine_audio16.yaml \
           experiments/configs/afgnn_focal_adamw_cosine_audio64.yaml \
           experiments/configs/afgnn_focal_adamw_cosine_concat.yaml; do
  for seed in 42 43 44 45 46; do
    run_one "${cfg}" "${seed}"
  done
done

for seed in 43 44 45 46; do
  run_one experiments/configs/afgnn_face_enhanced.yaml "${seed}"
done

echo ""
echo "Phase E finished. Tell Kimi to update INDEX.md / PROGRESS.md / revision_log.md."
