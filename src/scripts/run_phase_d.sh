#!/bin/bash
# Phase D: remaining rerun items after Phase A/B/C/C-2.
#   1. ablation_no_velocity x seeds 43~46   (4 runs; seed 42 = exp_23 already done)
#   2. afgnn_audio_only x seeds 42~46       (5 runs; new audio-only config,
#                                             counterpart of exp_9 face-only)
#   3. region mask 00~08 x seeds 43~46      (36 runs; seed 42 = exp_26~34 done)
# Total: 45 train+test units, auto-numbered into experiments/exp_N.
#
# Safe to re-run: a (config, seed) pair is skipped when some exp_N already has
# its run_info.txt (exact command line) AND test_test_results.json.
#
# Usage:  bash src/scripts/run_phase_d.sh
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

echo "##### Part 1: no_velocity x seeds 43~46"
for seed in 43 44 45 46; do
  run_one ablation/configs/ablation_no_velocity.yaml "${seed}"
done

echo "##### Part 2: afgnn_audio_only x seeds 42~46"
for seed in 42 43 44 45 46; do
  run_one experiments/configs/afgnn_audio_only.yaml "${seed}"
done

echo "##### Part 3: region mask 00~08 x seeds 43~46"
for cfg in ablation/configs/ablation_region_mask_*.yaml; do
  for seed in 43 44 45 46; do
    run_one "${cfg}" "${seed}"
  done
done

echo ""
echo "Phase D finished. Tell Kimi to update INDEX.md / PROGRESS.md."
