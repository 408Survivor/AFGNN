#!/bin/bash
# Phase C-2: multi-seed ablation runs (seeds 43~46) for the key component
# ablations + the full baseline, so that ablation deltas can be compared
# against seed noise (mean ± std).
#
# 6 configs x 4 seeds = 24 train+test units, auto-numbered into
# experiments/exp_N. no_velocity is intentionally excluded (its drop is
# already clear); region masks are deferred until component results are in.
#
# Safe to re-run: a (config, seed) pair is skipped when some exp_N already has
# its run_info.txt (exact command line) AND test_test_results.json.
#
# Usage:  bash src/scripts/run_phase_c2.sh
# Note: do NOT Ctrl+Z. Ctrl+C is fine; if a run is interrupted, delete the
# incomplete exp_N dir before re-running (see experiments/INDEX.md rules).
set -e

cd /home/ltq/Code/AFGNN
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DVlog

SEEDS=(43 44 45 46)
CONFIGS=(
  ablation/configs/ablation_full.yaml
  ablation/configs/ablation_no_static.yaml
  ablation/configs/ablation_no_temporal.yaml
  ablation/configs/ablation_no_region.yaml
  ablation/configs/ablation_no_selfloop.yaml
  ablation/configs/ablation_random_graph.yaml
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
echo "Phase C-2 finished. Tell Kimi to update INDEX.md / PROGRESS.md."
