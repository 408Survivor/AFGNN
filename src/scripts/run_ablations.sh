#!/bin/bash
# Run all facial graph ablation experiments sequentially and keep per-experiment outputs.
# Usage: bash scripts/run_ablations.sh
set -e

cd /home/ltq/DepressionCode/DepGNN/AFGNN
source $(conda info --base)/etc/profile.d/conda.sh
conda activate DVlog

RESULTS_DIR="ablation/results"
mkdir -p "${RESULTS_DIR}"

run_one() {
    local name="$1"
    local config="$2"
    echo ""
    echo "============================================================"
    echo "===== Experiment: ${name}"
    echo "===== Config: ${config}"
    echo "============================================================"

    # Skip if this experiment was already completed
    if [ -f "${RESULTS_DIR}/test_${name}.json" ]; then
        echo "Results already exist for ${name}, skipping."
        return 0
    fi

    python src/train.py --config "${config}"
    python src/test.py  --config "${config}" --split test
    cp "experiments/results/test_test_results.json" "${RESULTS_DIR}/test_${name}.json"
    cp "experiments/results/training_history.json" "${RESULTS_DIR}/history_${name}.json"
    echo "===== Finished ${name} ====="
}

# ---------------- Component ablations ----------------
for name in full no_static no_temporal no_region no_velocity no_selfloop; do
    run_one "${name}" "ablation/configs/ablation_${name}.yaml"
done

# ---------------- Region leave-one-out ablations ----------------
for config in ablation/configs/ablation_region_mask_*.yaml; do
    name=$(basename "${config}" .yaml)
    run_one "${name}" "${config}"
done

echo ""
echo "All ablation experiments completed. Results are in ${RESULTS_DIR}/"
