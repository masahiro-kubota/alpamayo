#!/bin/bash
set -e

source ar1_venv/bin/activate

CLIP="f789b390-1698-4f99-b237-6de4cbbb7666"
LOG_DIR="trajectory_bias_experiment/logs"

mkdir -p "$LOG_DIR"

run_exp() {
    ID=$1
    NAME=$2
    ARGS=$3
    LOG="${LOG_DIR}/case${ID}_${NAME}.log"
    echo "Running Case ${ID}: ${NAME} with args: ${ARGS}"
    python3 -u test_camera_ablation.py $CLIP $ARGS > "$LOG" 2>&1
    grep "RESULTS:" "$LOG" | tail -1
}

# 1. Baseline (Standard 4-cam)
run_exp "1" "baseline" "--cameras 0,1,2,3"

# 2. Front Only (Variable)
run_exp "2" "front_only" "--cameras 1"

# 3. Front Only (Padding)
run_exp "3" "front_only_pad" "--cameras 1 --padding"

# 4. No Tele (3-cam Variable)
run_exp "4" "no_tele_var" "--cameras 0,1,2"

# 5. No Tele (3-cam Padding)
run_exp "5" "no_tele_pad" "--cameras 0,1,2 --padding"

# 6. Front+Tele (2-cam Variable)
run_exp "6" "front_tele_var" "--cameras 1,3"

# 7. Front+Tele (2-cam Padding)
run_exp "7" "front_tele_pad" "--cameras 1,3 --padding"

# 8. LR Swap (Permuted)
run_exp "8" "lr_swap" "--cameras 2,1,0,3"

# 9. FL Swap (Permuted)
run_exp "9" "fl_swap" "--cameras 1,0,2,3"

# 10. Reversed (Permuted)
run_exp "10" "reversed" "--cameras 3,2,1,0"

echo ""
echo "=== ALL RESULTS ==="
grep "RESULTS" "$LOG_DIR"/case*.log
