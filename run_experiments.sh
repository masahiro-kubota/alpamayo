#!/bin/bash
set -e

# Ensure venv exists and activate
if [ ! -d "ar1_venv" ]; then
    echo "Creating venv..."
    python3 -m venv ar1_venv
fi
source ar1_venv/bin/activate

# Install dependencies if needed
pip install matplotlib torch numpy scipy || true

# Helper function to run experiment
run_exp() {
    CLIP="f789b390-1698-4f99-b237-6de4cbbb7666"
    NAME=$1
    ARGS=$2
    LOG="trajectory_bias_experiment/logs/${NAME}.log"
    echo "Running Case: ${NAME} with args: ${ARGS}"
    python3 -u test_camera_ablation.py $CLIP $ARGS > "$LOG" 2>&1
    echo "Finished ${NAME}. Log: ${LOG}"
}

mkdir -p trajectory_bias_experiment/logs

# Case 1: No Tele (Variable)
run_exp "case1_no_tele" "--cameras 0,1,2"

# Case 2: No Tele (Padding)
run_exp "case2_no_tele_pad" "--cameras 0,1,2 --padding"

# Case 3: Front Only (1 camera)
run_exp "case3_front_only" "--cameras 1"

# Case 4: Front Only (Padding)
run_exp "case4_front_only_pad" "--cameras 1 --padding"

# Case 5: Permutation (Front, Left, Right, Tele -> 1,0,2,3)
# Standard is 0(Left), 1(Front), 2(Right), 3(Tele)
run_exp "case5_permuted" "--cameras 1,0,2,3"

echo "All experiments completed. Logs and images are generated."
