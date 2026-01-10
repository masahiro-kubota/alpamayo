#!/bin/bash
# Activate venv
if [ -d "ar1_venv" ]; then
    source ar1_venv/bin/activate
else
    echo "Creating venv..."
    python3 -m venv ar1_venv
    source ar1_venv/bin/activate
fi

# Install dependencies just in case
pip install -e .
pip install matplotlib torch numpy

# Create logs dir
mkdir -p trajectory_bias_experiment/logs

echo "Running Case 1..."
python test_camera_ablation.py f789b390-1698-4f99-b237-6de4cbbb7666 --cameras 0,1,2 > trajectory_bias_experiment/logs/case1_no_tele.log 2>&1

echo "Running Case 2..."
python test_camera_ablation.py f789b390-1698-4f99-b237-6de4cbbb7666 --cameras 0,1,2 --padding > trajectory_bias_experiment/logs/case2_no_tele_pad.log 2>&1

echo "Running Case 3..."
python test_camera_ablation.py f789b390-1698-4f99-b237-6de4cbbb7666 --cameras 1,3 > trajectory_bias_experiment/logs/case3_front_only.log 2>&1

echo "Running Case 4..."
python test_camera_ablation.py f789b390-1698-4f99-b237-6de4cbbb7666 --cameras 1,3 --padding > trajectory_bias_experiment/logs/case4_front_only_pad.log 2>&1

echo "Running Case 5 (Permutation)..."
# Standard order is 0,1,2,3. We swap 0 (Left) and 1 (Front) -> 1,0,2,3
python test_camera_ablation.py f789b390-1698-4f99-b237-6de4cbbb7666 --cameras 1,0,2,3 > trajectory_bias_experiment/logs/case5_permuted.log 2>&1

echo "All experiments completed."
