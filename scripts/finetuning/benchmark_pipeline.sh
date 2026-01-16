#!/bin/bash
# Unified Pipeline for Alpamayo-R1 Trajectory Evaluation

set -e

# Configuration
CLIP_IDS="09312f4a-c618-46a8-a8ff-1db37e043b5d,297615c4-dae9-40b6-9051-309ef3dcb02d,559ac095-5325-4a2d-a13c-08944d35d106,df0687d0-a2a3-4534-95e0-164f02bec8af,7a22a3d6-1156-427f-b8d6-ac379c6f9acb"
CKPT_PATH="./debug_ckpt/ckpt_epoch_0.pt"

export PYTHONPATH=$PYTHONPATH:$(pwd)/src
source ar1_venv/bin/activate

echo "=========================================================="
echo " Starting Alpamayo-R1 Evaluation Pipeline"
echo "=========================================================="

# 1. Prepare Data (Only if files don't exist)
echo -e "\n[Step 1/2] Preparing Evaluation Data..."
python scripts/finetuning/prepare_training_data.py data/high_curv_eval_5samples.pt \
    --clip_ids "$CLIP_IDS" --samples 5

python scripts/finetuning/prepare_training_data.py data/high_curv_eval_full_vision.pt \
    --clip_ids "$CLIP_IDS" --samples 5 --no_mask

# 2. Run Comprehensive Benchmark
echo -e "\n[Step 2/2] Running Benchmark (Inference & Comparison)..."
python scripts/finetuning/run_comparison_benchmark.py \
    --masked_data data/high_curv_eval_5samples.pt \
    --full_data data/high_curv_eval_full_vision.pt \
    --ckpt_path "$CKPT_PATH" \
    --num_samples 5

echo -e "\n=========================================================="
echo " Evaluation Pipeline Completed Successfully!"
echo " Results: trajectory_bias_experiment/images/benchmark_vs_*.png"
echo "=========================================================="
