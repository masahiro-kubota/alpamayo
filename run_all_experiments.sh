#!/bin/bash
set -e

# Target Clip (New Top Clip from 500-scan)
CLIP_ID="09312f4a-c618-46a8-a8ff-1db37e043b5d"

echo "Starting ablation experiments for clip: $CLIP_ID"

# 1. Baseline (Standard 4-cam)
echo "Running Case 1: Baseline..."
python test_camera_ablation.py $CLIP_ID --cameras 0,1,2,3 > trajectory_bias_experiment/logs/case1_baseline.log

# 2. Front Only (Variable Length)
echo "Running Case 2: Front Only..."
python test_camera_ablation.py $CLIP_ID --cameras 1 > trajectory_bias_experiment/logs/case2_front_only.log

# 3. Front Only (with Padding)
echo "Running Case 3: Front Only (Pad)..."
python test_camera_ablation.py $CLIP_ID --cameras 1 --padding > trajectory_bias_experiment/logs/case3_front_only_pad.log

# 4. No Tele (Variable Length)
echo "Running Case 4: No Tele..."
python test_camera_ablation.py $CLIP_ID --cameras 0,1,2 > trajectory_bias_experiment/logs/case4_no_tele_var.log

# 5. No Tele (with Padding)
echo "Running Case 5: No Tele (Pad)..."
python test_camera_ablation.py $CLIP_ID --cameras 0,1,2 --padding > trajectory_bias_experiment/logs/case5_no_tele_pad.log

# 6. Front + Tele (Variable Length)
echo "Running Case 6: Front + Tele..."
python test_camera_ablation.py $CLIP_ID --cameras 1,3 > trajectory_bias_experiment/logs/case6_front_tele_var.log

# 7. Front + Tele (with Padding)
echo "Running Case 7: Front + Tele (Pad)..."
python test_camera_ablation.py $CLIP_ID --cameras 1,3 --padding > trajectory_bias_experiment/logs/case7_front_tele_pad.log

# 8. Left-Right Swap (Permutation - No Padding)
echo "Running Case 8: LR Swap..."
python test_camera_ablation.py $CLIP_ID --cameras 2,1,0,3 > trajectory_bias_experiment/logs/case8_lr_swap.log

# 9. Front-Left Swap (Permutation - No Padding)
echo "Running Case 9: FL Swap..."
python test_camera_ablation.py $CLIP_ID --cameras 1,0,2,3 > trajectory_bias_experiment/logs/case9_fl_swap.log

# 10. Completely Reversed (Permutation - No Padding)
echo "Running Case 10: Completely Reversed..."
python test_camera_ablation.py $CLIP_ID --cameras 3,2,1,0 > trajectory_bias_experiment/logs/case10_reversed.log

echo "All experiments completed!"
