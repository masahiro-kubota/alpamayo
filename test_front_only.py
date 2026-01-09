#!/usr/bin/env python3
"""
Test Alpamayo-R1 with FRONT-ONLY Camera Input on High-Curvature Clip
This script compares behavior when ONLY the front camera is provided vs all 4.
"""

import torch
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

if len(sys.argv) < 2:
    print("Usage: python test_front_only.py <clip_id>")
    sys.exit(1)

clip_id = sys.argv[1]
output_png = f"trajectory_bias_experiment/front_only_test_{clip_id[:8]}.png"

print(f"Testing clip (FRONT ONLY): {clip_id}")
print(f"{'='*80}\n")

# Load model
print("Loading Alpamayo-R1 model...")
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)
print("✓ Model loaded\n")

# Load data
print(f"Loading dataset...")
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
print("✓ Dataset loaded")

# Extract front camera image for visualization
# Index 1 is Front Wide 120°
front_camera_img = data["image_frames"][1, -1].permute(1, 2, 0).cpu().numpy()

# Prepare input: ONLY FRONT CAMERA
# data["image_frames"] shape: (4, 4, 3, H, W)
front_only_frames = data["image_frames"][1:2] # Shape: (1, 4, 3, H, W)
messages = helper.create_message(front_only_frames.flatten(0, 1))

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=False,
    continue_final_message=True, return_dict=True, return_tensors="pt",
)
model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}
model_inputs = helper.to_device(model_inputs, "cuda")

print("Running inference with FRONT CAMERA ONLY...")
torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs, top_p=0.98, temperature=0.6,
        num_traj_samples=1, max_generation_length=256, return_extra=True,
    )

# Compute metrics
gt_xyz = data["ego_future_xyz"].cpu()[0, 0, :, :3].numpy()
ego_history_xyz = data["ego_history_xyz"].cpu()[0, 0, :, :3].numpy()
pred_xyz_np = pred_xyz.cpu().numpy()[0, 0, 0, :, :3]
coc_text = extra['cot'][0][0]
max_lateral_dev = np.abs(pred_xyz_np[:, 1]).max()
min_ade = np.linalg.norm(pred_xyz_np - gt_xyz, axis=1).mean()

# Create visualization
fig = plt.figure(figsize=(24, 10))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

# Top Left: Front Camera Image (the ONLY input)
ax_img = fig.add_subplot(gs[0, 0])
ax_img.imshow(front_camera_img)
ax_img.set_title(f'Front Camera Input (ONLY THIS WAS FED TO MODEL)', fontsize=16, fontweight='bold', color='red')
ax_img.axis('off')

# Top Right: CoC Panel
ax_coc = fig.add_subplot(gs[0, 1])
ax_coc.axis('off')
coc_display = f"""Chain-of-Causation (Front-Only):

['{coc_text}']


Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Max Lateral Deviation: {max_lateral_dev:.2f}m
• minADE: {min_ade:.2f}m

Clip ID:
{clip_id}

Experiment: FRONT-ONLY INPUT
"""
ax_coc.text(0.05, 0.95, coc_display, transform=ax_coc.transAxes,
           fontsize=14, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Bottom row: Trajectory plot
ax_traj = fig.add_subplot(gs[1, :])

# Plot trajectories
ax_traj.plot(ego_history_xyz[:, 0], ego_history_xyz[:, 1], 
            'g-', linewidth=2, label='Ego History (Input)', alpha=0.7)
ax_traj.plot(gt_xyz[:, 0], gt_xyz[:, 1], 
            color='gold', linewidth=3, label='Ground Truth Future', alpha=0.8)
ax_traj.plot(pred_xyz_np[:, 0], pred_xyz_np[:, 1], 
            'b-', linewidth=3, label='Predicted Trajectory (Front-Only)', alpha=0.9)
ax_traj.plot(0, 0, 'ro', markersize=12, label='Ego Vehicle (t0)', zorder=10)

ax_traj.grid(True, alpha=0.3, linestyle='--')
ax_traj.set_xlabel('X (m) - Forward Direction', fontsize=13, fontweight='bold')
ax_traj.set_ylabel('Y (m) - Lateral Direction', fontsize=13, fontweight='bold')
ax_traj.set_title('Bird\'s Eye View: FRONT-ONLY INFERENCE', fontsize=16, fontweight='bold')
ax_traj.legend(fontsize=12, loc='upper right')
ax_traj.axis('equal')

plt.tight_layout()
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"✓ Front-only visualization saved: {output_png}\n")

# Print results
print(f"{'='*80}")
print(f"FRONT-ONLY RESULTS")
print(f"{'='*80}")
print(f"Clip ID: {clip_id}")
print(f"CoT: {coc_text}")
print(f"Max Lateral Deviation: {max_lateral_dev:.3f}m")
print(f"{'='*80}\n")
