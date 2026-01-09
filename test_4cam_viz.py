#!/usr/bin/env python3
"""
Test Alpamayo-R1 with 4-Camera Comprehensive Visualization
Layout: [4 Camera Images in 2x2] [CoC+Metrics]
        [Trajectory Plot - Full Width]
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
    print("Usage: python test_4cam_viz.py <clip_id>")
    sys.exit(1)

clip_id = sys.argv[1]
output_png = f"trajectory_bias_experiment/4cam_test_{clip_id[:8]}.png"

print(f"Testing clip: {clip_id}")
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

# Extract all 4 camera images
image_frames = data["image_frames"]  # Shape: (N_cameras, num_frames, 3, H, W)
camera_names = ['Cross Left 120°', 'Front Wide 120°', 'Cross Right 120°', 'Front Tele 30°']
camera_images = []
for i in range(4):
    cam_img = image_frames[i, -1].permute(1, 2, 0).numpy()
    camera_images.append(cam_img)

# Prepare input and run inference
messages = helper.create_message(data["image_frames"].flatten(0, 1))
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

print("Running inference...")
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
fig = plt.figure(figsize=(24, 14))
# 3 rows, 5 columns to give more control over widths
# Col 0-3 for cameras (spanning 2 wide each), Col 4 for CoT
gs = GridSpec(3, 5, figure=fig, height_ratios=[1, 1, 1.5], width_ratios=[1, 1, 1, 1, 0.8], hspace=0.2, wspace=0.15)

# Top 2 rows: 4 cameras (2x2)
# Camera 0: Row 0, Col 0-1
# Camera 1: Row 0, Col 2-3
# Camera 2: Row 1, Col 0-1
# Camera 3: Row 1, Col 2-3
camera_locations = [
    (0, 0, 0, 2), # Row 0, Col 0-2
    (0, 0, 2, 4), # Row 0, Col 2-4
    (1, 1, 0, 2), # Row 1, Col 0-2
    (1, 1, 2, 4), # Row 1, Col 2-4
]

for idx, (loc, cam_name, cam_img) in enumerate(zip(camera_locations, camera_names, camera_images)):
    ax = fig.add_subplot(gs[loc[0]:loc[1]+1, loc[2]:loc[3]])
    ax.imshow(cam_img)
    ax.set_title(f'{cam_name}', fontsize=14, fontweight='bold')
    ax.axis('off')

# Right side: CoC + Metrics (spanning 2 rows, last column)
ax_coc = fig.add_subplot(gs[0:2, 4])
ax_coc.axis('off')
coc_display = f"""Chain-of-Causation:

['{coc_text}']


Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Max Lateral Deviation: {max_lateral_dev:.2f}m
• minADE: {min_ade:.2f}m

Clip ID:
{clip_id}

Timestamp:
t0 = {data["t0_us"]:,} μs
"""
ax_coc.text(0.0, 1.0, coc_display, transform=ax_coc.transAxes,
           fontsize=13, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Bottom row: Trajectory plot (full width)
ax_traj = fig.add_subplot(gs[2, :])

# Plot trajectories
ax_traj.plot(ego_history_xyz[:, 0], ego_history_xyz[:, 1], 
            'g-', linewidth=2, label='Ego History (Input)', alpha=0.7)
ax_traj.plot(gt_xyz[:, 0], gt_xyz[:, 1], 
            color='gold', linewidth=3, label='Ground Truth Future', alpha=0.8)
ax_traj.plot(pred_xyz_np[:, 0], pred_xyz_np[:, 1], 
            'b-', linewidth=3, label='Predicted Trajectory', alpha=0.9)
ax_traj.plot(0, 0, 'ro', markersize=12, label='Ego Vehicle (t0)', zorder=10)

ax_traj.grid(True, alpha=0.3, linestyle='--')
ax_traj.set_xlabel('X (m) - Forward Direction', fontsize=13, fontweight='bold')
ax_traj.set_ylabel('Y (m) - Lateral Direction', fontsize=13, fontweight='bold')
ax_traj.set_title('Global Trajectories (Bird\'s Eye View)', fontsize=15, fontweight='bold')
ax_traj.legend(fontsize=12, loc='upper right')
ax_traj.axis('equal')

plt.tight_layout()
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"✓ 4-Camera visualization saved: {output_png}\n")

# Print results
print(f"{'='*80}")
print(f"RESULTS")
print(f"{'='*80}")
print(f"Clip ID: {clip_id}")
print(f"CoT: {coc_text}")
print(f"minADE: {min_ade:.3f} meters")
print(f"Max Lateral Deviation: {max_lateral_dev:.3f}m")
if max_lateral_dev > 2.0:
    print(f"\n✓ Model shows CURVE prediction (deviation > 2m)")
else:
    print(f"\n✗ Model shows STRAIGHT bias (deviation < 2m)")
print(f"{'='*80}\n")
