#!/usr/bin/env python3
"""
Test Alpamayo-R1 with B-F-B-B (Black-Front-Black-Black) Camera Input
This experiment pads other camera slots with zero tensors (black images)
to keep the Front camera in its "correct" 2nd position in the input sequence.
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
    print("Usage: python test_bfbb_ablation.py <clip_id>")
    sys.exit(1)

clip_id = sys.argv[1]
output_png = f"trajectory_bias_experiment/bfbb_test_{clip_id[:8]}.png"

print(f"Testing clip (B-F-B-B Ablation): {clip_id}")
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

# Extract camera images
# Original shape: (4, 4, 3, 1080, 1920) for 4 cameras, 4 history frames
original_images = data["image_frames"]

# Create BLACK frames for 0, 2, 3 slots
black_frames = torch.zeros_like(original_images[0:1]) # (1, 4, 3, H, W)

# Construct B-F-B-B sequence
# Slot 0: Black
# Slot 1: Actual Front Wide
# Slot 2: Black
# Slot 3: Black
bfbb_images = torch.cat([
    black_frames,            # Slot 0: Cross Left (Black)
    original_images[1:2],   # Slot 1: Front Wide (Actual)
    black_frames,            # Slot 2: Cross Right (Black)
    black_frames             # Slot 3: Front Tele (Black)
], dim=0)

# Prepare input
messages = helper.create_message(bfbb_images.flatten(0, 1))
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

print("Running inference with B-F-B-B input...")
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
gs = GridSpec(3, 5, figure=fig, height_ratios=[1, 1, 1.5], width_ratios=[1, 1, 1, 1, 0.8], hspace=0.2, wspace=0.15)

# Top 2 rows: Input preview (mostly black)
camera_names = ['Cross Left (BLACK)', 'Front Wide (ACTUAL)', 'Cross Right (BLACK)', 'Front Tele (BLACK)']
camera_locations = [(0, 0, 0, 2), (0, 0, 2, 4), (1, 1, 0, 2), (1, 1, 2, 4)]

for idx, (loc, cam_name) in enumerate(zip(camera_locations, camera_names)):
    ax = fig.add_subplot(gs[loc[0]:loc[1]+1, loc[2]:loc[3]])
    img_to_show = bfbb_images[idx, -1].permute(1, 2, 0).cpu().numpy()
    ax.imshow(img_to_show)
    ax.set_title(f'{cam_name}', fontsize=14, fontweight='bold', color='red' if 'BLACK' in cam_name else 'green')
    ax.axis('off')

# Right side: CoC + Metrics
ax_coc = fig.add_subplot(gs[0:2, 4])
ax_coc.axis('off')
coc_display = f"""Chain-of-Causation (B-F-B-B):

['{coc_text}']


Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Max Lateral Deviation: {max_lateral_dev:.2f}m
• minADE: {min_ade:.2f}m

Experiment:
B-F-B-B Ablation
(Black-Front-Black-Black)

Clip ID:
{clip_id}
"""
ax_coc.text(0.0, 1.0, coc_display, transform=ax_coc.transAxes,
           fontsize=13, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Bottom row: Trajectory plot
ax_traj = fig.add_subplot(gs[2, :])
ax_traj.plot(ego_history_xyz[:, 0], ego_history_xyz[:, 1], 'g-', linewidth=2, label='Ego History (Input)', alpha=0.7)
ax_traj.plot(gt_xyz[:, 0], gt_xyz[:, 1], color='gold', linewidth=3, label='Ground Truth Future', alpha=0.8)
ax_traj.plot(pred_xyz_np[:, 0], pred_xyz_np[:, 1], 'b-', linewidth=3, label='Predicted Trajectory (B-F-B-B)', alpha=0.9)
ax_traj.plot(0, 0, 'ro', markersize=12, label='Ego Vehicle (t0)', zorder=10)

ax_traj.grid(True, alpha=0.3, linestyle='--')
ax_traj.set_xlabel('X (m) - Forward Direction', fontsize=13, fontweight='bold')
ax_traj.set_ylabel('Y (m) - Lateral Direction', fontsize=13, fontweight='bold')
ax_traj.set_title('Bird\'s Eye View: B-F-B-B ABLATION TEST', fontsize=15, fontweight='bold')
ax_traj.legend(fontsize=12, loc='upper right')
ax_traj.axis('equal')

plt.tight_layout()
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"✓ B-F-B-B ablation visualization saved: {output_png}\n")

# Print results
print(f"{'='*80}")
print(f"B-F-B-B RESULTS")
print(f"{'='*80}")
print(f"CoT: {coc_text}")
print(f"Max Lateral Deviation: {max_lateral_dev:.3f}m")
print(f"minADE: {min_ade:.3f}m")
print(f"{'='*80}\n")
