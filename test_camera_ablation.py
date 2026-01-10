#!/usr/bin/env python3
"""
Test Alpamayo-R1 with Variable Camera Configurations (Ablation Study)
Usage: python test_camera_ablation.py <clip_id> --cameras <indices> [--output <filename>]

Camera Indices (Standard Order):
0: Cross Left 120°
1: Front Wide 120°
2: Cross Right 120°
3: Front Tele 30°

Example:
    python test_camera_ablation.py f789b390 --cameras 0,1,2  (No Tele)
    python test_camera_ablation.py f789b390 --cameras 1,3    (Front + Tele)
"""

import torch
import numpy as np
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

# Standard Camera Names mapped to index 0-3
STD_CAMERA_NAMES = ['Cross Left 120', 'Front Wide 120', 'Cross Right 120', 'Front Tele 30']

def main():
    parser = argparse.ArgumentParser(description="Alpamayo-R1 Camera Ablation Test")
    parser.add_argument("clip_id", help="Dataset Clip ID (e.g., f789b390)")
    parser.add_argument("--cameras", type=str, required=True, 
                        help="Comma-separated List of camera indices to uses. 0=Left, 1=Front, 2=Right, 3=Tele. E.g. '0,1,2'")
    parser.add_argument("--output", type=str, default=None, help="Output PNG filename. If None, auto-generated.")
    parser.add_argument("--padding", action="store_true", help="If set, pad missing cameras with black images instead of removing them.")
    args = parser.parse_args()

    # Parse camera indices
    try:
        cam_indices = [int(x.strip()) for x in args.cameras.split(',')]
    except ValueError:
        print("Error: --cameras must be a comma-separated list of integers.")
        sys.exit(1)

    print(f"Testing clip: {args.clip_id}")
    print(f"Selected Cameras: {cam_indices} -> {[STD_CAMERA_NAMES[i] for i in cam_indices]}")
    print(f"Padding Mode: {'ENABLED (Black Image)' if args.padding else 'DISABLED (Variable Length)'}")
    
    if args.output is None:
        cam_str = "".join(map(str, cam_indices))
        pad_str = "_pad" if args.padding else ""
        args.output = f"trajectory_bias_experiment/images/ablation_cam{cam_str}{pad_str}_{args.clip_id[:8]}.png"

    print(f"{'='*80}\n")

    # Load model
    print("Loading Alpamayo-R1 model...")
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print("✓ Model loaded\n")

    # Load data
    print(f"Loading dataset...")
    data = load_physical_aiavdataset(args.clip_id, t0_us=5_100_000)
    print("✓ Dataset loaded")

    # Extract ALL images first
    all_image_frames = data["image_frames"]  # Shape: (4, num_frames, 3, H, W)
    
    selected_camera_images = []
    selected_camera_names = []
    viz_images = []
    
    if args.padding:
        # PADDING MODE: Always 4 cameras, missing ones are black
        input_list = []
        dummy_black = torch.zeros_like(all_image_frames[0]) # (num_frames, 3, H, W)
        
        for i in range(4):
            if i in cam_indices:
                input_list.append(all_image_frames[i])
                viz_images.append(all_image_frames[i, -1].permute(1, 2, 0).numpy())
                selected_camera_names.append(STD_CAMERA_NAMES[i])
            else:
                input_list.append(dummy_black)
                viz_images.append(np.zeros((all_image_frames.shape[-2], all_image_frames.shape[-1], 3), dtype=np.uint8))
                selected_camera_names.append(f"{STD_CAMERA_NAMES[i]} (BLACK)")
        
        input_image_frames = torch.stack(input_list) # (4, num_frames, 3, H, W)
        
    else:
        # ABLATION MODE: Variable length
        indices_tensor = torch.tensor(cam_indices)
        input_image_frames = all_image_frames.index_select(0, indices_tensor)
        
        for idx in cam_indices:
            img = all_image_frames[idx, -1].permute(1, 2, 0).numpy()
            viz_images.append(img)
            selected_camera_names.append(STD_CAMERA_NAMES[idx])

    # Prepare input and run inference
    # Note: helper.create_message flattens (N, T, C, H, W) -> (N*T, C, H, W)
    messages = helper.create_message(input_image_frames.flatten(0, 1))
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
    n_cams = len(viz_images)
    fig = plt.figure(figsize=(24, 14))
    
    # Always use 2x2 layout for cameras
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.5], width_ratios=[1, 1, 0.8])
    
    # Camera positions: always 2x2 grid
    # If less than 4 cameras, fill remaining slots with placeholder
    cam_locs = [(0,0), (0,1), (1,0), (1,1)]
    all_camera_names = ['Cross Left 120', 'Front Wide 120', 'Cross Right 120', 'Front Tele 30']
    
    for slot_idx in range(4):
        ax = fig.add_subplot(gs[cam_locs[slot_idx][0], cam_locs[slot_idx][1]])
        slot_name = all_camera_names[slot_idx]
        
        # Check if this slot has an image
        if args.padding:
            # Padding mode: all 4 slots filled, black images for missing
            img = viz_images[slot_idx]
            is_black = slot_idx not in cam_indices
            label = f"#{slot_idx}: {slot_name}" + (" [BLACK]" if is_black else "")
            ax.imshow(img)
            ax.set_title(label, fontsize=12, fontweight='bold', 
                        color='red' if is_black else 'black')
        else:
            # Variable length: only show selected cameras
            if slot_idx < n_cams:
                img = viz_images[slot_idx]
                cam_idx = cam_indices[slot_idx]
                label = f"#{cam_idx}: {selected_camera_names[slot_idx]}"
                ax.imshow(img)
                ax.set_title(label, fontsize=12, fontweight='bold')
            else:
                # Empty slot
                ax.text(0.5, 0.5, f"(Not Used)\n{slot_name}", 
                       ha='center', va='center', fontsize=14, color='gray',
                       transform=ax.transAxes)
                ax.set_facecolor('#f0f0f0')
                ax.set_title(f"#{slot_idx}: {slot_name} [EMPTY]", fontsize=12, color='gray')
        ax.axis('off')
    
    # CoT panel
    ax_coc = fig.add_subplot(gs[0:2, 2])
    # Trajectory panel
    # Trajectory panel
    ax_traj = fig.add_subplot(gs[2, :])


    # CoT on the right of cameras
    # ax_coc is already created above
    ax_coc.axis('off')
    coc_display = f"""Config: {cam_indices}
Cameras: {[n.split()[0:2] for n in selected_camera_names]}

Chain-of-Causation:
['{coc_text}']

Metrics:
• Max LatDev: {max_lateral_dev:.2f}m
• minADE: {min_ade:.2f}m
"""
    ax_coc.text(0.0, 1.0, coc_display, transform=ax_coc.transAxes,
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Trajectory at bottom
    # ax_traj is already created above
    ax_traj.plot(ego_history_xyz[:, 0], ego_history_xyz[:, 1], 'g-', linewidth=2, label='Ego History')
    ax_traj.plot(gt_xyz[:, 0], gt_xyz[:, 1], color='gold', linewidth=3, label='Ground Truth')
    ax_traj.plot(pred_xyz_np[:, 0], pred_xyz_np[:, 1], 'b-', linewidth=3, label='Prediction')
    ax_traj.plot(0, 0, 'ro', label='Ego')
    
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_title(f'Trajectory Result (MaxDev: {max_lateral_dev:.3f}m)', fontsize=14)
    ax_traj.legend()
    ax_traj.axis('equal')

    plt.tight_layout()
    plt.savefig(args.output, dpi=100)
    print(f"✓ Visualization saved: {args.output}\n")
    
    print(f"RESULTS: MaxDev={max_lateral_dev:.3f}m | minADE={min_ade:.3f}m | CoT={coc_text[:50]}...")

if __name__ == "__main__":
    main()
