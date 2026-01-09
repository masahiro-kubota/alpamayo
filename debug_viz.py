#!/usr/bin/env python
"""
Alpamayo Debug Visualization Script

DESCRIPTION:
    This script is a modified version of run_final_viz.py designed for debugging and "trial-and-error"
    experimentation. It allows you to adjust model generation parameters (temperature, top_p)
    and visualize multiple sampled trajectories to understand the model's uncertainty and distribution.

USAGE:
    # Generate 10 samples with higher temperature to encourage diversity
    python debug_viz.py --ratio 0.6 --num_samples 10 --temperature 0.8 --output debug_output.png

ARGUMENTS:
    ... (Standard args from run_final_viz.py) ...
    
    --num_samples INT   Number of trajectories to sample (default: 5)
    --temperature FLOAT Sampling temperature (default: 0.6). Higher = more diverse/random.
    --top_p FLOAT       Nucleus sampling probability (default: 0.98).

"""
import os, sys

# Environment setup
os.environ["HF_HOME"] = "/workspace/hf"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf/transformers"
os.environ["HF_DATASETS_CACHE"] = "/workspace/hf/datasets"

sys.path.insert(0, "/workspace/alpamayo/src")
sys.path.insert(0, "/workspace/alpamayo/scripts")

print("="*60, flush=True)
print("ALPAMAYO DEBUG VISUALIZATON", flush=True)
print("="*60, flush=True)

import argparse
parser = argparse.ArgumentParser(description="Debug Alpamayo inference")
parser.add_argument("--bag", default="/workspace/alpamayo/rosbag2_autoware/rosbag2_autoware_0.mcap")
parser.add_argument("--map", default="/workspace/alpamayo/lanelet2_map.osm")
parser.add_argument("--output", default="/workspace/alpamayo/debug_viz.png")
parser.add_argument("--ratio", type=float, default=0.6, help="Time ratio (0.0-1.0) in bag")
parser.add_argument("--prompt", type=str, default=None, help="Custom prompt to append/replace")
parser.add_argument("--multicam", action="store_true", help="Simulate 3-camera setup (Black, Front, Black)")

# New debug arguments
parser.add_argument("--num_samples", type=int, default=5, help="Number of trajectories to sample")
parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
parser.add_argument("--top_p", type=float, default=0.98, help="Top-p sampling")

args = parser.parse_args()

print(f"\nConfig:", flush=True)
print(f"  Bag: {args.bag}", flush=True)
print(f"  Ratio: {args.ratio}", flush=True)
print(f"  Samples: {args.num_samples}", flush=True)
print(f"  Temp: {args.temperature}", flush=True)

print("\n[1/7] Importing libraries...", flush=True)
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
import convert_rosbag
from mcap_ros2.reader import read_ros2_messages
import xml.etree.ElementTree as ET
print("✓ Imports complete", flush=True)

print("\n[2/7] Scanning bag for time range...", flush=True)
times = [msg.log_time_ns for msg in read_ros2_messages(args.bag, topics=["/localization/kinematic_state"])]
t0 = int(min(times) + (max(times) - min(times)) * args.ratio)
print(f"✓ Calculated t0 = {t0} ns", flush=True)

print("\n[3/7] Processing bag and extracting data...", flush=True)
temp_file = "/workspace/alpamayo/temp_debug_data.pt"
convert_rosbag.process_bag(args.bag, temp_file, t0)
data = torch.load(temp_file, weights_only=False)
print(f"✓ Data loaded", flush=True)

print("\n[4/7] Loading Alpamayo-R1 model...", flush=True)
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", torch_dtype=torch.bfloat16)
model = model.eval().cuda()
print("✓ Model loaded", flush=True)

print("\n[5/7] Running inference...", flush=True)
if data.get("tokenized_data"):
    model_inputs_tokenized = data["tokenized_data"].to("cuda")
else:
    processor = helper.get_processor(model.tokenizer)
    
    # [Modify] Handle Multicam Simulation
    if args.multicam:
        print("  > Simulating Multi-Camera (Left=Black, Front=Valid, Right=Black)...")
        # data["image_frames"] is (1, T, 3, H, W)
        front_frames = data["image_frames"][0] # (T, 3, H, W)
        black_frames = torch.zeros_like(front_frames)
        
        # Stack to (T, 3, 3, H, W) -> [Left, Front, Right] per timestamp
        multicam_frames = torch.stack([black_frames, front_frames, black_frames], dim=1) 
        # Flatten: (T*3, 3, H, W) -> [L_T0, F_T0, R_T0, L_T1...]
        flat_frames = multicam_frames.flatten(0, 1)
        print(f"    New shape: {flat_frames.shape}")
        
    else:
        flat_frames = data["image_frames"].flatten(0, 1)

    messages = helper.create_message(flat_frames)
    
    # Custom interaction: Modify the last user message if prompt is provided
    if args.prompt:
        print(f"  > Injecting custom prompt: '{args.prompt}'")
        # Structure: messages[1]['content'][-1]['text'] is the text prompt
        # Standard text: "...output the chain-of-thought preventing..."
        # We will append the custom prompt before the final instruction
        current_text = messages[1]['content'][-1]['text']
        # Insert before "output the chain..."
        # Actually helper.py says: {hist}output the chain...
        messages[1]['content'][-1]['text'] = f"{messages[1]['content'][-1]['text']} {args.prompt}"
        
    model_inputs_tokenized = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt"
    ).to("cuda")

model_inputs = {
    "tokenized_data": model_inputs_tokenized,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}
model_inputs = helper.to_device(model_inputs, "cuda")

# Run inference with DEBUG parameters
print(f"  > Sampling {args.num_samples} trajectories (temp={args.temperature})...")

with torch.autocast("cuda", dtype=torch.bfloat16):
    # Note: alpamayo_r1.py returns (pred_xyz, pred_rot, extra) when return_extra=True
    # pred_xyz shape: [B, num_traj_sets, num_traj_samples, T, 3]
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=args.top_p,
        temperature=args.temperature,
        num_traj_samples=args.num_samples, # Pass the debug number of samples
        max_generation_length=256,
        return_extra=True,
    )

cot_text = extra["cot"][0][0][0] if extra.get("cot") is not None and extra["cot"].size > 0 else "No CoT generated"
# If we have multiple samples, the CoT might vary if sampled multiples times (num_traj_sets > 1).
# By default num_traj_sets=1, but the VLM generates num_traj_samples sequences?
# Looking at code: num_traj_samples is passed to VLM generation config `num_return_sequences`.
# So we effectively get 'num_samples' different reasoning paths AND trajectories?
# The code: extra["cot"] shape is reshaped to [B, ns, nj]. So we do have multiple CoTs!

print(f"✓ Inference complete. CoT preview: {cot_text[:60]}...", flush=True)

print("\n[6/7] Creating visualization...", flush=True)
fig = plt.figure(figsize=(24, 12))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1]) 
# Layout: 
# [Image] [CoT Text]
# [Map (span all)]

# Top-left: Camera image
ax_img = fig.add_subplot(gs[0, 0])
last_frame = data["image_frames"][0, -1].permute(1, 2, 0).numpy()
ax_img.imshow(last_frame)
ax_img.set_title(f"Camera Input (t0={t0})", fontsize=14)
ax_img.axis("off")

# Top-right: Chain of Causation
ax_cot = fig.add_subplot(gs[0, 1:])
ax_cot.text(0.02, 0.95, f"Chain-of-Causation (Sample 0):\n\n{cot_text}", 
            fontsize=10, verticalalignment='top', family='monospace', wrap=True)
ax_cot.axis("off")

# Bottom: Map with trajectories
ax_map = fig.add_subplot(gs[1, :])

# Parse and plot Lanelet2 map
if os.path.exists(args.map):
    tree = ET.parse(args.map)
    root = tree.getroot()
    nodes = {}
    for node in root.findall('node'):
        node_id = node.get('id')
        tags = {tag.get('k'): tag.get('v') for tag in node.findall('tag')}
        if 'local_x' in tags and 'local_y' in tags:
            nodes[node_id] = (float(tags['local_x']), float(tags['local_y']))
    
    for way in root.findall('way'):
        node_refs = [nd.get('ref') for nd in way.findall('nd')]
        coords = [nodes[ref] for ref in node_refs if ref in nodes]
        if len(coords) > 1:
            way_array = np.array(coords)
            ax_map.plot(way_array[:, 0], way_array[:, 1], 'k-', linewidth=0.5, alpha=0.3)

# Plot trajectories
if "t0_pos" in data and "t0_Rot" in data:
    t0_pos = data["t0_pos"]
    t0_Rot = data["t0_Rot"]
    
    def transform_to_global(local_coords):
        if local_coords.shape[-1] == 2:
            local_coords = np.hstack([local_coords, np.zeros((local_coords.shape[0], 1))])
        global_coords = t0_pos + (t0_Rot @ local_coords.T).T
        return global_coords[:, :2]
    
    # Plot ego position
    ax_map.plot(t0_pos[0], t0_pos[1], 'ro', markersize=12, label="Ego (t0)", zorder=10)
    
    # Plot ego history
    if "ego_history_xyz" in data:
        ego_hist_local = data["ego_history_xyz"][0, 0].numpy()
        ego_hist_global = transform_to_global(ego_hist_local)
        ax_map.plot(ego_hist_global[:, 0], ego_hist_global[:, 1], 
                   'g-', linewidth=2, alpha=0.7, label="History", zorder=9)
    
    # Plot ALL predicted trajectories
    # pred_xyz shape: [B, 1, num_samples, T, 3]
    # We want to iterate over num_samples
    print(f"  Plotting {args.num_samples} predictions...", flush=True)
    
    predictions_local = pred_xyz.cpu().numpy()[0, 0] # Shape: [num_samples, T, 3]
    
    # Calculate statistics
    print("\n  Trajectory Statistics (Local Frame):", flush=True)
    print("  ID | Max Lat Dev (m) | Final Y (m) | Curvature Score", flush=True)
    print("  ---|-----------------|-------------|----------------", flush=True)
    
    stats_data = []

    # Use a colormap
    cmap = matplotlib.colormaps['viridis']
    
    for i in range(args.num_samples):
        pred_local = predictions_local[i, :, :3]
        pred_global = transform_to_global(pred_local)
        
        # Calculate deviation from straight line (start to end)
        start_pt = pred_local[0, :2]
        end_pt = pred_local[-1, :2]
        
        # Simple lateral deviation: max distance from the line connecting start and end
        # Or just max absolute Y if we assume local frame starts at 0,0 and faces X.
        # Let's use max absolute Y value as a proxy for "turning" if the car starts facing X.
        # Usually local frame: x is forward, y is left.
        max_lat_dev = np.max(np.abs(pred_local[:, 1]))
        final_y = pred_local[-1, 1]
        
        # Curvature score roughly: sum of absolute heading changes? 
        # Or just use max_lat_dev for now.
        
        print(f"  {i:2d} | {max_lat_dev:13.3f}   | {final_y:11.3f} | N/A", flush=True)
        stats_data.append(max_lat_dev)
        
        # Color based on index to differentiate
        color = cmap(i / max(args.num_samples, 1))
        
        label = "Predicted" if i == 0 else "_nolegend_"
        alpha = 0.8 if args.num_samples == 1 else 0.4
        width = 3 if args.num_samples == 1 else 1.5
        
        ax_map.plot(pred_global[:, 0], pred_global[:, 1], 
                   color='blue', linewidth=width, alpha=alpha, label=label, zorder=5)

    avg_dev = np.mean(stats_data)
    max_dev_all = np.max(stats_data)
    print(f"\n  [Summary] Avg Max Lat Dev: {avg_dev:.3f} m, Max of all: {max_dev_all:.3f} m", flush=True)

    ax_map.legend(loc='upper right', fontsize=12)
    ax_map.set_aspect('equal')
    ax_map.set_title(f"Global Trajectories (N={args.num_samples}, T={args.temperature})", fontsize=14)
    ax_map.set_xlim(t0_pos[0] - 60, t0_pos[0] + 60)
    ax_map.set_ylim(t0_pos[1] - 60, t0_pos[1] + 60)
    ax_map.grid(True, alpha=0.3)

print("✓ Visualization created", flush=True)
plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"✓ Saved to: {args.output}", flush=True)

if os.path.exists(temp_file):
    os.remove(temp_file)

print("\n" + "="*60, flush=True)
print("SUCCESS! Debug visualization complete.", flush=True)
print("="*60, flush=True)
