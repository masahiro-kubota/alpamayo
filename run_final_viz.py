#!/usr/bin/env python
"""
Alpamayo Rosbag to Lanelet2 Visualization Script

DESCRIPTION:
    This script performs end-to-end inference using the Alpamayo-R1 autonomous driving model
    on ROS 2 bag data and visualizes the results on a Lanelet2 map. It produces a comprehensive
    visualization showing:
    - Camera input image
    - Chain-of-Causation (CoT) reasoning from the model
    - Lanelet2 road map with ego vehicle history and predicted trajectory in global coordinates

USAGE:
    Basic usage (uses default paths and ratio=0.6):
        python run_final_viz.py

    Specify custom bag and output:
        python run_final_viz.py --bag /path/to/data.mcap --output result.png

    Use specific time in bag (0.0=start, 1.0=end):
        python run_final_viz.py --ratio 0.5

    Full example:
        python run_final_viz.py \\
            --bag rosbag2_autoware/rosbag2_autoware_0.mcap \\
            --map lanelet2_map.osm \\
            --output my_visualization.png \\
            --ratio 0.6

ARGUMENTS:
    --bag PATH      Path to input ROS 2 bag (.mcap file)
                    Default: /workspace/alpamayo/rosbag2_autoware/rosbag2_autoware_0.mcap
    
    --map PATH      Path to Lanelet2 map (.osm file)
                    Default: /workspace/alpamayo/lanelet2_map.osm
    
    --output PATH   Output image path (.png)
                    Default: /workspace/alpamayo/visualization_output.png
    
    --ratio FLOAT   Time position in bag as ratio (0.0 to 1.0)
                    0.0 = start of bag, 1.0 = end of bag
                    Default: 0.6

OUTPUT:
    The script generates a single PNG image with 3 panels:
    1. Top-left: Camera image at selected timestamp
    2. Top-right: Chain-of-Causation text from model
    3. Bottom: Lanelet2 map showing:
       - Black lines: Road/lane boundaries
       - Red dot: Ego vehicle position at t0
       - Green line: Ego vehicle history (past trajectory, model input)
       - Blue line: Predicted future trajectory (model output)

REQUIREMENTS:
    - Environment: ar1_venv with all dependencies installed
    - GPU: CUDA-capable GPU for model inference
    - Data: ROS 2 bag with /localization/kinematic_state and camera topics
    - Model: nvidia/Alpamayo-R1-10B (auto-downloaded to HF_HOME)

ENVIRONMENT VARIABLES:
    Set these before running:
        export HF_HOME=/workspace/hf
        export TRANSFORMERS_CACHE=/workspace/hf/transformers
        export HF_DATASETS_CACHE=/workspace/hf/datasets

EXAMPLE SESSION:
    cd /workspace/alpamayo
    source ar1_venv/bin/activate
    python run_final_viz.py --ratio 0.6
    # Output: visualization_output.png

AUTHOR:
    Created for Alpamayo-R1 rosbag inference workflow
"""
import os, sys

# Environment setup
os.environ["HF_HOME"] = "/workspace/hf"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf/transformers"
os.environ["HF_DATASETS_CACHE"] = "/workspace/hf/datasets"

sys.path.insert(0, "/workspace/alpamayo/src")
sys.path.insert(0, "/workspace/alpamayo/scripts")

print("="*60, flush=True)
print("ALPAMAYO ROSBAG VISUALIZATION", flush=True)
print("="*60, flush=True)

import argparse
parser = argparse.ArgumentParser(description="Visualize Alpamayo inference on Rosbag data")
parser.add_argument("--bag", default="/workspace/alpamayo/rosbag2_autoware/rosbag2_autoware_0.mcap")
parser.add_argument("--map", default="/workspace/alpamayo/lanelet2_map.osm")
parser.add_argument("--output", default="/workspace/alpamayo/visualization_output.png")
parser.add_argument("--ratio", type=float, default=0.6, help="Time ratio (0.0-1.0) in bag")
args = parser.parse_args()

print(f"\nConfig:", flush=True)
print(f"  Bag: {args.bag}", flush=True)
print(f"  Map: {args.map}", flush=True)
print(f"  Output: {args.output}", flush=True)
print(f"  Ratio: {args.ratio}", flush=True)

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
temp_file = "/workspace/alpamayo/temp_inference_data.pt"
convert_rosbag.process_bag(args.bag, temp_file, t0)
data = torch.load(temp_file, weights_only=False)
print(f"✓ Data loaded, keys: {list(data.keys())}", flush=True)

print("\n[4/7] Loading Alpamayo-R1 model...", flush=True)
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", torch_dtype=torch.bfloat16)
model = model.eval().cuda()
print("✓ Model loaded and ready", flush=True)

print("\n[5/7] Running inference...", flush=True)
if data.get("tokenized_data"):
    model_inputs_tokenized = data["tokenized_data"].to("cuda")
    print("  Using pre-tokenized data", flush=True)
else:
    print("  Tokenizing with model tokenizer...", flush=True)
    processor = helper.get_processor(model.tokenizer)
    flat_frames = data["image_frames"].flatten(0, 1)
    messages = helper.create_message(flat_frames)
    model_inputs_tokenized = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt"
    ).to("cuda")

# Prepare model inputs with ego history
model_inputs = {
    "tokenized_data": model_inputs_tokenized,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}
model_inputs = helper.to_device(model_inputs, "cuda")

# Run inference
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,
        max_generation_length=256,
        return_extra=True,
    )

cot_text = extra["cot"][0][0] if extra.get("cot") and len(extra["cot"]) > 0 and len(extra["cot"][0]) > 0 else "No CoT generated"
print(f"✓ Inference complete. CoT preview: {cot_text[:60]}...", flush=True)

print("\n[6/7] Creating visualization...", flush=True)
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

# Top-left: Camera image
ax_img = fig.add_subplot(gs[0, 0])
last_frame = data["image_frames"][0, -1].permute(1, 2, 0).numpy()
ax_img.imshow(last_frame)
ax_img.set_title(f"Camera Input (t0={t0})", fontsize=14)
ax_img.axis("off")

# Top-right: Chain of Causation
ax_cot = fig.add_subplot(gs[0, 1])
ax_cot.text(0.05, 0.95, f"Chain-of-Causation:\n\n{cot_text}", 
            fontsize=11, verticalalignment='top', family='monospace', wrap=True)
ax_cot.axis("off")

# Bottom: Map with trajectories
ax_map = fig.add_subplot(gs[1, :])

# Parse and plot Lanelet2 map
if os.path.exists(args.map):
    print("  Parsing Lanelet2 map...", flush=True)
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
    print(f"  Plotted {len(nodes)} nodes and map ways", flush=True)

# Plot trajectories if global pose available
if "t0_pos" in data and "t0_Rot" in data:
    print("  Plotting trajectories...", flush=True)
    t0_pos = data["t0_pos"]
    t0_Rot = data["t0_Rot"]
    
    def transform_to_global(local_coords):
        """Transform from local ego frame to global map frame"""
        if local_coords.shape[-1] == 2:
            local_coords = np.hstack([local_coords, np.zeros((local_coords.shape[0], 1))])
        global_coords = t0_pos + (t0_Rot @ local_coords.T).T
        return global_coords[:, :2]
    
    # Plot ego position
    ax_map.plot(t0_pos[0], t0_pos[1], 'ro', markersize=12, label="Ego Vehicle (t0)", zorder=5)
    
    # Plot ego history (input to model)
    if "ego_history_xyz" in data:
        ego_hist_local = data["ego_history_xyz"][0, 0].numpy()
        ego_hist_global = transform_to_global(ego_hist_local)
        ax_map.plot(ego_hist_global[:, 0], ego_hist_global[:, 1], 
                   'g-', linewidth=2, alpha=0.7, label="Ego History (input)", zorder=4)
        print("  ✓ Ego history plotted", flush=True)
    
    # Plot predicted trajectory (model output)
    pred_local = pred_xyz.cpu().numpy()[0, 0, 0, :, :3]  # First sample
    pred_global = transform_to_global(pred_local)
    ax_map.plot(pred_global[:, 0], pred_global[:, 1], 
               'b-', linewidth=3, label="Predicted Trajectory", zorder=4)
    print("  ✓ Prediction plotted", flush=True)
    
    ax_map.legend(loc='upper right', fontsize=12)
    ax_map.set_aspect('equal')
    ax_map.set_title("Global Trajectories on Lanelet2 Map", fontsize=14)
    ax_map.set_xlim(t0_pos[0] - 60, t0_pos[0] + 60)
    ax_map.set_ylim(t0_pos[1] - 60, t0_pos[1] + 60)
    ax_map.grid(True, alpha=0.3)
else:
    ax_map.text(0.5, 0.5, "Global pose not available in data", 
                ha='center', va='center', fontsize=14)
    ax_map.axis('off')

print("✓ Visualization created", flush=True)

print("\n[7/7] Saving output...", flush=True)
plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"✓ Saved to: {args.output}", flush=True)

# Cleanup
if os.path.exists(temp_file):
    os.remove(temp_file)
    print("✓ Cleaned up temporary files", flush=True)

print("\n" + "="*60, flush=True)
print("SUCCESS! Visualization complete.", flush=True)
print("="*60, flush=True)