#!/usr/bin/env python3
"""
Test Alpamayo-R1 on a Single High-Curvature Clip
More memory-efficient: loads model, tests one clip, exits
"""

import torch
import numpy as np
import sys
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

# Get clip ID from command line
if len(sys.argv) < 2:
    print("Usage: python test_single_curve.py <clip_id>")
    sys.exit(1)

clip_id = sys.argv[1]

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

# Prepare input
messages = helper.create_message(data["image_frames"].flatten(0, 1))
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)
model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}
model_inputs = helper.to_device(model_inputs, "cuda")

# Run inference
print("Running inference...")
torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,
        max_generation_length=256,
        return_extra=True,
    )

# Compute metrics
gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)

# Compute lateral deviation (Y axis in local frame)
pred_lateral_devs = pred_xy[:, 1, :]  # Shape: (num_samples, num_steps)
max_lateral_dev = np.abs(pred_lateral_devs).max()

# Compute minADE
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
min_ade = diff.min()

# Print results
print(f"\n{'='*80}")
print(f"RESULTS")
print(f"{'='*80}")
print(f"Clip ID: {clip_id}")
print(f"CoT: {extra['cot'][0][0]}")
print(f"minADE: {min_ade:.3f} meters")
print(f"Max Lateral Deviation: {max_lateral_dev:.3f}m")

if max_lateral_dev > 2.0:
    print(f"\n✓ Model shows CURVE prediction (deviation > 2m)")
else:
    print(f"\n✗ Model shows STRAIGHT bias (deviation < 2m)")

print(f"\n{'='*80}\n")

# Save result
import json
result = {
    'clip_id': clip_id,
    'cot': extra["cot"][0][0],
    'min_ade': float(min_ade),
    'max_lateral_dev': float(max_lateral_dev),
}
print(json.dumps(result, indent=2))
