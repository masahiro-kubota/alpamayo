#!/usr/bin/env python3
"""
Test Alpamayo-R1 on High-Curvature Clips

Runs inference on discovered high-curvature clips to verify model behavior
on sharp curves.
"""

import torch
import numpy as np
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

# Top 3 highest curvature clips from dataset scan
HIGH_CURVATURE_CLIPS = [
    ("f789b390-1698-4f99-b237-6de4cbbb7666", "Radius ~3.6m, Curvature 0.277"),
    ("34b09f15-9b59-4baf-ac4d-a51fdad7a85b", "Radius ~3.6m, Curvature 0.277"),
    ("2ae4608f-2405-472e-81f1-2788b3912ac1", "Radius ~3.6m, Curvature 0.277"),
]

print("="*80)
print("Testing Alpamayo-R1 on High-Curvature Clips")
print("="*80)

# Load model once
print("\nLoading Alpamayo-R1 model...")
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)
print("✓ Model loaded\n")

results = []

for i, (clip_id, description) in enumerate(HIGH_CURVATURE_CLIPS, 1):
    print(f"\n{'='*80}")
    print(f"Test {i}/3: {description}")
    print(f"Clip ID: {clip_id}")
    print(f"{'='*80}\n")
    
    try:
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
                num_traj_samples=1,  # Reduced to 1 to avoid OOM
                max_generation_length=256,
                return_extra=True,
            )
        
        # Compute metrics
        gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
        pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        
        # Compute lateral deviation (Y axis in local frame)
        pred_lateral_devs = pred_xy[:, 1, :]  # Shape: (num_samples, num_steps)
        max_lateral_devs = np.abs(pred_lateral_devs).max(axis=1)
        
        # Compute minADE
        diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
        min_ade = diff.min()
        
        # Store results
        result = {
            'clip_id': clip_id,
            'description': description,
            'cot': extra["cot"][0][0],  # First sample's CoT
            'min_ade': float(min_ade),
            'max_lateral_dev_per_sample': max_lateral_devs.tolist(),
            'avg_max_lateral_dev': float(max_lateral_devs.mean()),
        }
        results.append(result)
        
        # Print result
        print(f"\n{'─'*80}")
        print(f"RESULTS:")
        print(f"{'─'*80}")
        print(f"CoT: {result['cot']}")
        print(f"minADE: {result['min_ade']:.3f} meters")
        print(f"\nLateral Deviation by Sample:")
        for j, dev in enumerate(result['max_lateral_dev_per_sample'], 1):
            print(f"  Sample {j}: {dev:.3f}m")
        print(f"Average Max Lateral Deviation: {result['avg_max_lateral_dev']:.3f}m")
        
        #判定
        if result['avg_max_lateral_dev'] > 2.0:
            print(f"\n✓ Model shows CURVE prediction (avg deviation > 2m)")
        else:
            print(f"\n✗ Model shows STRAIGHT bias (avg deviation < 2m)")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        results.append({
            'clip_id': clip_id,
            'description': description,
            'error': str(e)
        })

# Summary
print(f"\n\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}\n")

successful = [r for r in results if 'error' not in r]
if successful:
    avg_ade = np.mean([r['min_ade'] for r in successful])
    avg_lateral_dev = np.mean([r['avg_max_lateral_dev'] for r in successful])
    
    print(f"Successfully tested: {len(successful)}/3 clips")
    print(f"Average minADE: {avg_ade:.3f}m")
    print(f"Average Lateral Deviation: {avg_lateral_dev:.3f}m")
    
    if avg_lateral_dev > 2.0:
        print(f"\n✓ Model CAN predict curves on official high-curvature data")
    else:
        print(f"\n✗ Model shows STRAIGHT BIAS even on extreme curves (radius 3.6m)")
        print(f"   → This confirms the bias is a MODEL LIMITATION, not data/script issue")

# Save results
import json
with open('trajectory_bias_experiment/high_curve_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed results saved to: trajectory_bias_experiment/high_curve_test_results.json")
