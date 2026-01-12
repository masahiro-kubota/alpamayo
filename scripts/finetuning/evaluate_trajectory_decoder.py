
import argparse
import sys
import os
import torch
import json
import numpy as np
from pathlib import Path
from typing import Any

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

import time

def load_data(pt_file):
    """Load preprocessed validation data."""
    try:
        data = torch.load(pt_file, weights_only=False)
    except TypeError:
        data = torch.load(pt_file)
    if not isinstance(data, list):
        data = [data]
    return data

def evaluate(args):
    # 1. Config & Model
    print("Loading Model...", flush=True)
    model = AlpamayoR1.from_pretrained(args.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    device = "cuda"
    model.to(device)
    model.eval()
    
    # Load Checkpoint weights if provided
    if args.ckpt_path and args.ckpt_path != "None":
        print(f"Loading checkpoint from {args.ckpt_path}...", flush=True)
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        if 'expert' in checkpoint:
            model.expert.load_state_dict(checkpoint['expert'])
            model.action_in_proj.load_state_dict(checkpoint['action_in_proj'])
            model.action_out_proj.load_state_dict(checkpoint['action_out_proj'])
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Checkpoint loaded successfully.", flush=True)
    elif args.ckpt_path == "None":
        print("Using original model weights (None specified).", flush=True)

    # 2. Load Data
    print(f"Loading Validation Data from {args.data_path}...", flush=True)
    dataset = load_data(args.data_path)
    
    num_to_eval = min(len(dataset), args.num_samples)
    indices = range(num_to_eval)
    
    results = []

    for i in indices:
        item = dataset[i]
        clip_id = item.get("clip_id", f"sample_{i}")
        print(f"[{i+1}/{num_to_eval}] Evaluating Clip: {clip_id}...", flush=True)
        
        # Prepare Sample Data
        sample_data = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                sample_data[k] = v.to(device)
            elif isinstance(v, dict):
                sample_data[k] = {}
                for sk, sv in v.items():
                    if isinstance(sv, torch.Tensor):
                        tensor = sv.to(device)
                        if sk in ['input_ids', 'attention_mask'] and tensor.dim() == 1:
                            tensor = tensor.unsqueeze(0)
                        elif sk == 'image_grid_thw' and tensor.dim() == 1:
                            tensor = tensor.unsqueeze(0)
                        sample_data[k][sk] = tensor
                    else:
                        sample_data[k][sk] = sv
            else:
                sample_data[k] = v

        start_time = time.time()
        with torch.no_grad():
            # Match official Nvidia parameters
            pred_xyz, pred_rot = model.sample_trajectories_from_data_with_vlm_rollout(
                sample_data,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=args.num_preds,
                max_generation_length=256
            )
        duration = time.time() - start_time
        
        # --- COORDINATE TRANSFORMATION ---
        # Transform GT and History into the current ego-local frame (centered at 0,0, facing forward)
        # to match the model's prediction space.
        
        # Get current pose (last step of history)
        curr_xyz = sample_data["ego_history_xyz"][0, 0, -1:]   # (1, 3)
        curr_rot = sample_data["ego_history_rot"][0, 0, -1]     # (3, 3) - ego to world
        
        # World to Local transformation: p_local = R.T @ (p_world - t)
        # In matrix form with (T, 3) points: P_local = (P_world - T) @ R
        gt_future_world = sample_data["ego_future_xyz"][0, 0] # (Tf, 3)
        history_world = sample_data["ego_history_xyz"][0, 0]   # (Th, 3)
        
        gt_future_local = (gt_future_world - curr_xyz) @ curr_rot
        history_local = (history_world - curr_xyz) @ curr_rot
        
        # Serialize result
        result_item = {
            "clip_id": clip_id,
            "duration_sec": duration,
            "gt_future": gt_future_local.cpu().numpy().tolist(),
            "history": history_local.cpu().numpy().tolist(),
            "predictions": pred_xyz.cpu().numpy()[0, 0].tolist() # (num_preds, T, 3)
        }
        results.append(result_item)
        print(f"  -> Done in {duration:.2f} seconds.", flush=True)

    # Save to JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True) if os.path.dirname(args.output_json) else None
    # Create Output Structure with Metadata
    output_data = {
        "metadata": {
            "model_id": args.base_model,
            "ckpt_path": str(args.ckpt_path),
            "data_path": args.data_path,
            "num_samples": num_to_eval,
            "parameters": {
                "temperature": 0.6,
                "top_p": 0.98,
                "max_generation_length": 256
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "results": results
    }

    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nFinal results with metadata saved to {args.output_json}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to validation .pt data")
    parser.add_argument("--base_model", type=str, default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Optional trained head checkpoint")
    parser.add_argument("--output_json", type=str, default="eval_results.json")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_preds", type=int, default=3)
    args = parser.parse_args()
    
    evaluate(args)
