
import argparse
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from transformers import AutoConfig

def load_data(pt_file):
    data = torch.load(pt_file)
    if not isinstance(data, list):
        data = [data]
    return data

def evaluate(args):
    # 1. Config & Model
    print("Loading Model...")
    model = AlpamayoR1.from_pretrained(args.base_model, trust_remote_code=True)
    device = "cuda"
    
    model.to(device)
    model.eval()

    # 2. Load Checkpoint (Fine-tuned Head)
    if args.ckpt_path:
        print(f"Loading checkpoint from {args.ckpt_path}...")
        ckpt = torch.load(args.ckpt_path, map_location=device)
        model.expert.load_state_dict(ckpt['expert'])
        model.action_in_proj.load_state_dict(ckpt['action_in_proj'])
        model.action_out_proj.load_state_dict(ckpt['action_out_proj'])
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint provided. Running with base model weights.")

    # 3. Load Data
    print(f"Loading Validation Data from {args.data_path}...")
    dataset = load_data(args.data_path)
    
    # 4. Inference & Visualization
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select a few samples
    indices = range(min(len(dataset), args.num_samples))
    
    for i in indices:
        item = dataset[i]
        clip_id = item.get("clip_id", f"sample_{i}")
        print(f"Evaluating Sample {i} (Clip: {clip_id})...")
        
        # Move tensors to device
        # item is a dict, need to handle nesting for sample_trajectories...
        # The method expects 'ego_history_xyz', 'tokenized_data' directly in dict
        
        # Deep copy to avoid in-place modification issues if retrying
        sample_data = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                sample_data[k] = v.to(device)
                # Expand batch dim if needed (prepare_training_data saves as batch 1)
                # If saved as (1, ...), it's fine.
            elif isinstance(v, dict):
                # Handle tokenized_data dict
                sample_data[k] = {sk: sv.to(device) if isinstance(sv, torch.Tensor) else sv for sk, sv in v.items()}
            else:
                sample_data[k] = v
                
        # Run Inference
        with torch.no_grad():
            # sample_trajectories... handles everything
            # Returns: pred_xyz, pred_rot
            # pred_xyz: (B, num_traj_sets, num_traj_samples, T, 3)
            pred_xyz, pred_rot = model.sample_trajectories_from_data_with_vlm_rollout(
                sample_data,
                num_traj_samples=args.num_preds,
                num_traj_sets=1,
                top_p=0.9,
                temperature=0.8
            )
            
        # Ground Truth
        gt_xyz = sample_data["ego_future_xyz"].cpu().numpy() # (B, 1, T, 3)
        gt_x = gt_xyz[0, 0, :, 0]
        gt_y = gt_xyz[0, 0, :, 1]
        
        # Predictions
        pred_xyz_np = pred_xyz.cpu().numpy() # (1, 1, N, T, 3)
        
        # Plot
        plt.figure(figsize=(10, 10))
        plt.plot(gt_x, gt_y, 'g-', linewidth=3, label='Ground Truth')
        
        # Plot predictions
        for n in range(args.num_preds):
            pred_x = pred_xyz_np[0, 0, n, :, 0]
            pred_y = pred_xyz_np[0, 0, n, :, 1]
            plt.plot(pred_x, pred_y, 'b--', alpha=0.5, label='Prediction' if n==0 else "")
            
        plt.title(f"Trajectory Prediction - Clip {clip_id}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        save_path = os.path.join(args.output_dir, f"eval_{clip_id}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to validation .pt file")
    parser.add_argument("--ckpt_path", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--base_model", type=str, default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_preds", type=int, default=3)
    args = parser.parse_args()
    
    evaluate(args)
