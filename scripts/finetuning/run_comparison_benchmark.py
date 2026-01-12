
import argparse
import sys
import os
import torch
import json
import time
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from scripts.finetuning.evaluate_trajectory_decoder import load_data
from scripts.finetuning.visualize_eval_results import visualize

def run_inference(model, dataset, device, num_samples, num_preds, desc):
    """Run inference for a specific dataset and return results."""
    print(f"\n>>> Running Inference: {desc}", flush=True)
    num_to_eval = min(len(dataset), num_samples)
    results = []

    for i in range(num_to_eval):
        item = dataset[i]
        clip_id = item.get("clip_id", f"sample_{i}")
        print(f"  [{i+1}/{num_to_eval}] Clip: {clip_id}...", flush=True)
        
        # Prepare Sample Data
        sample_data = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                sample_data[k] = v.to(device)
            elif isinstance(v, dict):
                sample_data[k] = {sk: sv.to(device) if isinstance(sv, torch.Tensor) else sv for sk, sv in v.items()}
                # Ensure batch dimensions for model
                for sk in ['input_ids', 'attention_mask', 'image_grid_thw']:
                    if sk in sample_data[k] and sample_data[k][sk].dim() == 1:
                        sample_data[k][sk] = sample_data[k][sk].unsqueeze(0)
            else:
                sample_data[k] = v

        start_time = time.time()
        with torch.no_grad():
            pred_xyz, _ = model.sample_trajectories_from_data_with_vlm_rollout(
                sample_data,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=num_preds,
                max_generation_length=256
            )
        duration = time.time() - start_time
        
        # Coordinate Transformation (Ego-local)
        curr_xyz = sample_data["ego_history_xyz"][0, 0, -1:] 
        curr_rot = sample_data["ego_history_rot"][0, 0, -1]
        gt_future_local = (sample_data["ego_future_xyz"][0, 0] - curr_xyz) @ curr_rot
        history_local = (sample_data["ego_history_xyz"][0, 0] - curr_xyz) @ curr_rot
        
        results.append({
            "clip_id": clip_id,
            "duration_sec": duration,
            "gt_future": gt_future_local.cpu().numpy().tolist(),
            "history": history_local.cpu().numpy().tolist(),
            "predictions": pred_xyz.cpu().numpy()[0, 0].tolist()
        })
        print(f"    Done in {duration:.2f}s", flush=True)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masked_data", type=str, required=True, help="PT file with masked Left/Right")
    parser.add_argument("--full_data", type=str, required=True, help="PT file with full 4 cameras")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint to evaluate")
    parser.add_argument("--output_dir", type=str, default="trajectory_bias_experiment")
    parser.add_argument("--model_id", type=str, default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--force_baseline", action="store_true", help="Redo baseline inference even if JSON exists")
    args = parser.parse_args()

    device = "cuda"
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

def is_cached(json_path, data_path, ckpt_path, num_samples):
    """Check if existing JSON matches the requested criteria."""
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            meta = data.get("metadata", {})
            # Verify critical factors
            if meta.get("data_path") != data_path: return False
            if meta.get("ckpt_path") != str(ckpt_path): return False
            if meta.get("num_samples", 0) < num_samples: return False
            return True
    except:
        return False

def save_with_metadata(json_path, results, model_id, data_path, ckpt_path, num_samples):
    """Save results with comprehensive metadata."""
    output_data = {
        "metadata": {
            "model_id": model_id,
            "ckpt_path": str(ckpt_path),
            "data_path": data_path,
            "num_samples": num_samples,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "results": results
    }
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masked_data", type=str, required=True, help="PT file with masked Left/Right")
    parser.add_argument("--full_data", type=str, required=True, help="PT file with full 4 cameras")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint to evaluate")
    parser.add_argument("--output_dir", type=str, default="trajectory_bias_experiment")
    parser.add_argument("--model_id", type=str, default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Redo all inferences")
    args = parser.parse_args()

    device = "cuda"
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    json_baseline_full = os.path.join(args.output_dir, "logs", "eval_baseline_full.json")
    json_baseline_masked = os.path.join(args.output_dir, "logs", "eval_baseline_masked.json")
    ckpt_name = Path(args.ckpt_path).stem
    json_ft_masked = os.path.join(args.output_dir, "logs", f"eval_ft_{ckpt_name}.json")
    
    # Advanced Cache Check
    need_full = args.force or not is_cached(json_baseline_full, args.full_data, "None", args.num_samples)
    need_masked = args.force or not is_cached(json_baseline_masked, args.masked_data, "None", args.num_samples)
    need_ft = args.force or not is_cached(json_ft_masked, args.masked_data, args.ckpt_path, args.num_samples)

    if not (need_full or need_masked or need_ft):
        print("\n>>> All results are already cached and match criteria. Visualizing current results...", flush=True)
        # Skip everything and visualize
    else:
        # 1. Loading Model
        print("Step 1: Loading Model to GPU...", flush=True)
        model = AlpamayoR1.from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model.to(device)
        model.eval()

        # 2. Base Inferences
        if need_full or need_masked:
            print("Step 2: Running Baseline Inferences...", flush=True)
            if need_full:
                ds = load_data(args.full_data)
                res = run_inference(model, ds, device, args.num_samples, 3, "Baseline (Full)")
                save_with_metadata(json_baseline_full, res, args.model_id, args.full_data, "None", args.num_samples)
            if need_masked:
                ds = load_data(args.masked_data)
                res = run_inference(model, ds, device, args.num_samples, 3, "Baseline (Masked)")
                save_with_metadata(json_baseline_masked, res, args.model_id, args.masked_data, "None", args.num_samples)

        # 3. FT Inference
        if need_ft:
            if args.ckpt_path != "None":
                print(f"\nStep 3: Loading FT Checkpoint from {args.ckpt_path}...", flush=True)
                ckpt = torch.load(args.ckpt_path, map_location=device)
                state_dict = ckpt['expert'] if 'expert' in ckpt else ckpt
                model.expert.load_state_dict(state_dict, strict=False)
                if 'action_in_proj' in ckpt: model.action_in_proj.load_state_dict(ckpt['action_in_proj'])
                if 'action_out_proj' in ckpt: model.action_out_proj.load_state_dict(ckpt['action_out_proj'])
                print("Checkpoint applied.")
            else:
                print("\nStep 3: Skipping checkpoint load (running as baseline-FT with original weights).", flush=True)

            ds = load_data(args.masked_data)
            res = run_inference(model, ds, device, args.num_samples, 3, f"FT-Model ({ckpt_name})")
            save_with_metadata(json_ft_masked, res, args.model_id, args.masked_data, args.ckpt_path, args.num_samples)

    # 4. Final Visualization
    print("\nStep 4: Generating Comparison Visualization...", flush=True)
    class VisArgs: pass
    vis_args = VisArgs()
    vis_args.results_jsons = f"{json_baseline_full},{json_baseline_masked},{json_ft_masked}"
    vis_args.labels = f"Original (Full),Original (Masked),FT-{ckpt_name}"
    vis_args.data_path = args.full_data
    vis_args.output_file = os.path.join(args.output_dir, "images", f"benchmark_vs_{ckpt_name}.png")
    
    visualize(vis_args)
    print(f"\nBenchmark Complete!")
    print(f" - JSON Result: {json_ft_masked}")
    print(f" - Comparison Image: {vis_args.output_file}")

if __name__ == "__main__":
    main()
