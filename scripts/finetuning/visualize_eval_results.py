
import argparse
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_pt_data(pt_file):
    """Load preprocessed validation data to get images."""
    try:
        data = torch.load(pt_file, weights_only=False)
    except TypeError:
        data = torch.load(pt_file)
    if not isinstance(data, list):
        data = [data]
    # Create a mapping from clip_id to images_np
    id_to_imgs = {item.get("clip_id", f"sample_{i}"): item.get("images_np", []) for i, item in enumerate(data)}
    return id_to_imgs

def visualize(args):
    # 1. Load multiple result JSONs
    json_paths = args.results_jsons.split(',')
    labels = args.labels.split(',') if args.labels else [f"Model {i}" for i in range(len(json_paths))]
    colors = ['blue', 'purple', 'orange', 'cyan'] # Colors for different models
    
    all_results = []
    metadata_list = []
    for path in json_paths:
        print(f"Loading results from {path}...", flush=True)
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and "results" in data:
                all_results.append(data["results"])
                metadata_list.append(data.get("metadata", {}))
            else:
                # Fallback for old list-only JSONs
                all_results.append(data)
                metadata_list.append({})
    
    print(f"Loading source images from {args.data_path}...", flush=True)
    id_to_imgs = load_pt_data(args.data_path)
    
    # Use the first result set to determine number of samples
    num_eval = len(all_results[0])
    if num_eval == 0:
        print("No results to visualize.")
        return

    # 2. Setup Plot
    fig, axes = plt.subplots(num_eval, 5, figsize=(25, 5 * num_eval), dpi=100)
    plt.suptitle("Alpamayo-R1: Baseline vs Fine-tuned Comparison", fontsize=26, y=0.98, fontweight='bold')

    for i in range(num_eval):
        # Base info from the first model (GT, History, ClipID are same across all models for same data)
        base_res = all_results[0][i]
        clip_id = base_res["clip_id"]
        gt = np.array(base_res["gt_future"])
        hist = np.array(base_res["history"])
        
        # Get images
        imgs = id_to_imgs.get(clip_id, [])
        cam_indices = [3, 7, 11, 15] 
        cam_titles = ["Left (Masked)", "Front Wide", "Right (Masked)", "Front Tele"]

        # 1-4: Camera Images
        for col in range(4):
            ax = axes[i, col] if num_eval > 1 else axes[col]
            if len(imgs) >= 16:
                img = imgs[cam_indices[col]]
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Image N/A", ha='center', va='center', fontsize=12)
            
            if i == 0:
                ax.set_title(cam_titles[col], fontsize=18, fontweight='bold', pad=15)
            ax.axis('off')

        # 5: Trajectory Plot
        ax_plot = axes[i, 4] if num_eval > 1 else axes[4]
        
        # Plotting GT and History
        ax_plot.plot(hist[:, 0], hist[:, 1], 'r-', linewidth=2, label='History' if i==0 else "")
        ax_plot.plot(gt[:, 0], gt[:, 1], 'g-', linewidth=5, label='Ground Truth' if i==0 else "", zorder=5)
        
        # Plot each model's predictions
        for m_idx, res_list in enumerate(all_results):
            preds = np.array(res_list[i]["predictions"]) # (N, T, 3)
            duration = res_list[i].get("duration_sec", 0)
            label = f"{labels[m_idx]} ({duration:.1f}s)"
            color = colors[m_idx % len(colors)]
            
            for n, pred in enumerate(preds):
                ax_plot.plot(pred[:, 0], pred[:, 1], '--', color=color, alpha=0.7, 
                             linewidth=2.5, label=label if i==0 and n==0 else "")
            
        # Ego car position
        ax_plot.scatter([0], [0], color='red', marker='X', s=150, zorder=10, label='Ego' if i==0 else "")
        
        ax_plot.set_title(f"BEV Comparison ({clip_id[:8]})", fontsize=16, pad=10)
        ax_plot.set_xlabel("X (forward) [m]", fontsize=12)
        ax_plot.set_ylabel("Y (left) [m]", fontsize=12)
        ax_plot.grid(True, linestyle=':', alpha=0.7)
        ax_plot.axis('equal')
        
        # Add a subtle background color to highlight the plot
        ax_plot.set_facecolor('#f9f9f9')

        if i == 0:
            ax_plot.legend(loc='upper right', fontsize=12, framealpha=0.9, shadow=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True) if os.path.dirname(args.output_file) else None
    plt.savefig(args.output_file, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to {args.output_file}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_jsons", type=str, required=True, help="Comma-separated paths to result JSONs (e.g. baseline.json,ft.json)")
    parser.add_argument("--labels", type=str, required=True, help="Comma-separated labels for the models (e.g. Baseline,Fine-tuned)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to source .pt data (to get images)")
    parser.add_argument("--output_file", type=str, default="baseline_comparison.png")
    args = parser.parse_args()
    
    visualize(args)
