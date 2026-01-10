#!/usr/bin/env python3
"""
Curvature Distribution Visualization & High-Curvature Scene Sampling

Creates:
1. Histogram of curvature distribution across the dataset
2. Sample images from top high-curvature clips
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import physical_ai_av

def plot_curvature_distribution(results_path, output_path):
    """Create histogram of curvature distribution."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    all_results = data.get('all_results', [])
    curvatures = [r['max_abs_curv'] for r in all_results if 'max_abs_curv' in r]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Full histogram
    ax1.hist(curvatures, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='Threshold (0.05 = 20m radius)')
    ax1.set_xlabel('Max Absolute Curvature (1/m)', fontsize=12)
    ax1.set_ylabel('Number of Clips', fontsize=12)
    ax1.set_title('Curvature Distribution in PhysicalAI-AV Dataset', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotation
    high_curv_count = sum(1 for c in curvatures if c > 0.05)
    total = len(curvatures)
    ax1.annotate(
        f'High Curvature (>0.05): {high_curv_count}/{total} ({100*high_curv_count/total:.1f}%)',
        xy=(0.05, ax1.get_ylim()[1] * 0.8),
        xytext=(0.1, ax1.get_ylim()[1] * 0.8),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='red'),
        color='red'
    )
    
    # Right: Log-scale histogram (to show rare high-curvature events)
    bins_log = np.logspace(np.log10(0.001), np.log10(max(curvatures)), 30)
    ax2.hist(curvatures, bins=bins_log, edgecolor='black', alpha=0.7, color='coral')
    ax2.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='Threshold (radius < 20m)')
    ax2.set_xscale('log')
    ax2.set_xlabel('Max Absolute Curvature (1/m) [Log Scale]', fontsize=12)
    ax2.set_ylabel('Number of Clips', fontsize=12)
    ax2.set_title('Log-Scale View: Rare High-Curvature Clips', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved curvature distribution plot: {output_path}")
    return high_curv_count, total


def extract_high_curve_samples(results_path, output_dir, top_n=5):
    """Extract sample images and trajectory plots from top high-curvature clips."""
    import scipy.spatial.transform as spt
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    top_clips = data.get('top_clips', [])[:top_n]
    
    print(f"\nExtracting sample images from top {top_n} high-curvature clips...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Layout: 5 columns (4 cameras + 1 trajectory), top_n rows
    fig, axes = plt.subplots(top_n, 5, figsize=(20, 4 * top_n))
    
    # Use correct camera feature paths
    camera_features = [
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
    ]
    camera_names = ['Left', 'Front', 'Right', 'Tele']
    
    t0_us = 5_100_000
    time_step = 0.1
    num_history_steps = 16
    num_future_steps = 64
    
    for i, clip_info in enumerate(top_clips):
        clip_id = clip_info['clip_id']
        curvature = clip_info['max_abs_curv']
        radius = 1.0 / curvature if curvature > 0 else float('inf')
        
        print(f"  {i+1}. {clip_id} (curvature: {curvature:.4f}, radius: {radius:.1f}m)")
        
        try:
            # Get images from all 4 cameras
            for j, (cam_feature, cam_name) in enumerate(zip(camera_features, camera_names)):
                camera = avdi.get_clip_feature(
                    clip_id,
                    cam_feature,
                    maybe_stream=True
                )
                frames, _ = camera.decode_images_from_timestamps(np.array([t0_us], dtype=np.int64))
                img = frames[0]
                
                ax = axes[i, j] if top_n > 1 else axes[j]
                ax.imshow(img)
                ax.set_title(f"{cam_name}", fontsize=10)
                ax.axis('off')
                
                if j == 0:
                    ax.set_ylabel(f"#{i+1}: {clip_id[:8]}...\nR={radius:.1f}m", fontsize=9)
            
            # Get trajectory data for XY plot
            egomotion = avdi.get_clip_feature(
                clip_id,
                avdi.features.LABELS.EGOMOTION,
                maybe_stream=True
            )
            
            # Compute timestamps
            history_offsets_us = np.arange(
                -(num_history_steps - 1) * time_step * 1_000_000,
                time_step * 1_000_000 / 2,
                time_step * 1_000_000,
            ).astype(np.int64)
            history_timestamps = t0_us + history_offsets_us
            
            future_offsets_us = np.arange(
                time_step * 1_000_000,
                (num_future_steps + 0.5) * time_step * 1_000_000,
                time_step * 1_000_000,
            ).astype(np.int64)
            future_timestamps = t0_us + future_offsets_us
            
            ego_history = egomotion(history_timestamps)
            ego_future = egomotion(future_timestamps)
            
            # Transform to local frame
            ego_history_xyz = ego_history.pose.translation
            ego_history_quat = ego_history.pose.rotation.as_quat()
            ego_future_xyz = ego_future.pose.translation
            
            t0_xyz = ego_history_xyz[-1].copy()
            t0_rot = spt.Rotation.from_quat(ego_history_quat[-1])
            t0_rot_inv = t0_rot.inv()
            
            ego_history_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
            ego_future_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
            
            # Plot trajectory
            ax_traj = axes[i, 4] if top_n > 1 else axes[4]
            ax_traj.plot(ego_history_local[:, 0], ego_history_local[:, 1], 'g-', linewidth=2, label='History')
            ax_traj.plot(ego_future_local[:, 0], ego_future_local[:, 1], 'gold', linewidth=3, label='GT Future')
            ax_traj.plot(0, 0, 'ro', markersize=8, label='Ego')
            ax_traj.set_title(f"Trajectory (R={radius:.1f}m)", fontsize=10)
            ax_traj.legend(fontsize=8, loc='upper right')
            ax_traj.grid(True, alpha=0.3)
            ax_traj.axis('equal')
            ax_traj.set_xlabel('X (m)')
            ax_traj.set_ylabel('Y (m)')
                    
        except Exception as e:
            print(f"    Error loading clip: {e}")

    
    plt.suptitle('Top High-Curvature Clips in PhysicalAI-AV Dataset', fontsize=14, y=1.02)
    plt.tight_layout()
    
    sample_output = output_dir / 'high_curvature_samples.png'
    plt.savefig(sample_output, dpi=150, bbox_inches='tight')
    print(f"✓ Saved high-curvature samples: {sample_output}")
    
    return top_clips


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='trajectory_bias_experiment/logs/curve_scan_500samples.json')
    parser.add_argument('--output_dir', type=str, default='trajectory_bias_experiment/images')
    args = parser.parse_args()
    
    # Plot distribution
    high_count, total = plot_curvature_distribution(
        args.results,
        f"{args.output_dir}/curvature_distribution.png"
    )
    
    # Extract samples
    top_clips = extract_high_curve_samples(
        args.results,
        args.output_dir,
        top_n=5
    )
    
    print("\nDone!")
