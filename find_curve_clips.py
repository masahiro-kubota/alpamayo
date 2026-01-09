#!/usr/bin/env python3
"""
Scan NVIDIA PhysicalAI-AV Dataset for Curve Scenarios

This script analyzes the egomotion data from the NVIDIA dataset to identify
clips containing significant curves/turns, which can be used to verify that
the model can handle curved trajectories on official data.

Usage:
    python find_curve_clips.py --top_n 10

Requirements:
    - physical-ai-av package installed
    - Dataset access (will stream from HuggingFace if not local)
"""

import argparse
import numpy as np
import physical_ai_av
from tqdm import tqdm


def analyze_clip_curvature(avdi, clip_id, t0_us=5_100_000):
    """Analyze curvature characteristics of a clip.
    
    Args:
        avdi: PhysicalAIAVDatasetInterface instance
        clip_id: Clip ID to analyze
        t0_us: Timestamp to analyze around (default 5.1s)
        
    Returns:
        dict with curvature statistics
    """
    try:
        egomotion = avdi.get_clip_feature(
            clip_id,
            avdi.features.LABELS.EGOMOTION,
            maybe_stream=True,
        )
        
        # Sample curvature around t0 (±3 seconds at 10Hz = 60 samples)
        timestamps = np.arange(
            t0_us - 3_000_000,  # -3s
            t0_us + 3_000_000,  # +3s
            100_000,             # 0.1s = 10Hz
        ).astype(np.int64)
        
        ego_data = egomotion(timestamps)
        
        # Extract curvature values (1/meters = inverse radius)
        # Higher absolute value = sharper curve
        curvatures = ego_data.curvature
        
        return {
            'clip_id': clip_id,
            'max_abs_curv': np.abs(curvatures).max(),
            'mean_abs_curv': np.abs(curvatures).mean(),
            'std_curv': curvatures.std(),
            'curvature_data': curvatures,
        }
    except Exception as e:
        print(f"Error analyzing {clip_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Find curve scenarios in NVIDIA dataset")
    parser.add_argument('--top_n', type=int, default=10, help="Number of top curved clips to find")
    parser.add_argument('--sample_size', type=int, default=100, 
                       help="Number of clips to sample (set to -1 to scan all)")
    args = parser.parse_args()
    
    print("Initializing NVIDIA PhysicalAI-AV Dataset Interface...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    
    # Get list of clip IDs
    # Note: You may need to adjust this based on how the dataset provides clip IDs
    # For now, we'll use a known example and suggest manual exploration
    
    print("\n" + "="*80)
    print("IMPORTANT: Dataset Exploration Strategy")
    print("="*80)
    print("""
The NVIDIA dataset likely contains metadata files (parquet) listing all clip_ids.
To efficiently find curve scenarios:

1. Check for metadata files in the dataset (e.g., 'vla_golden.parquet')
2. Use the example clip_id from test_inference.py as a starting point
3. The dataset includes 'curvature' field in egomotion labels

Example known clip_id: '030c760c-ae38-49aa-9ad8-f5650a545d26'

Analyzing this clip for curvature...
""")
    
    # Analyze the known example clip
    example_clip = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    result = analyze_clip_curvature(avdi, example_clip, t0_us=5_100_000)
    
    if result:
        print(f"\nResults for example clip '{example_clip}':")
        print(f"  Max Absolute Curvature:  {result['max_abs_curv']:.6f} (1/m)")
        print(f"  Mean Absolute Curvature: {result['mean_abs_curv']:.6f} (1/m)")
        print(f"  Std Deviation:           {result['std_curv']:.6f}")
        
        # Convert curvature to approximate turn radius
        if result['max_abs_curv'] > 0:
            radius = 1.0 / result['max_abs_curv']
            print(f"  Minimum Turn Radius:     ~{radius:.1f} meters")
            
            if result['max_abs_curv'] > 0.01:  # Radius < 100m
                print("\n  ✓ This clip contains CURVES (curvature > 0.01)")
            else:
                print("\n  ✗ This clip is mostly STRAIGHT (curvature < 0.01)")
        
        print(f"\nYou can test this clip with:")
        print(f"  python test_inference.py  # (already uses this clip)")
        print(f"\nOr modify debug_viz.py to load from NVIDIA dataset instead of rosbag.")


if __name__ == "__main__":
    main()
