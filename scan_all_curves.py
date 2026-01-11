#!/usr/bin/env python3
"""
Comprehensive Scan of NVIDIA PhysicalAI-AV Dataset for High-Curvature Clips

Scans the dataset to find clips with significant curves (curvature > 0.05 = radius < 20m).

Usage:
    python scan_all_curves.py --threshold 0.05 --max_clips 1000 --output curve_clips.json
"""

import argparse
import json
import numpy as np
import physical_ai_av
from tqdm import tqdm
import pandas as pd


import time

def analyze_clip_curvature(avdi, clip_id, t0_us=5_100_000, sample_window_us=6_000_000, retries=5):
    """Analyze curvature characteristics of a clip with retries for network resilience."""
    last_error = None
    for attempt in range(retries):
        try:
            egomotion = avdi.get_clip_feature(
                clip_id,
                avdi.features.LABELS.EGOMOTION,
                maybe_stream=True,
            )
            
            # Sample curvature around t0 (±3 seconds at 10Hz)
            half_window = sample_window_us // 2
            timestamps = np.arange(
                t0_us - half_window,
                t0_us + half_window,
                100_000,  # 0.1s = 10Hz
            ).astype(np.int64)
            
            ego_data = egomotion(timestamps)
            curvatures = ego_data.curvature
            
            return {
                'clip_id': clip_id,
                'max_abs_curv': float(np.abs(curvatures).max()),
                'mean_abs_curv': float(np.abs(curvatures).mean()),
                'std_curv': float(curvatures.std()),
                'has_sharp_curve': bool(np.abs(curvatures).max() > 0.05),
            }
        except Exception as e:
            last_error = str(e)
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))  # Exponential backoff
            continue
            
    return {'clip_id': clip_id, 'error': last_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.05, 
                       help="Curvature threshold (1/m), default 0.05 = radius 20m")
    parser.add_argument('--max_clips', type=int, default=1000, 
                       help="Maximum number of clips to scan (for testing)")
    parser.add_argument('--output', type=str, default='curve_scan_results.json',
                       help="Output JSON file")
    parser.add_argument('--sample_every', type=int, default=1,
                       help="Sample every Nth clip (for faster scanning)")
    args = parser.parse_args()
    
    print("="*80)
    print("NVIDIA PhysicalAI-AV Dataset Curvature Scanner")
    print("="*80)
    print(f"Threshold: {args.threshold} (1/m) = Radius {1.0/args.threshold:.1f}m")
    print(f"Max clips to scan: {args.max_clips}")
    print(f"Sample rate: every {args.sample_every} clip(s)")
    print("="*80 + "\n")
    
    # Initialize dataset interface
    print("Initializing dataset...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    
    # Get list of all clip IDs
    print("Loading clip index...")
    
    try:
        # Access sensor_presence DataFrame - clip_id is the index
        sensor_df = avdi.sensor_presence
        all_clip_ids = sensor_df.index.tolist()
        print(f"✓ Successfully loaded {len(all_clip_ids):,} clip IDs from sensor_presence")
    except Exception as e:
        print(f"ERROR: Cannot access sensor_presence: {e}")
        print("Falling back to single example clip...")
        all_clip_ids = ["030c760c-ae38-49aa-9ad8-f5650a545d26"]
        args.max_clips = 1
    
    # Sample clips
    total_available = len(all_clip_ids)
    print(f"Total clips available: {total_available:,}")
    
    sampled_clips = all_clip_ids[::args.sample_every][:args.max_clips]
    print(f"Scanning {len(sampled_clips):,} clips...\n")
    
    # Scan clips
    results = []
    high_curv_count = 0
    
    for clip_id in tqdm(sampled_clips, desc="Analyzing clips"):
        result = analyze_clip_curvature(avdi, clip_id)
        if result and 'error' not in result:
            results.append(result)
            if result['has_sharp_curve']:
                high_curv_count += 1
                print(f"\n✓ High-curvature clip found: {clip_id}")
                print(f"  Max curvature: {result['max_abs_curv']:.6f} (1/m)")
                print(f"  Turn radius: ~{1.0/result['max_abs_curv']:.1f}m\n")
    
    # Sort by max curvature
    results.sort(key=lambda x: x['max_abs_curv'], reverse=True)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            'threshold': args.threshold,
            'total_scanned': len(sampled_clips),
            'high_curvature_count': high_curv_count,
            'top_clips': results[:50],  # Top 50
            'all_results': results
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("SCAN COMPLETE")
    print("="*80)
    print(f"Scanned: {len(results):,} clips")
    print(f"High-curvature clips (>{args.threshold}): {high_curv_count:,}")
    print(f"Results saved to: {args.output}")
    
    # Print top 10
    if results:
        print("\nTop 10 highest-curvature clips:")
        print("-" * 80)
        for i, r in enumerate(results[:10], 1):
            radius = 1.0 / r['max_abs_curv'] if r['max_abs_curv'] > 0 else float('inf')
            print(f"{i:2d}. {r['clip_id']}")
            print(f"    Curvature: {r['max_abs_curv']:.6f} (1/m), Radius: ~{radius:.1f}m")


if __name__ == "__main__":
    main()
