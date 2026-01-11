import physical_ai_av
import numpy as np
import json
from tqdm import tqdm

def robust_scan():
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    sensor_df = avdi.sensor_presence
    all_clip_ids = sensor_df.index.tolist()
    
    sampled = all_clip_ids[:500]
    results = []
    errors = []
    
    print(f"Starting robust scan of {len(sampled)} clips...")
    
    for clip_id in tqdm(sampled):
        try:
            # Replicating original logic
            egomotion = avdi.get_clip_feature(
                clip_id,
                avdi.features.LABELS.EGOMOTION,
                maybe_stream=True,
            )
            timestamps = np.array([5100000], dtype=np.int64)
            ego_data = egomotion(timestamps)
            curvatures = ego_data.curvature
            
            results.append({
                'clip_id': clip_id,
                'max_abs_curv': float(np.abs(curvatures).max()),
                'has_sharp_curve': bool(np.abs(curvatures).max() > 0.05)
            })
        except Exception as e:
            errors.append({'clip_id': clip_id, 'error': str(e)})

    print("\nSCAN RESULTS:")
    print(f"Total Attempted: {len(sampled)}")
    print(f"Total Succeeded: {len(results)}")
    print(f"Total Failed: {len(errors)}")
    
    high_curv_count = sum(1 for r in results if r['has_sharp_curve'])
    print(f"High-curvature count (>0.05): {high_curv_count}")

    # Save results for visualization
    output_path = 'trajectory_bias_experiment/logs/curve_scan_500samples.json'
    with open(output_path, 'w') as f:
        json.dump({
            'threshold': 0.05,
            'total_scanned': len(results),
            'high_curvature_count': high_curv_count,
            'top_clips': sorted(results, key=lambda x: x['max_abs_curv'], reverse=True)[:50],
            'all_results': results
        }, f, indent=2)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    robust_scan()
