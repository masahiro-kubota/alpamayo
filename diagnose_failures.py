import physical_ai_av
import numpy as np
import tqdm

def diagnose():
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    sensor_df = avdi.sensor_presence
    all_clip_ids = sensor_df.index.tolist()
    
    # Check first 50 clips to see if we get errors
    sampled = all_clip_ids[:50]
    
    print(f"Diagnosing {len(sampled)} clips...")
    failures = 0
    error_types = {}
    
    for clip_id in sampled:
        try:
            # Replicating logic from scan_all_curves.py
            egomotion = avdi.get_clip_feature(
                clip_id,
                avdi.features.LABELS.EGOMOTION,
                maybe_stream=True,
            )
            # Try to actually access data
            timestamps = np.array([5100000], dtype=np.int64)
            ego_data = egomotion(timestamps)
        except Exception as e:
            failures += 1
            err_msg = str(e)
            error_types[err_msg] = error_types.get(err_msg, 0) + 1
            print(f"X Fail: {clip_id} - {err_msg}")
            
    print("\nDIAGNOSIS SUMMARY:")
    print(f"Total: {len(sampled)}")
    print(f"Success: {len(sampled) - failures}")
    print(f"Failures: {failures}")
    print("\nError Breakdown:")
    for err, count in error_types.items():
        print(f" - {err}: {count}")

if __name__ == "__main__":
    diagnose()
