import physical_ai_av
import pandas as pd

def inspect_presence():
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    df = avdi.sensor_presence
    
    # Check first 500 clips
    subset = df.head(500)
    
    print(f"Total rows in head(500): {len(subset)}")
    print("\nColumns and non-null counts:")
    print(subset.info())
    
    # In PhysicalAI-AV, egomotion is often a required label but some might be missing.
    # Let's see if there's an 'egomotion' column or similar.
    # Actually, the SDK manages this. 
    
    # Let's just try to get EGOMOTION for each of the 500 and log the FAILURES.
    failures = []
    print("\nChecking EGOMOTION availability for first 500 clips...")
    for clip_id in subset.index:
        try:
            # Check meta with streaming enabled
            feat = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
            # Actually try to sample once to ensure it's valid
            ego = feat(np.array([5100000], dtype=np.int64))
        except Exception as e:
            failures.append((clip_id, str(e)))
            
    print(f"\nFound {len(failures)} failures in first 500 clips.")
    if failures:
        print("\nCommon errors:")
        errors = {}
        for _, err in failures:
            errors[err] = errors.get(err, 0) + 1
        for err, count in errors.items():
            print(f" - {err}: {count}")
        
        print("\nFirst 10 failing IDs:")
        for cid, _ in failures[:10]:
            print(f"   {cid}")

if __name__ == "__main__":
    inspect_presence()
