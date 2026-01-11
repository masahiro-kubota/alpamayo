import physical_ai_av
import numpy as np

def test_single():
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    clip_id = "25cd4769-5dcf-4b53-a351-bf2c5deb6124"
    print(f"Testing clip: {clip_id}")
    
    try:
        print("Attempting to get EGOMOTION with maybe_stream=True...")
        feat = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
        print("✓ Feature handle obtained")
        
        print("Attempting to sample data at 5.1s...")
        ego = feat(np.array([5100000], dtype=np.int64))
        print("✓ Data sampled successfully!")
        print(f"  Curvature: {ego.curvature[0]}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

if __name__ == "__main__":
    test_single()
