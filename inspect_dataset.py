
import os
os.environ["HF_HOME"] = "/workspace/hf"
from datasets import load_dataset

ds_name = "nvidia/PhysicalAI-Autonomous-Vehicles"

# Load and inspect more samples
try:
    ds = load_dataset(ds_name, "default", split="train", streaming=True)
    print("Sample keys and values:")
    for i, item in enumerate(ds):
        print(f"\n=== Sample {i} ===")
        for k, v in item.items():
            if isinstance(v, bytes):
                print(f"  {k}: <bytes len={len(v)}>")
            elif hasattr(v, '__len__') and len(str(v)) > 100:
                print(f"  {k}: <truncated, type={type(v).__name__}>")
            else:
                print(f"  {k}: {v}")
        if i >= 2:
            break
except Exception as e:
    print(f"Error: {e}")
