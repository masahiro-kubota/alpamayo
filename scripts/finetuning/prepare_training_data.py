
import argparse
import sys
import os
import torch
import numpy as np
import scipy.spatial.transform as spt
from pathlib import Path
import hashlib

# Add src to python path to import alpamayo_r1
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
# from alpamayo_r1 import helper # Moved to lazy import

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    sys.exit(1)

def quaternion_to_matrix(q):
    return spt.Rotation.from_quat(q).as_matrix()

def get_split(uuid_str, train_ratio=0.9):
    # Deterministic split based on UUID
    hash_val = int(hashlib.md5(uuid_str.encode()).hexdigest(), 16)
    return "train" if (hash_val % 100) < (train_ratio * 100) else "val"

def process_hf_dataset(output_path, num_samples=100, target_split="train"):
    print(f"Loading HF Dataset (Streaming mode) for target split: {target_split}")
    
    # 1. Load Ego Motion (Labels)
    try:
        ds_labels = load_dataset("nvidia/PhysicalAI-Autonomous-Vehicles", "labels", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading labels: {e}")
        return

    label_buffer = {}
    print("Buffering initial labels...")
    for i, item in enumerate(ds_labels):
        clip_id = item.get('clip_uuid', str(i))
        
        # Check Split
        if get_split(clip_id) != target_split:
            continue

        if clip_id not in label_buffer:
            label_buffer[clip_id] = []
        label_buffer[clip_id].append(item)
        
        if len(label_buffer) > num_samples * 2: # heuristic buffer limit
            if i > num_samples * 1000: break
            
    print(f"Buffered labels for {len(label_buffer)} clips ({target_split}).")

    # 2. Load Camera Data
    ds_camera = load_dataset("nvidia/PhysicalAI-Autonomous-Vehicles", "camera_front_wide_120fov", split="train", streaming=True)
    
    processed_count = 0
    
    # 3. Model Processor
    from transformers import AutoTokenizer
    print("Loading processor...")
    try:
        from alpamayo_r1 import helper
        tokenizer = AutoTokenizer.from_pretrained("nvidia/Alpamayo-R1-10B", trust_remote_code=True)
        processor = helper.get_processor(tokenizer)
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        return

    data_list = []

    for video_item in ds_camera:
        clip_id = video_item.get('clip_uuid', None)
        
        # Check Split and Availability
        if clip_id not in label_buffer:
            continue
        if get_split(clip_id) != target_split:
            continue
            
        print(f"Processing Clip {clip_id}...")
        
        # ... (Actual processing logic would go here) ...
        # For demo, creating dummy data
        
        mock_output = {
            "tokenized_data": {"input_ids": torch.zeros(1, 10).long(), "attention_mask": torch.ones(1, 10)},
            "ego_future_xyz": torch.randn(1, 1, 64, 3), 
            "ego_future_rot": torch.randn(1, 1, 64, 3, 3),
            "ego_history_xyz": torch.randn(1, 1, 16, 3),
            "ego_history_rot": torch.randn(1, 1, 16, 3, 3),
            "split": target_split, # Tagging split
            "clip_id": clip_id
        }
        
        data_list.append(mock_output)
        processed_count += 1
        
        if processed_count >= num_samples:
            break

    # Save
    torch.save(data_list, output_path)
    print(f"Saved {len(data_list)} samples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_pt", help="Path to output .pt file")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train", help="Which split to extract")
    args = parser.parse_args()
    
    process_hf_dataset(args.output_pt, args.samples, args.split)
