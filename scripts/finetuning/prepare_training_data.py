#!/usr/bin/env python3
"""
Prepare training data for Trajectory Decoder finetuning using physical_ai_av library.

This script uses the official PhysicalAIAVDatasetInterface to properly load
video frames and egomotion labels from the nvidia/PhysicalAI-Autonomous-Vehicles dataset.
"""

import argparse
import os
import sys
import hashlib
from pathlib import Path

import torch
import numpy as np
import scipy.spatial.transform as spt

# Ensure HF_HOME is set before importing datasets
os.environ.setdefault("HF_HOME", "/workspace/hf")

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
from transformers import AutoTokenizer, AutoProcessor


def get_split(uuid_str: str, train_ratio: float = 0.9) -> str:
    """Deterministic split based on clip UUID hash."""
    hash_val = int(hashlib.md5(uuid_str.encode()).hexdigest(), 16)
    return "train" if (hash_val % 100) < (train_ratio * 100) else "val"


def process_dataset(output_path: str, num_samples: int = 10, target_split: str = "train"):
    """Process the PhysicalAI-AV dataset and save training data."""
    
    print(f"Initializing PhysicalAIAVDatasetInterface...")
    dataset = PhysicalAIAVDatasetInterface(
        cache_dir="/workspace/hf",
        confirm_download_threshold_gb=100.0,  # Allow larger downloads without prompt
    )
    
    print(f"Available features: {dataset.features.ALL}")
    print(f"Total clips: {len(dataset.clip_index)}")
    
    # Load tokenizer and processor
    print("Loading processor...")
    from alpamayo_r1 import helper
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", trust_remote_code=True)
    processor = helper.get_processor(tokenizer)
    
    data_list = []
    processed_count = 0
    skipped_count = 0
    
    # Iterate through clips
    clip_ids = dataset.clip_index.index.tolist()
    
    for clip_id in clip_ids:
        # Check split
        if get_split(clip_id) != target_split:
            continue
            
        # Check if front camera is available
        if not dataset.sensor_presence.at[clip_id, "camera_front_wide_120fov"]:
            skipped_count += 1
            continue
            
        try:
            # Download features for this clip (caches automatically)
            print(f"Processing clip {clip_id}...")
            dataset.download_clip_features(clip_id, ["egomotion", "camera_front_wide_120fov"])
            
            # Get egomotion interpolator
            egomotion_interp = dataset.get_clip_feature(clip_id, "egomotion")
            
            # Get video reader
            video_reader = dataset.get_clip_feature(clip_id, "camera_front_wide_120fov")
            
            # Get timestamp range
            timestamps = video_reader.timestamps  # in microseconds
            if timestamps is None or len(timestamps) < 100:
                print(f"  Skipping {clip_id}: insufficient timestamps")
                video_reader.close()
                continue
            
            # Choose a reference timestamp (middle of clip to ensure history/future exist)
            # Alpamayo uses 10Hz: 1.6s history (16 frames), 6.4s future (64 frames)
            # Timestamps are in microseconds
            hist_duration_us = 1_600_000  # 1.6s
            fut_duration_us = 6_400_000   # 6.4s
            
            min_ts = timestamps.min()
            max_ts = timestamps.max()
            
            ref_ts = (min_ts + max_ts) // 2
            
            # Ensure we have enough history and future
            if ref_ts - hist_duration_us < min_ts or ref_ts + fut_duration_us > max_ts:
                print(f"  Skipping {clip_id}: clip too short")
                video_reader.close()
                continue
            
            # Generate timestamps for history and future at 10Hz
            hist_timestamps = np.linspace(ref_ts - hist_duration_us, ref_ts, 16, dtype=np.int64)
            fut_timestamps = np.linspace(ref_ts, ref_ts + fut_duration_us, 64, dtype=np.int64)
            
            # Get image at reference time
            images, actual_ts = video_reader.decode_images_from_timestamps(np.array([ref_ts], dtype=np.int64))
            image = images[0]  # (H, W, C) numpy array
            
            # Get egomotion at history and future timestamps
            hist_ego = egomotion_interp(hist_timestamps)
            fut_ego = egomotion_interp(fut_timestamps)
            
            # Extract pose (position and rotation)
            # hist_ego.pose is RigidTransform with .translation (N, 3) and .rotation
            hist_xyz = torch.tensor(hist_ego.pose.translation, dtype=torch.float32)  # (16, 3)
            hist_rot = torch.tensor(hist_ego.pose.rotation.as_matrix(), dtype=torch.float32)  # (16, 3, 3)
            
            fut_xyz = torch.tensor(fut_ego.pose.translation, dtype=torch.float32)  # (64, 3)
            fut_rot = torch.tensor(fut_ego.pose.rotation.as_matrix(), dtype=torch.float32)  # (64, 3, 3)
            
            # Convert image to PIL for processor
            from PIL import Image
            pil_image = Image.fromarray(image)
            
            # Create VLM input using same format as helper.create_message
            num_traj_token = 48
            hist_traj_placeholder = f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a driving assistant that generates safe and accurate actions."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "<|cot_start|>"}],
                },
            ]
            
            # Process with Qwen processor
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = processor(
                text=[text],
                images=[pil_image],
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            
            tokenized_data = {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "pixel_values": inputs["pixel_values"].squeeze(0) if "pixel_values" in inputs else None,
                "image_grid_thw": inputs["image_grid_thw"].squeeze(0) if "image_grid_thw" in inputs else None,
            }
            
            data_item = {
                "tokenized_data": tokenized_data,
                "ego_future_xyz": fut_xyz.unsqueeze(0).unsqueeze(0),  # (1, 1, 64, 3)
                "ego_future_rot": fut_rot.unsqueeze(0).unsqueeze(0),  # (1, 1, 64, 3, 3)
                "ego_history_xyz": hist_xyz.unsqueeze(0).unsqueeze(0),  # (1, 1, 16, 3)
                "ego_history_rot": hist_rot.unsqueeze(0).unsqueeze(0),  # (1, 1, 16, 3, 3)
                "split": target_split,
                "clip_id": clip_id,
            }
            
            data_list.append(data_item)
            processed_count += 1
            print(f"  Processed {processed_count}/{num_samples} samples")
            
            video_reader.close()
            
            if processed_count >= num_samples:
                break
                
        except Exception as e:
            print(f"  Error processing {clip_id}: {e}")
            skipped_count += 1
            continue
    
    # Save
    print(f"\nSaving {len(data_list)} samples to {output_path}")
    torch.save(data_list, output_path)
    print(f"Done! Processed: {processed_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for Trajectory Decoder finetuning")
    parser.add_argument("output_pt", help="Path to output .pt file (e.g., data/debug.pt)")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to extract")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train", help="Which split to extract")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_pt) or ".", exist_ok=True)
    
    process_dataset(args.output_pt, args.samples, args.split)
