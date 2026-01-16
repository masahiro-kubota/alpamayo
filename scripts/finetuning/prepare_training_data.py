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
import time
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


def process_dataset(output_path: str, num_samples: int = 10, target_split: str = "train", target_clip_ids: list[str] = None, no_mask: bool = False):
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
    if target_clip_ids:
        # Use specific clip IDs but filter only those actually in the dataset
        all_available_ids = set(dataset.clip_index.index.tolist())
        clip_ids = [cid for cid in target_clip_ids if cid in all_available_ids]
        num_samples = len(clip_ids)
        print(f"Targeting {len(clip_ids)} specific clips.")
    else:
        clip_ids = dataset.clip_index.index.tolist()
    
    for clip_id in clip_ids:
        # Check split if not explicitly targeting IDs
        if not target_clip_ids and get_split(clip_id) != target_split:
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
            min_ts = timestamps.min()
            max_ts = timestamps.max()
            ref_ts = (min_ts + max_ts) // 2
            # This initial ref_ts is not used if max curvature logic is applied
            # min_ts = timestamps.min()
            # max_ts = timestamps.max()
            # ref_ts = (min_ts + max_ts) // 2
            
            # Durations
            hist_duration_us = 1_600_000  # 1.6s
            fut_duration_us = 6_400_000   # 6.4s
            
            # 1. Find the point of maximum curvature in the clip
            egomotion_interp = dataset.get_clip_feature(clip_id, "egomotion")
            
            # Use official attributes of the Interpolator
            min_ts, max_ts = egomotion_interp.time_range
            scan_ts = np.arange(min_ts + 2_000_000, max_ts - 7_000_000, 100_000) # Buffer for history/future
            
            if len(scan_ts) < 10:
                 ref_ts = min_ts + 5_000_000
            else:
                egos = egomotion_interp(scan_ts)
                # egos.curvature is a numpy array of shape (N, 1) or (N,)
                curv_data = np.abs(egos.curvature).flatten()
                
                max_idx = np.argmax(curv_data)
                ref_ts = scan_ts[max_idx]
                print(f"  -> Max curvature {curv_data[max_idx]:.4f} found at {ref_ts/1e6:.1f}s")

            # 2. Extract Images at ref_ts
            # Generate timestamps for images: 4 frames at 10Hz [t0-0.3s, t0-0.2s, t0-0.1s, t0]
            img_duration_us = 300_000 # 0.3s
            img_timestamps = np.linspace(ref_ts - img_duration_us, ref_ts, 4, dtype=np.int64)
            
            # 1. Front Wide
            video_reader_wide = dataset.get_clip_feature(clip_id, "camera_front_wide_120fov")
            imgs_wide, _ = video_reader_wide.decode_images_from_timestamps(img_timestamps)
            video_reader_wide.close()
            
            # 2. Front Tele
            dataset.download_clip_features(clip_id, ["camera_front_tele_30fov"])
            video_reader_tele = dataset.get_clip_feature(clip_id, "camera_front_tele_30fov")
            imgs_tele, _ = video_reader_tele.decode_images_from_timestamps(img_timestamps)
            video_reader_tele.close()
            
            if no_mask:
                # 3. Left
                dataset.download_clip_features(clip_id, ["camera_cross_left_120fov"])
                video_reader_left = dataset.get_clip_feature(clip_id, "camera_cross_left_120fov")
                imgs_left, _ = video_reader_left.decode_images_from_timestamps(img_timestamps)
                video_reader_left.close()
                
                # 4. Right
                dataset.download_clip_features(clip_id, ["camera_cross_right_120fov"])
                video_reader_right = dataset.get_clip_feature(clip_id, "camera_cross_right_120fov")
                imgs_right, _ = video_reader_right.decode_images_from_timestamps(img_timestamps)
                video_reader_right.close()
            else:
                # Create black images for Left and Right (same shape as wide)
                black_frame = np.zeros_like(imgs_wide[0])
                imgs_left = np.stack([black_frame] * 4)
                imgs_right = np.stack([black_frame] * 4)
            
            # Arrange in Alpamayo-R1 order: Left, FrontWide, Right, FrontTele (each 4 frames)
            # This matches load_physical_aiavdataset.py indices [0, 1, 2, 6]
            all_imgs_np = np.concatenate([imgs_left, imgs_wide, imgs_right, imgs_tele], axis=0)
            # all_imgs_np: (16, H, W, 3)
            
            # Trajectory History & Future
            hist_timestamps = np.linspace(ref_ts - 1_500_000, ref_ts, 16, dtype=np.int64)
            fut_timestamps = np.linspace(ref_ts + 100_000, ref_ts + 6_400_000, 64, dtype=np.int64)
            
            hist_ego = egomotion_interp(hist_timestamps)
            fut_ego = egomotion_interp(fut_timestamps)
            
            hist_xyz = torch.tensor(hist_ego.pose.translation, dtype=torch.float32)
            hist_rot = torch.tensor(hist_ego.pose.rotation.as_matrix(), dtype=torch.float32)
            fut_xyz = torch.tensor(fut_ego.pose.translation, dtype=torch.float32)
            fut_rot = torch.tensor(fut_ego.pose.rotation.as_matrix(), dtype=torch.float32)
            
            from PIL import Image
            pil_images = [Image.fromarray(img) for img in all_imgs_np]
            
            # VLM Prompt
            num_traj_token = 48
            hist_traj_placeholder = f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
            
            user_content = [{"type": "image", "image": img} for img in pil_images]
            user_content.append({"type": "text", "text": f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory."})
            
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a driving assistant that generates safe and accurate actions."}]},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]}
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = processor(text=[text], images=pil_images, videos=None, padding=True, return_tensors="pt")
            
            tokenized_data = {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "pixel_values": inputs["pixel_values"].squeeze(0) if "pixel_values" in inputs else None,
                "image_grid_thw": inputs["image_grid_thw"].squeeze(0) if "image_grid_thw" in inputs else None,
            }
            
            data_item = {
                "tokenized_data": tokenized_data,
                "ego_future_xyz": fut_xyz.unsqueeze(0).unsqueeze(0),
                "ego_future_rot": fut_rot.unsqueeze(0).unsqueeze(0),
                "ego_history_xyz": hist_xyz.unsqueeze(0).unsqueeze(0),
                "ego_history_rot": hist_rot.unsqueeze(0).unsqueeze(0),
                "split": target_split if not target_clip_ids else "custom",
                "clip_id": clip_id,
                "images_np": all_imgs_np,
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
    
    print(f"\nSaving {len(data_list)} samples to {output_path}")
    torch.save(data_list, output_path)
    print(f"Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for Trajectory Decoder finetuning")
    parser.add_argument("output_pt", help="Path to output .pt file")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to extract")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train", help="Which split to extract")
    parser.add_argument("--clip_ids", type=str, help="Comma-separated list of clip IDs to extract")
    parser.add_argument("--no_mask", action="store_true", help="Do not mask Left/Right cameras (Full Vision)")
    args = parser.parse_args()
    
    if os.path.exists(args.output_pt):
        print(f"File {args.output_pt} already exists. Skipping data generation.", flush=True)
        sys.exit(0)
    
    os.makedirs(os.path.dirname(args.output_pt) or ".", exist_ok=True)
    clip_id_list = args.clip_ids.split(",") if args.clip_ids else None
    
    start_time = time.time()
    process_dataset(args.output_pt, args.samples, args.split, clip_id_list, args.no_mask)
    end_time = time.time()
    
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")