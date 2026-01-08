# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Set HF Cache to workspace (large storage)
os.environ["HF_HOME"] = "/workspace/hf"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf/transformers"
os.environ["HF_DATASETS_CACHE"] = "/workspace/hf/datasets"

# Add scripts dir to path to import convert_rosbag
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))
try:
    from convert_rosbag import process_bag
except ImportError:
    # Fallback if running from proper root
    sys.path.append(os.path.abspath("scripts"))
    from convert_rosbag import process_bag

from mcap_ros2.reader import read_ros2_messages

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

def get_bag_range(bag_path):
    print(f"Scanning bag {bag_path} for time range...")
    start_t = None
    end_t = None
    # We only care about odom or any msg
    # Filter to known topics to avoid DecoderNotFoundError on custom types
    topics = ["/localization/kinematic_state"]
    for msg in read_ros2_messages(str(bag_path), topics=topics):
        t = msg.log_time_ns
        if start_t is None or t < start_t:
            start_t = t
        if end_t is None or t > end_t:
            end_t = t
    return start_t, end_t

def run_test():
    # Define paths
    bag_path = Path("/workspace/alpamayo/rosbag2_autoware/rosbag2_autoware_0.mcap")
    if not bag_path.exists():
        print(f"Error: Bag file not found at {bag_path}")
        return

    # Get range
    start_ns, end_ns = get_bag_range(bag_path)
    print(f"Bag Range: {start_ns} to {end_ns} (Duration: {(end_ns-start_ns)/1e9:.2f}s)")

    # Sample 3 points: 20%, 50%, 80%
    offsets = [0.2, 0.5, 0.8]
    
    # Load model ONCE
    print("Loading model...")
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
    # Get processor from model tokenizer (robust)
    processor = helper.get_processor(model.tokenizer)

    for ratio in offsets:
        t0_ns = int(start_ns + (end_ns - start_ns) * ratio)
        print(f"\n\n=== Processing Sample at {ratio*100}% (t0={t0_ns}) ===")
        
        output_pt = Path(f"test_rosbag_{int(ratio*100)}.pt")
        
        # Force process bag
        process_bag(str(bag_path), str(output_pt), t0_ns=t0_ns)

        print(f"Loading data from {output_pt}...")
        data = torch.load(output_pt, weights_only=False)
        
        # Prepare inputs using fallback tokenization
        # Since convert_rosbag fails to load tokenizer, tokenized_data is None.
        # This is good.
        
        # We need image_frames
        flat_frames = data["image_frames"].flatten(0, 1)
        messages = helper.create_message(flat_frames)
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = helper.to_device(inputs, "cuda")

        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        
        model_inputs = helper.to_device(model_inputs, "cuda")

        print("Running Inference...")
        torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1, 
                max_generation_length=256,
                return_extra=True,
            )

        print("Chain-of-Causation:")
        cot = extra["cot"][0]
        print(cot)

if __name__ == "__main__":
    run_test()
