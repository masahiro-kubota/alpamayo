import argparse
import sys
import os

"""
Convert Rosbag to Alpamayo Input Script

Usage:
    uv run python scripts/convert_rosbag.py <input_mcap> <output_pt>

Description:
    This script converts a ROS 2 mcap file into the tensor format required for Alpamayo inference.
    
    Key Logic:
    1.  **Time Synchronization**: Since topic frequencies differ (e.g., Odom 100Hz, Camera 10-30Hz), 
        we sample data at fixed 0.1s intervals by finding the message with the closest timestamp (nearest neighbor).
    2.  **Single Camera Handling**: The model expects input shape including a camera dimension: (N_cameras, T, 3, H, W).
        For single-camera setups, we stack frames to (T, 3, H, W) and then unsqueeze dimension 0 to get (1, T, 3, H, W).
"""
import torch
import numpy as np
import scipy.spatial.transform as spt
from pathlib import Path
from mcap_ros2.reader import read_ros2_messages
from einops import rearrange
# Add src to python path to import alpamayo_r1
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from alpamayo_r1 import helper

# Camera topics mapping
CAMERA_TOPICS = {
    "/sensing/camera/image_raw": 1, # Default as front_wide if only one
}

ODOM_TOPIC = "/localization/kinematic_state"

def get_message_time(msg_obj):
    return msg_obj.log_time

def ros_time_to_us(ros_time):
    return ros_time * 1e-3 # log_time in nanoseconds -> microseconds

def quaternion_to_matrix(q):
    return spt.Rotation.from_quat(q).as_matrix()

def process_bag(input_path, output_path, t0_ns=None):
    print(f"Processing {input_path}...")
    
    # Store data
    images = [] # list of (time_ns, camera_topic, data_bytes, width, height)
    odom_msgs = [] # list of (time_ns, position, orientation_quat)

    # 1. Read messages
    # We need to scan the bag to find a suitable t0 if not provided
    # For simplicity, we'll read everything into memory first (assuming reasonable bag size as per metadata ~200k msgs)
    # Optimization: Read only relevant topics
    topics = list(CAMERA_TOPICS.keys()) + [ODOM_TOPIC]
    
    msg_count = 0
    for msg in read_ros2_messages(input_path, topics=topics):
        msg_count += 1
        if msg.channel.topic not in topics:
            continue
        
        msg_count += 1
        t_ns = msg.log_time_ns
        
        if msg.channel.topic == ODOM_TOPIC:
            # Odometry msg: nav_msgs/msg/Odometry
            # structure: msg.ros_msg.pose.pose.position.x ...
            ros_msg = msg.ros_msg
            pos = ros_msg.pose.pose.position
            ori = ros_msg.pose.pose.orientation
            odom_msgs.append((t_ns, 
                              np.array([pos.x, pos.y, pos.z]), 
                              np.array([ori.x, ori.y, ori.z, ori.w])))
            
        elif msg.channel.topic in CAMERA_TOPICS:
            # Image msg: sensor_msgs/msg/Image
            ros_msg = msg.ros_msg
            # We assume rgb8 encoding or similar.
            # Convert raw data to numpy array
            # msg.data is bytes
            width = ros_msg.width
            height = ros_msg.height
            encoding = ros_msg.encoding
            
            # Simple decoding for raw, assuming rgb8 or bgr8
            # If compressed, needs cv2 or PIL. Metadata says sensor_msgs/msg/Image usually raw.
            # But let's check encoding if possible.
            # For now store the object to decode later based on t0
            images.append({
                'time_ns': t_ns,
                'topic': msg.channel.topic,
                'ros_msg': ros_msg
            })

    print(f"Read {len(odom_msgs)} odom messages and {len(images)} images.")
    
    if not odom_msgs:
        print("No odometry data found!")
        return

    # Sort by time
    odom_msgs.sort(key=lambda x: x[0])
    images.sort(key=lambda x: x['time_ns'])

    # Determine t0
    # Default: pick a time near the end to have enough history, and ensure we have images
    # Let's say 2 seconds before the end of odometry, or based on image availability
    if t0_ns is None:
        last_odom_time = odom_msgs[-1][0]
        # Try 5 seconds before end
        t0_ns = last_odom_time - 2 * 1_000_000_000
    
    print(f"Target t0: {t0_ns} ns")

    # Parameters
    num_history_steps = 16
    time_step = 0.1 # seconds
    num_frames = 4 # frames per camera
    
    # --- 2. Extract Ego History ---
    # History timestamps: t0 - (N-1)*dt ... t0
    history_timestamps_ns = [int(t0_ns - (num_history_steps - 1 - i) * time_step * 1e9) for i in range(num_history_steps)]
    
    ego_history_xyz = []
    ego_history_rot = []
    
    # Helper to interpolate odom
    # Use simple nearest neighbor or linear interp. For 100Hz odom, nearest is fine.
    # Odom is sorted.
    odom_times = np.array([x[0] for x in odom_msgs])
    
    for t_query in history_timestamps_ns:
        # --- Time Synchronization (Odometry) ---
        # Since odometry is high frequency (~100Hz), we search for the timestamp closest 
        # to our target query time (nearest neighbor sampling) to handle frequency differences.
        idx = np.searchsorted(odom_times, t_query)
        if idx == 0:
            val = odom_msgs[0]
        elif idx == len(odom_msgs):
            val = odom_msgs[-1]
        else:
            # Nearest
            t1 = odom_msgs[idx-1][0]
            t2 = odom_msgs[idx][0]
            if abs(t_query - t1) < abs(t_query - t2):
                val = odom_msgs[idx-1]
            else:
                val = odom_msgs[idx]
        
        ego_history_xyz.append(val[1])
        ego_history_rot.append(quaternion_to_matrix(val[2]))

    ego_history_xyz = np.array(ego_history_xyz) # (T, 3)
    ego_history_Rot = np.array(ego_history_rot) # (T, 3, 3)

    # Transform to local frame at t0
    t0_pos = ego_history_xyz[-1]
    t0_Rot = ego_history_Rot[-1]
    t0_Rot_inv = t0_Rot.T # rotation matrix inverse is transpose

    ego_history_xyz_local = (ego_history_xyz - t0_pos) @ t0_Rot_inv.T # (T, 3)
    # Check rotation calc: R_local = R_t0^T * R_current
    ego_history_rot_local = np.matmul(t0_Rot_inv, ego_history_Rot)

    # Format tensors (1, 1, T, ...)
    tensor_xyz = torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    tensor_rot = torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)

    # --- 3. Extract Images ---
    # Timestamps: t0 - (frames-1)*dt ... t0 (approx 10Hz intervals)
    # Camera frequencies might differ. We want frames closest to these times.
    img_timestamps_ns = [int(t0_ns - (num_frames - 1 - i) * time_step * 1e9) for i in range(num_frames)]
    
    final_frames_list = []
    
    # Process only the single configured camera for now
    cam_topic = "/sensing/camera/image_raw"
    cam_images = [img for img in images if img['topic'] == cam_topic]
    
    if not cam_images:
        print(f"No images found for topic {cam_topic}")
        # dummy fallback?
        return 

    cam_times = np.array([img['time_ns'] for img in cam_images])
    
    frames_tensor_list = []
    for t_query in img_timestamps_ns:
        # --- Time Synchronization (Camera) ---
        # Find the frame with the timestamp closest to the target query time.
        idx = np.abs(cam_times - t_query).argmin()
        best_img = cam_images[idx]
        
        # Decode image
        ros_msg = best_img['ros_msg']
        # Assume rgb8, 8-bit.
        # Data is a flat buffer.
        img_arr = np.frombuffer(ros_msg.data, dtype=np.uint8)
        img_arr = img_arr.reshape((ros_msg.height, ros_msg.width, 3))
        
        # ROS images are typically BGR (bgr8). Model expects RGB.
        # Check encoding or force convert if user reports issues.
        # Blue sky -> Orange => BGR is interpreted as RGB. We need to swap B and R.
        # BGR -> RGB
        img_arr = img_arr[..., ::-1].copy()
        
        # Tensor (3, H, W)
        frame_tensor = torch.from_numpy(img_arr.copy()).permute(2, 0, 1)
        frames_tensor_list.append(frame_tensor)

    # Stack frames (num_frames, 3, H, W)
    frames_stacked = torch.stack(frames_tensor_list)
    
    # --- Single Camera Handling ---
    # The model expects input shape: (N_cameras, num_frames, 3, H, W)
    # Since we only have 1 camera, we add the first dimension manually.
    image_frames = frames_stacked.unsqueeze(0) 

    # --- 4. Tokenize ---
    # Create message
    # flatten to list of separate images for create_message?
    # helper.create_message expects (N_total_frames, 3, H, W) or list of tensors?
    # The helper says: assert frames.ndim == 4. Flatten cameras and time.
    flat_frames = image_frames.flatten(0, 1)
    
    messages = helper.create_message(flat_frames)
    
    # Load processor
    # Note: Loading model/processor might require download. 
    # If this fails in restricted env, we might need to skip this step or mock it.
    # Assuming user has internet or cached models.
    try:
        from transformers import AutoTokenizer
        # Dummy tokenizer loading if actual path not found or restrictive
        # We try to load the processor as in inference.ipynb
        # "Qwen/Qwen3-VL-2B-Instruct" -> "nvidia/Alpamayo-R1-10B"
        # We must use the fine-tuned tokenizer so that special tokens (e.g. <|cot_start|>) are recognized.
        print("Loading processor...")
        tokenizer = AutoTokenizer.from_pretrained("nvidia/Alpamayo-R1-10B", trust_remote_code=True)
        processor = helper.get_processor(tokenizer)
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as e:
        print(f"Warning: Tokenization failed (likely due to missing model/internet). Saving raw data. Error: {e}")
        inputs = None

    # --- 5. Save ---
    output_dict = {
        "tokenized_data": inputs,
        "ego_history_xyz": tensor_xyz,
        "ego_history_rot": tensor_rot,
        "t0_pos": t0_pos, # Global position at t0
        "t0_Rot": t0_Rot, # Global rotation at t0
        "t0_ns": t0_ns,
        # Save raw images too just in case re-tokenization is needed
        "image_frames": image_frames 
    }
    
    torch.save(output_dict, output_path)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Rosbag to Alpamayo Input")
    parser.add_argument("input_bag", help="Path to input .mcap file")
    parser.add_argument("output_pt", help="Path to output .pt file")
    args = parser.parse_args()
    
    process_bag(args.input_bag, args.output_pt)
