import torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

def test_variable_steps(num_frames):
    CLIP = "f789b390-1698-4f99-b237-6de4cbbb7666"
    print(f"--- Testing with num_frames={num_frames} ---")
    try:
        model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
        
        # Load data with custom num_frames
        data = load_physical_aiavdataset(CLIP, t0_us=5_100_000, num_frames=num_frames)
        print(f"Data shape: {data['image_frames'].shape}") # (4, T, 3, H, W)
        
        print("Model inference test start...")
        with torch.no_grad():
            from alpamayo_r1 import helper
            processor = helper.get_processor(model.tokenizer)
            tokenized_data = processor(
                image_frames=data["image_frames"],
                camera_indices=data["camera_indices"],
                ego_history_xyz=data["ego_history_xyz"],
                ego_history_rot=data["ego_history_rot"],
            )
            data["tokenized_data"] = tokenized_data
            
            # This will call sample_trajectories_from_data_with_vlm_rollout
            pred_xyz, pred_rot = model.sample_trajectories_from_data_with_vlm_rollout(data)
            
        print(f"✓ Success. pred_xyz shape: {pred_xyz.shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_variable_steps(1) # Try 1 frame instead of 4

if __name__ == "__main__":
    # test_variable_steps(1)
    pass
