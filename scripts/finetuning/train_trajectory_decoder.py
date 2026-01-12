
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import einops

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from transformers import AutoConfig, AutoModel

class TrajectoryDataset(Dataset):
    def __init__(self, pt_file_path):
        self.data = torch.load(pt_file_path)
        # In a real scenario, this would load a list of files. 
        # Here we assume a single .pt file for simplicity as per example.
        # Ensure data is in list format or handle single item
        if not isinstance(self.data, list): 
             # If .pt contains a single dict (as per prepare_training_data.py)
             self.data = [self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Return dict with tensors on CPU
        return item

def compute_flow_matching_loss(model: AlpamayoR1, batch):
    """
    Computes Conditional Flow Matching Loss for the Trajectory Decoder.
    Loss = || v_pred(x_t, t, cond) - (x_1 - x_0) ||^2
    """
    device = model.device
    
    # 1. Prepare Data
    tokenized_data = batch["tokenized_data"]
    input_ids = tokenized_data["input_ids"].to(device)
    attention_mask = tokenized_data["attention_mask"].to(device)
    pixel_values = tokenized_data.get("pixel_values", None)
    if pixel_values is not None: pixel_values = pixel_values.to(device)
    image_grid_thw = tokenized_data.get("image_grid_thw", None)
    if image_grid_thw is not None: image_grid_thw = image_grid_thw.to(device)
    
    # Ground Truth Trajectory (Future)
    traj_future_xyz = batch["ego_future_xyz"].to(device) # (B, 1, Tf, 3)
    traj_future_rot = batch["ego_future_rot"].to(device)
    traj_history_xyz = batch["ego_history_xyz"].to(device)
    traj_history_rot = batch["ego_history_rot"].to(device)
    
    # Fuse trajectory history tokens into input_ids
    traj_data_vlm = {
        "ego_history_xyz": traj_history_xyz,
        "ego_history_rot": traj_history_rot,
        # We assume training data preparation handles "ego_future" availability
        # Note: fusing future tokens? Usually for Training VLM we assume future tokens are targets or inputs?
        # But we are freezing VLM and training only Diffusion Head.
        # Diffusion Head conditions on VLM's hidden states (past_key_values).
        # We do NOT want to fuse future tokens into VLM input for conditioning, 
        # otherwise we leak future info to the condition!
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj_data_vlm)

    # 2. VLM Forward (Conditioning)
    # We run VLM to get the KV cache (representations of image + text + history)
    with torch.no_grad():
        vlm_outputs = model.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = vlm_outputs.past_key_values
        # Get hidden state of the last token (usually where generation starts)
        # Actually, 'expert' uses 'past_key_values' via cross-attention or direct continuation?
        # Alpamayo Expert architecture:
        # It takes `past_key_values` and `inputs_embeds` (projected noisy action).
        # It treats the noisy action tokens as "continuation" of the prompt.
    
    # 3. Flow Matching Setup
    batch_size = input_ids.shape[0]
    
    # Convert GT Trajectory to Action Space (Latent x_1)
    # Target x_1
    x_1 = model.action_space.traj_to_action(
        traj_history_xyz=traj_history_xyz.squeeze(1), # (B, Th, 3)
        traj_history_rot=traj_history_rot.squeeze(1),
        traj_future_xyz=traj_future_xyz.squeeze(1),   # (B, Tf, 3)
        traj_future_rot=traj_future_rot.squeeze(1)
    ) # (B, Tf, 2) -> (B, Tf, C_action)
    
    # Flatten if needed? Action dimensions: (Tf, 2) usually.
    # BaseDiffusion expects x_dims.
    
    # Sample x_0 (Noise)
    x_0 = torch.randn_like(x_1)
    
    # Sample t uniform [0, 1]
    t = torch.rand(batch_size, device=device)
    # Reshape t for broadcasting (B, 1, 1) or similar
    t_expanded = t.view(batch_size, *([1]*(x_1.ndim-1)))
    
    # Interpolate x_t
    # Flow Matching: x_t = (1 - t) * x_0 + t * x_1
    #              = x_0 + t * (x_1 - x_0)
    x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
    
    # Target Velocity u_t
    u_t = x_1 - x_0
    
    # 4. Neural Network Head Forward (Predict v_t)
    # Project x_t to embedding dimension
    # t is also passed to projection (Time embedding)
    future_token_embeds = model.action_in_proj(x_t, t) 
    # (B, Tf, hidden_size)

    # Position IDs for future tokens
    # Need to align with VLM's sequence length
    seq_len = past_key_values.get_seq_length()
    n_diffusion_tokens = model.action_space.get_action_space_dims()[0] # Tf
    
    # Create simple position ids extending the sequence
    position_ids = torch.arange(seq_len, seq_len + n_diffusion_tokens, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    
    # Create attention mask
    # Allow attending to all past tokens + current future tokens? 
    # Depends on 'expert_non_causal_attention'. If True, future tokens attend to each other?
    # Usually standard is Causal or Bidirectional for Diffusion?
    # Alpamayo config says 'expert_non_causal_attention=True' by default.
    # We construct mask matching expert call in inference.
    attention_mask_expert = torch.ones(
        (batch_size, 1, n_diffusion_tokens, seq_len + n_diffusion_tokens),
        device=device
    )
    
    # Run Expert
    # Note: 'use_cache=True' in inference updates cache. In training, we usually don't need to return new cache unless iterating.
    # Flow Matching is mostly single step training per batch.
    expert_out = model.expert(
        inputs_embeds=future_token_embeds,
        position_ids=position_ids,
        past_key_values=past_key_values, # Reuse VLM cache
        attention_mask=attention_mask_expert,
        use_cache=False # Don't need to update cache for training step
    )
    
    last_hidden = expert_out.last_hidden_state # (B, Tf, hidden)
    
    # Project back to action space
    v_pred = model.action_out_proj(last_hidden) # (B, Tf, C_action)
    
    # 5. Loss Calculation
    loss = nn.functional.mse_loss(v_pred, u_t)
    
    return loss


def train(args):
    # 1. Config & Model
    print("Loading Model Config...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    print("Loading Model...")
    # Load in bfloat16 to save memory (fits in 24GB VRAM)
    model = AlpamayoR1.from_pretrained(args.model_path, torch_dtype=torch.bfloat16) 
    device = "cuda"

    model.to(device)
    
    # 2. Freeze VLM, Unfreeze Head

    print("Freezing VLM Backbone...")
    for param in model.vlm.parameters():
        param.requires_grad = False
        
    print("Unfreezing Diffusion Head...")
    # Trainable modules: expert, action_in/out_proj, diffusion(if any params)
    trainable_modules = [model.expert, model.action_in_proj, model.action_out_proj]
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True
            
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    
    # 3. Data
    print(f"Loading Dataset from {args.data_path}...")
    dataset = TrajectoryDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 4. Training Loop
    model.train()
    print("Starting Training...")
    
    for epoch in range(args.epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Compute Loss
            loss = compute_flow_matching_loss(model, batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch} Finished. Avg Loss: {total_loss / len(dataloader):.4f}")
        
        # Save Checkpoint
        save_path = os.path.join(args.output_dir, f"ckpt_epoch_{epoch}.pt")
        # Save only trainable parts to save space
        torch.save({
            'expert': model.expert.state_dict(),
            'action_in_proj': model.action_in_proj.state_dict(),
            'action_out_proj': model.action_out_proj.state_dict(),
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to .pt training data")
    parser.add_argument("--model_path", type=str, default="nvidia/Alpamayo-R1-10B", help="Model path")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
