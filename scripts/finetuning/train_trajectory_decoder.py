
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import einops
from transformers import AutoConfig

print("Starting train_trajectory_decoder.py...", flush=True)

# Ensure HF cache goes to /workspace where there's disk space
os.environ.setdefault("HF_HOME", "/workspace/hf")

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

class TrajectoryDataset(Dataset):
    def __init__(self, pt_file_path):
        try:
            self.data = torch.load(pt_file_path, weights_only=False)
        except TypeError:
            self.data = torch.load(pt_file_path)
            
        if not isinstance(self.data, list): 
             self.data = [self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_fn(batch):
    collated = {}
    tokenized_keys = batch[0]["tokenized_data"].keys()
    collated["tokenized_data"] = {}
    for key in tokenized_keys:
        items = [item["tokenized_data"][key] for item in batch]
        if items[0] is None:
            collated["tokenized_data"][key] = None
            continue
            
        if key in ["input_ids", "attention_mask"]:
            max_len = max(t.shape[0] for t in items)
            padded = []
            for t in items:
                if t.shape[0] < max_len:
                    pad_size = max_len - t.shape[0]
                    padding = torch.zeros(pad_size, dtype=t.dtype)
                    t = torch.cat([t, padding])
                padded.append(t)
            collated["tokenized_data"][key] = torch.stack(padded)
        elif key in ["pixel_values", "image_grid_thw"]:
            collated["tokenized_data"][key] = torch.cat(items, dim=0)
        else:
            collated["tokenized_data"][key] = torch.stack(items)
    
    for key in ["ego_future_xyz", "ego_future_rot", "ego_history_xyz", "ego_history_rot"]:
        tensors = [item[key] for item in batch]
        collated[key] = torch.cat(tensors, dim=0)
    
    collated["clip_id"] = [item["clip_id"] for item in batch]
    return collated

def compute_flow_matching_loss(model: AlpamayoR1, batch):
    device = model.device
    tokenized_data = batch["tokenized_data"]
    input_ids = tokenized_data["input_ids"].to(device)
    attention_mask = tokenized_data["attention_mask"].to(device)
    pixel_values = tokenized_data.get("pixel_values", None)
    if pixel_values is not None: pixel_values = pixel_values.to(device)
    image_grid_thw = tokenized_data.get("image_grid_thw", None)
    if image_grid_thw is not None: image_grid_thw = image_grid_thw.to(device)
    
    traj_history_xyz = batch["ego_history_xyz"].to(device)
    traj_history_rot = batch["ego_history_rot"].to(device)
    traj_future_xyz = batch["ego_future_xyz"].to(device)
    traj_future_rot = batch["ego_future_rot"].to(device)

    # Prepare diffusion targets
    x_1 = traj_future_xyz # (B, 1, Tf, 3)
    x_0 = torch.randn_like(x_1)
    t = torch.rand((x_1.shape[0], 1, 1, 1), device=device)
    x_t = (1 - t) * x_0 + t * x_1
    u_t = x_1 - x_0

    traj_data_vlm = {
        "ego_history_xyz": traj_history_xyz,
        "ego_history_rot": traj_history_rot,
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj_data_vlm)

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

    # Diffusion head forward
    batch_size = x_t.shape[0]
    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
    
    # Project to embedding space
    # t is (B, 1, 1, 1), we need (B, 1) for action_in_proj
    # Also ensure dtype matches model (bfloat16)
    model_dtype = model.expert.dtype
    x_t = x_t.to(dtype=model_dtype)
    t_proj = t.view(batch_size, 1).to(dtype=model_dtype)
    
    future_token_embeds = model.action_in_proj(x_t.squeeze(1), timesteps=t_proj) # (B, Tf, hidden)
    
    seq_len = past_key_values.get_seq_length()
    position_ids = torch.arange(seq_len, seq_len + n_diffusion_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
    
    attention_mask_expert = torch.ones((batch_size, 1, n_diffusion_tokens, seq_len + n_diffusion_tokens), device=device, dtype=model.vlm.dtype)
    
    expert_out = model.expert(
        inputs_embeds=future_token_embeds,
        position_ids=position_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask_expert,
        use_cache=False
    )
    
    
    v_pred = model.action_out_proj(expert_out.last_hidden_state)
    
    # Check output dimension match
    # Model might predict 2D (x, y) while data has 3D (x, y, z)
    out_dim = v_pred.shape[-1]
    u_t_target = u_t.squeeze(1) # (B, Tf, 3)
    if u_t_target.shape[-1] > out_dim:
        u_t_target = u_t_target[..., :out_dim]
        
    loss = nn.functional.mse_loss(v_pred, u_t_target.to(v_pred.dtype))
    return loss

def train(args):
    print("Loading Model Config...", flush=True)
    model = AlpamayoR1.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16) 
    device = "cuda"
    model.to(device)
    
    # Freeze/Unfreeze
    print("Freezing VLM / Unfreezing Diffusion Head...", flush=True)
    for param in model.vlm.parameters(): param.requires_grad = False
    for module in [model.expert, model.action_in_proj, model.action_out_proj]:
        for param in module.parameters(): param.requires_grad = True
            
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    print(f"Loading Dataset from {args.data_path}...", flush=True)
    dataset = TrajectoryDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    print(f"Starting Training for {args.epochs} epochs...", flush=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss = compute_flow_matching_loss(model, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}", flush=True)
            
        print(f"Epoch {epoch} Finished. Avg Loss: {total_loss / len(dataloader):.4f}", flush=True)
        
        # Save Safety
        save_path = os.path.join(args.output_dir, f"ckpt_epoch_{epoch}.pt")
        checkpoint = {
            'expert': {k: v.cpu() for k, v in model.expert.state_dict().items()},
            'action_in_proj': {k: v.cpu() for k, v in model.action_in_proj.state_dict().items()},
            'action_out_proj': {k: v.cpu() for k, v in model.action_out_proj.state_dict().items()},
        }
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
