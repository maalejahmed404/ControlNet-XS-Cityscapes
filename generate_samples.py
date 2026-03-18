import os
import sys
import json
import torch
import argparse
import numpy as np

# --- AUTO-PATCH: PyTorch >= 2.6 compat (weights_only default) ---
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
# --- END AUTO-PATCH ---
from PIL import Image
from omegaconf import OmegaConf

# Add ControlNet-XS to path to import its modules
sys.path.append(os.path.abspath("ControlNet-XS"))
from sgm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd

    # Create the model using the ControlNet-XS utilities
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def generate_samples(config_path, controlnet_id, output_dir, eval_data_dir="data/eval"):
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Generating samples on {device}...")
    
    # Load config and model
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, controlnet_id)
    sampler = DDIMSampler(model)
    
    metadata_path = os.path.join(eval_data_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing eval metadata at: {metadata_path}")
        
    with open(metadata_path, "r") as f:
        samples = [json.loads(line) for line in f]
        
    print(f"Generating {len(samples)} images. Saving to {output_dir}/")
    
    for i, meta in enumerate(samples):
        out_path = os.path.join(output_dir, f"gen_{i}.png")
        if os.path.exists(out_path):
            continue
            
        prompt = meta["text"]
        cond_path = os.path.join(eval_data_dir, meta["conditioning_image"])
        control_img = Image.open(cond_path).convert("RGB")
        control_img = control_img.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Preprocess control image (H, W, C) -> (1, C, H, W) in [-1, 1] range for Canny
        control_img_np = np.array(control_img).astype(np.float32) / 255.0
        # If the controlnet was trained on Canny, it expects 1-channel or 3-channel edges.
        # Here we just pass the RGB image through the model's prepare_cond step if available, 
        # or format it directly. Assuming RGB to match DummyBase midas/canny shapes.
        c_tensor = torch.from_numpy(control_img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Get text conditioning
        c_crossattn = model.get_learned_conditioning([prompt])
        
        # 50 DDIM steps and 9.5 CFG
        uc_crossattn = model.get_learned_conditioning([""])
        
        # Create full conditioning dict expected by TwoStreamControlLDM
        cond = {
            "c_crossattn": [c_crossattn],
            "c_concat": [c_tensor]
        }
        uncond = {
            "c_crossattn": [uc_crossattn],
            "c_concat": [c_tensor]
        }
        
        shape = (4, 512 // 8, 512 // 8)
        samples_cfg, _ = sampler.sample(
            S=50,
            conditioning=cond,
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=9.5,
            unconditional_conditioning=uncond,
            eta=0.0,
        )
        
        # Decode latent
        x_samples = model.decode_first_stage(samples_cfg)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
        
        img = Image.fromarray((x_samples[0] * 255).astype(np.uint8))
        img.save(out_path)
        
        if i % 10 == 0:
            print(f"Generated {i}/{len(samples)}...")
            
    print("Generation complete!")

def main():
    parser = argparse.ArgumentParser()
    # model_id is not needed anymore since the base model weights are loaded via the config sync_path
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5", help="Ignored, kept for CLI compat")
    parser.add_argument("--controlnet_id", required=True, help="Path to trained ControlNet .ckpt file")
    parser.add_argument("--config", default="ControlNet-XS/configs/training/sd/cityscapes_seg.yaml", help="Path to YAML config")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated outputs")
    parser.add_argument("--eval_data_dir", default="data/eval", help="Path to held-out evaluation dataset")
    # subfolder ignored
    parser.add_argument("--subfolder", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    
    generate_samples(
        args.config, 
        args.controlnet_id, 
        args.output_dir, 
        eval_data_dir=args.eval_data_dir,
    )

if __name__ == "__main__":
    main()
