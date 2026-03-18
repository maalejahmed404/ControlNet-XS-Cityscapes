import os
import argparse
import urllib.request
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Train Standard Base ControlNet.")
    parser.add_argument("--data_dir", type=str, default="data/train", help="Path to training data.")
    parser.add_argument("--output_dir", type=str, default="models/standard_controlnet", help="Path to save weights.")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    args = parser.parse_args()

    # Step 1: Download the official train_controlnet.py script from diffusers
    script_url = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/controlnet/train_controlnet.py"
    script_path = "train_controlnet_hf.py"
    
    if not os.path.exists(script_path):
        print("Downloading standard diffusers train_controlnet.py...")
        urllib.request.urlretrieve(script_url, script_path)
    
    # Step 2: Configure Accelerate
    data_dir_abs = os.path.abspath(args.data_dir)
    output_dir_abs = os.path.abspath(args.output_dir)
    
    cmd = [
        "accelerate", "launch", script_path,
        "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
        f"--dataset_name={data_dir_abs}",
        f"--output_dir={output_dir_abs}",
        "--resolution=512",
        "--learning_rate=1e-5",
        f"--train_batch_size={args.train_batch_size}",
        f"--gradient_accumulation_steps={args.gradient_accumulation_steps}",
        "--mixed_precision=fp16",
        "--enable_xformers_memory_efficient_attention",
        "--checkpointing_steps=500",
        f"--max_train_steps={args.max_train_steps}"
    ]
    
    print("\n--- Starting Standard ControlNet Training ---")
    print(f"Data Dir: {data_dir_abs}")
    print(f"Executing: {' '.join(cmd)}\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
