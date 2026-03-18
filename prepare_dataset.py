import os
import json
import io
import argparse
from datasets import load_dataset
from PIL import Image

def get_image(data):
    if isinstance(data, dict) and "bytes" in data:
        return Image.open(io.BytesIO(data["bytes"])).convert("RGB")
    return data.convert("RGB")

def prepare_split(dataset_split, output_dir, limit=None):
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/conditioning_images", exist_ok=True)
    
    metadata = []
    count = 0
    for i, sample in enumerate(dataset_split):
        if limit and count >= limit:
            break
            
        img = get_image(sample["image"]).resize((512, 512), Image.Resampling.LANCZOS)
        seg = get_image(sample["seg"]).resize((512, 512), Image.Resampling.NEAREST)
        caption = sample["caption"]
        
        img_path = f"images/{count}.png"
        seg_path = f"conditioning_images/{count}.png"
        
        img.save(f"{output_dir}/{img_path}")
        seg.save(f"{output_dir}/{seg_path}")
        
        metadata.append({
            "text": caption,
            "image": img_path,
            "conditioning_image": seg_path
        })
        count += 1
        
    with open(f"{output_dir}/metadata.jsonl", "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {count} samples to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Prepare Cityscapes Dataset for ControlNet")
    parser.add_argument("--eval_size", type=int, default=100, help="Number of samples to hold out for evaluation.")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory to save train/eval splits.")
    args = parser.parse_args()

    print("Loading liuch37/controlnet-cityscapes dataset from Hugging Face...")
    ds = load_dataset("liuch37/controlnet-cityscapes", split="train")
    
    # Shuffle consistently to ensure reproducible splits
    ds = ds.shuffle(seed=42)
    
    if args.eval_size >= len(ds):
        raise ValueError(f"Eval size ({args.eval_size}) must be smaller than dataset size ({len(ds)}).")
        
    eval_ds = ds.select(range(args.eval_size))
    train_ds = ds.select(range(args.eval_size, len(ds)))
    
    print(f"Preparing EVAL split ({args.eval_size} samples)...")
    prepare_split(eval_ds, os.path.join(args.data_dir, "eval"), limit=args.eval_size)
    
    print(f"Preparing TRAIN split ({len(train_ds)} samples)...")
    # For a quick test, you can restrict train size by passing limit parameter inside prepare_split
    prepare_split(train_ds, os.path.join(args.data_dir, "train"))
    print("\nDataset split complete. Ready for training & evaluation.")

if __name__ == "__main__":
    main()
