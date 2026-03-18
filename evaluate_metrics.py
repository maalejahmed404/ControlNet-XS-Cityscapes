import os
import json
import torch
import argparse
from PIL import Image
import torchvision.transforms.functional as TF
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import torch.nn.functional as F

# We use Segformer for measuring mIoU on the generated images vs the Ground Truth seg map
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, CLIPProcessor, CLIPModel

def calculate_miou(pred_map, gt_map, num_classes=19):
    # mIoU calculation between numpy arrays
    pred_map = np.array(pred_map)
    gt_map = np.array(gt_map)
    # This is a highly abstracted placeholder for Cityscapes 19-class mIoU
    # In practice, you evaluate intersection and union per valid class layout
    intersection = np.logical_and(pred_map == gt_map, gt_map < num_classes)
    union = np.logical_or(pred_map == gt_map, gt_map < num_classes)
    iou = np.sum(intersection) / (np.sum(union) + 1e-6)
    return iou

def evaluate(gen_dir, eval_dir="data/eval"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running Evaluation on {device} ---")
    
    metadata_path = os.path.join(eval_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Run prepare_dataset.py first to create data/eval/")
        
    with open(metadata_path, "r") as f:
        samples = [json.loads(line) for line in f]
        
    # Initialize Metrics
    print("Loading FID...")
    fid = FrechetInceptionDistance(feature=64).to(device)
    
    print("Loading CLIP (Transformers)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_scores = []
    
    print("Loading Segformer (for mIoU extraction)...")
    feature_extractor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024").to(device)

    print(f"Computing metrics over {len(samples)} samples...")
    miou_scores = []
    
    for i, meta in enumerate(samples):
        real_img_path = os.path.join(eval_dir, meta["image"])
        fake_img_path = os.path.join(gen_dir, f"gen_{i}.png")
        cond_img_path = os.path.join(eval_dir, meta["conditioning_image"])
        
        if not os.path.exists(fake_img_path):
            print(f"Warning: {fake_img_path} missing. Skipping.")
            continue
            
        real_img = Image.open(real_img_path).convert("RGB")
        fake_img = Image.open(fake_img_path).convert("RGB")
        
        # Format for FID
        real_tensor = TF.pil_to_tensor(real_img).unsqueeze(0).to(device)
        fake_tensor = TF.pil_to_tensor(fake_img).unsqueeze(0).to(device)
        
        fid.update(real_tensor, real=True)
        fid.update(fake_tensor, real=False)
        
        # Format for CLIP
        prompt = meta["text"]
        clip_inputs = clip_processor(text=[prompt], images=fake_img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            clip_outputs = clip_model(**clip_inputs)
            # The logits_per_image is equivalent to 100 * cos_sim
            score = clip_outputs.logits_per_image.squeeze().item()
        clip_scores.append(max(score, 0.0))
        
        # Segformer mIoU Extraction
        inputs = feature_extractor(images=fake_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = seg_model(**inputs)
        
        logits = outputs.logits  # (batch_size, num_classes, height, width)
        # Rescale logits to original image size
        logits = F.interpolate(logits, size=fake_img.size[::-1], mode="bilinear", align_corners=False)
        pred_seg = logits.argmax(dim=1).squeeze().cpu().numpy()
        
        # Pseudo-mIoU proxy metric
        gt_seg = np.array(Image.open(cond_img_path).convert("L")) # Load original conditioning mask
        miou = calculate_miou(pred_seg, gt_seg)
        miou_scores.append(miou)
        
    final_fid = fid.compute().item()
    final_clip = np.mean(clip_scores)
    avg_miou = np.mean(miou_scores)
    
    separator = "=" * 40
    report = (
        f"{separator}\n"
        f"Results for generation directory: {gen_dir}\n"
        f"FID (Visual Realism)   : {final_fid:.4f}\n"
        f"CLIPScore (Text Align) : {final_clip:.4f}\n"
        f"mIoU Proxy (Ctrl Align): {avg_miou:.4f}\n"
        f"{separator}\n"
    )
    
    print(report)
    
    # Save automatically to a text file
    report_file = "evaluation_results.txt"
    with open(report_file, "a") as f:
        f.write(report)
    print(f"Metrics saved to {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", required=True, help="Directory containing generated gen_X.png images")
    parser.add_argument("--eval_dir", default="data/eval", help="Path to evaluation ground-truth dataset")
    args = parser.parse_args()
    evaluate(args.gen_dir, args.eval_dir)
