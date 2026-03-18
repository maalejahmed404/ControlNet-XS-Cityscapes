import os
import sys
import subprocess
import argparse
import glob
import time

def main():
    parser = argparse.ArgumentParser(description="Full ControlNet-XS Pipeline: Train -> Generate -> Evaluate")
    parser.add_argument("--data_dir", default="data/train", help="Path to training data")
    parser.add_argument("--eval_data_dir", default="data/eval", help="Path to evaluation data")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and proceed directly to generation (if model exists)")
    args = parser.parse_args()

    repo_dir = "ControlNet-XS"

    # ================================================================
    # Phase 1: Training
    # ================================================================
    if not args.skip_training:
        print("\n--- Phase 3.1: Starting ControlNet-XS Training ---")
        
        train_cmd = [sys.executable, "train_xs.py", f"--data_dir={args.data_dir}"]
        if args.epochs is not None:
            train_cmd.append(f"--epochs={args.epochs}")
            
        subprocess.run(train_cmd, check=True)
        print("\n[INFO] Training script finished.")
    else:
        print("\n--- Phase 3.1: Skipping Training (using existing checkpoints) ---")

    # ================================================================
    # Phase 2: Locate the trained checkpoint
    # ================================================================
    # First, check the user-provided explicit model path
    xs_model_path = os.path.abspath("models/controlnet_xs")

    if not os.path.exists(xs_model_path):
        # Automatically find the latest checkpoint in the repo logs
        logs_path = os.path.join(repo_dir, "logs")
        if os.path.isdir(logs_path):
            ckpt_files = glob.glob(os.path.join(logs_path, "**", "checkpoints", "*.ckpt"), recursive=True)
            if ckpt_files:
                latest_ckpt = max(ckpt_files, key=os.path.getmtime)
                xs_model_path = latest_ckpt
                print(f"\n[INFO] Automatically selected checkpoint: {xs_model_path}")
            else:
                print(f"\n[ALERT] No .ckpt files found under {logs_path}.")
                print("Please copy/link your desired checkpoint folder to 'models/controlnet_xs' and re-run.")
                return
        else:
            print(f"\n[ALERT] Neither 'models/controlnet_xs' nor '{logs_path}' exist.")
            print("Ensure training completed successfully, then re-run.")
            return

    # ================================================================
    # Phase 3: Generation
    # ================================================================
    print("\n--- Phase 3.2: Generating Evaluation Samples ---")
    output_dir = "results/controlnet_xs"
    gen_cmd = [
        sys.executable, "generate_samples.py",
        f"--config={os.path.join(repo_dir, 'configs', 'training', 'sd', 'cityscapes_seg.yaml')}",
        f"--controlnet_id={xs_model_path}",
        f"--output_dir={output_dir}",
        f"--eval_data_dir={args.eval_data_dir}",
    ]
    subprocess.run(gen_cmd, check=True)

    # ================================================================
    # Phase 4: Evaluation
    # ================================================================
    print("\n--- Phase 3.3: Computing Metrics ---")
    eval_cmd = [
        sys.executable, "evaluate_metrics.py",
        f"--gen_dir={output_dir}",
        f"--eval_dir={args.eval_data_dir}",
    ]
    subprocess.run(eval_cmd, check=True)

    print("\nXS Pipeline Complete. Metrics appended to `evaluation_results.txt`.")

if __name__ == "__main__":
    main()
