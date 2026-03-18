import os
import sys
import subprocess
import argparse
from omegaconf import OmegaConf


def _patch_repo(repo_dir):
    """Patch the entire ControlNet-XS repo for compatibility with modern PyTorch/PL.

    Fixes applied:
      1. opt.resume_from_checkpoint AttributeError  (PL >= 2.0)
      2. torch.load weights_only default change      (PyTorch >= 2.6)
    """
    main_py = os.path.join(repo_dir, "main.py")
    if not os.path.exists(main_py):
        return

    with open(main_py, "r", encoding="utf-8") as f:
        src = f.read()

    changed = False

    # --- Fix 1: resume_from_checkpoint AttributeError ---
    old = "ckpt_resume_path = opt.resume_from_checkpoint"
    new = "ckpt_resume_path = getattr(opt, 'resume_from_checkpoint', None)"
    if old in src and new not in src:
        src = src.replace(old, new)
        changed = True

    # --- Fix 2: Monkey-patch torch.load at the top of main.py ---
    # This globally restores the old default (weights_only=False) so that
    # EVERY torch.load call in the entire codebase works with PyTorch >= 2.6.
    monkey_patch = (
        "\n# --- AUTO-PATCH: PyTorch >= 2.6 compat (weights_only default) ---\n"
        "import torch as _torch\n"
        "_orig_load = _torch.load\n"
        "def _patched_load(*args, **kwargs):\n"
        "    if 'weights_only' not in kwargs:\n"
        "        kwargs['weights_only'] = False\n"
        "    return _orig_load(*args, **kwargs)\n"
        "_torch.load = _patched_load\n"
        "# --- END AUTO-PATCH ---\n"
    )
    if "_patched_load" not in src:
        # Insert right after the first line (or at the very top)
        src = monkey_patch + src
        changed = True

    # --- Fix 4: Remove trainer.test() call which crashes on PL >= 2.0 ---
    # PL >= 2.0 throws a strict MisconfigurationException if test() is called
    # without a test_step() defined. 
    old_test = "trainer.test(model, data)"
    if old_test in src:
        src = src.replace(old_test, "# " + old_test + "  # Patched out to avoid PL 2.0 crash")
        changed = True

    # --- Fix 3: Patch ldm.data.dummy_set.DummyBase to find images recursively ---
    # DummyBase does: os.listdir(data_root), which fails (0 samples) if images
    # are in subfolders (e.g., standard Cityscapes structure).
    dummy_set_py = os.path.join(repo_dir, "ldm", "data", "dummy_set.py")
    if os.path.exists(dummy_set_py):
        with open(dummy_set_py, "r", encoding="utf-8") as f:
            dummy_src = f.read()
        
        # We replace the simple listdir with an os.walk approach
        old_listdir = "self.image_paths = os.listdir(data_root)"
        new_listdir = (
            "self.image_paths = []\n"
            "        for root, _, files in os.walk(data_root):\n"
            "            for f in files:\n"
            "                if f.lower().endswith(('.png', '.jpg', '.jpeg')):\n"
            "                    # Store path relative to data_root\n"
            "                    self.image_paths.append(os.path.relpath(os.path.join(root, f), data_root))"
        )
        if old_listdir in dummy_src:
            dummy_src = dummy_src.replace(old_listdir, new_listdir)
            # Also need to fix the filter line which is now redundant but might crash
            old_filter = "self.image_paths = [path for path in self.image_paths if \".png\" in path or 'jpg' in path]"
            dummy_src = dummy_src.replace(old_filter, "# " + old_filter)
            
            with open(dummy_set_py, "w", encoding="utf-8") as f:
                f.write(dummy_src)
            print(f"[PATCH] Patched {dummy_set_py} to support recursive image discovery.")

    if changed:
        with open(main_py, "w", encoding="utf-8") as f:
            f.write(src)
        print(f"[PATCH] Patched {main_py} for PL >= 2.0 and PyTorch >= 2.6 compatibility.")
    else:
        print(f"[PATCH] {main_py} already patched.")

def main():
    parser = argparse.ArgumentParser(description="Train ControlNet-XS.")
    parser.add_argument("--data_dir", type=str, default="data/train", help="Path to training data.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    args = parser.parse_args()

    repo_dir = "ControlNet-XS"

    # --- Step 1: Clone the repo if it doesn't exist ---
    if not os.path.exists(repo_dir):
        print("Cloning official ControlNet-XS repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/vislearn/ControlNet-XS.git", repo_dir],
            check=True,
        )

    # --- Step 2: Install the sgm package from the repo (required by main.py) ---
    # main.py does: from sgm.util import exists, instantiate_from_config, isheatmap
    sgm_setup_py = os.path.join(repo_dir, "pyproject.toml")
    if os.path.exists(sgm_setup_py):
        print("\n--- Installing sgm package from ControlNet-XS repo ---")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", repo_dir, "--quiet"],
            check=True,
        )
    else:
        print("[WARN] pyproject.toml not found in ControlNet-XS — sgm may already be installed.")

    # --- Step 3: Patch main.py for compatibility with modern PyTorch Lightning ---
    # Bug: main.py accesses opt.resume_from_checkpoint, but PL >= 2.0 removed
    # 'resume_from_checkpoint' from Trainer.__init__, so the dynamic arg parser
    # no longer adds it. We patch the source to add a safe default.
    _patch_repo(repo_dir)

    # --- Step 4: Build / update the training YAML config ---
    data_dir_abs = os.path.abspath(args.data_dir)
    # The real config lives under configs/training/sd/ in the repo
    config_dir = os.path.join(repo_dir, "configs", "training", "sd")
    os.makedirs(config_dir, exist_ok=True)
    yaml_config_path = os.path.join(config_dir, "cityscapes_seg.yaml")

    print(f"\n--- Automating Configuration for ControlNet-XS ---")
    # Check if an existing SD1.5 canny config can be used as a base
    base_config = os.path.join(config_dir, "sd15_encD_canny_53m.yaml")
    if os.path.exists(base_config):
        print(f"Loading base config from {base_config}")
        conf = OmegaConf.load(base_config)
    else:
        print("[WARN] Base config not found, creating from scratch.")
        conf = OmegaConf.create({})

    # Patch the data section to point at our Cityscapes data
    if "data" in conf and "params" in conf.data and "train" in conf.data.params:
        conf.data.params.train.params.data_root = data_dir_abs
        # Keep canny control mode as-is (DummyBase supports: canny, midas, image)
    else:
        # Build a minimal data section matching the repo's expected structure
        conf.data = OmegaConf.create({
            "target": "main.DataModuleFromConfig",
            "params": {
                "batch_size": 1,
                "num_workers": 2,
                "wrap": False,
                "train": {
                    "target": "ldm.data.dummy_set.DummyBase",
                    "params": {
                        "size": 512,
                        "data_root": data_dir_abs,
                        "control_mode": "canny",
                    },
                },
            },
        })

    # Patch sync_path: download SD1.5 checkpoint if not present
    if "model" in conf and "params" in conf.model:
        sync_path = conf.model.params.get("sync_path", "")
        if not sync_path or "/PATH/" in sync_path or not os.path.exists(sync_path):
            ckpt_dir = os.path.abspath("models/sd15")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "v1-5-pruned.ckpt")
            if not os.path.exists(ckpt_path):
                print("\n--- Downloading SD1.5 checkpoint (required for ControlNet-XS training) ---")
                try:
                    from huggingface_hub import hf_hub_download
                    ckpt_path = hf_hub_download(
                        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                        filename="v1-5-pruned.ckpt",
                        local_dir=ckpt_dir,
                    )
                except Exception as e:
                    print(f"[WARN] Auto-download failed: {e}")
                    print("Please manually download v1-5-pruned.ckpt and place it in models/sd15/")
            conf.model.params.sync_path = os.path.abspath(ckpt_path)
            print(f"Set sync_path to {conf.model.params.sync_path}")

    # Fix deprecated lightning config values for PL >= 2.0
    if "lightning" in conf:
        trainer_conf = conf.lightning.get("trainer", OmegaConf.create({}))
        # 'accelerator: ddp' was replaced by strategy='ddp' in PL 2.x
        if "accelerator" in trainer_conf:
            acc = trainer_conf.accelerator
            if acc == "ddp":
                trainer_conf.accelerator = "gpu"
                # Do NOT set strategy="ddp" here, main.py explicitly configures it
                # via strategy=DDPStrategy(find_unused_parameters=False)
        # Remove deprecated 'gpus' key — PL 2.x uses 'devices'
        if "gpus" in trainer_conf:
            gpus_val = trainer_conf.pop("gpus")
            if "devices" not in trainer_conf:
                trainer_conf.devices = gpus_val
                
        # Handle custom epochs from CLI
        if args.epochs is not None:
            trainer_conf.max_epochs = args.epochs
            # Remove max_steps so PL relies on max_epochs
            trainer_conf.pop("max_steps", None)
            
        conf.lightning.trainer = trainer_conf

    OmegaConf.save(conf, yaml_config_path)
    print(f"Config saved to {yaml_config_path} with data_root={data_dir_abs}")

    # --- Step 5: Launch training ---
    print(f"\n--- Starting ControlNet-XS Training ---")
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"

    # main.py CLI:  python main.py -t --base <config> [--name <name>] [--logdir <dir>]
    # The trainer 'devices' and 'accelerator' are set inside main.py automatically.
    training_cmd = [
        sys.executable, "main.py",
        "-t",
        "--base", os.path.abspath(yaml_config_path),
        "--logdir", "logs",
        "--name", "cityscapes_controlnet_xs",
        "--scale_lr", "False",
    ]

    subprocess.run(training_cmd, cwd=repo_dir, env=env, check=True)

if __name__ == "__main__":
    main()
