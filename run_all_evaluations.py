import os
import subprocess
import argparse

def run_evaluation(name, model_id, controlnet_id, subfolder, output_dir, eval_data_dir):
    print(f"\n==================================================================")
    print(f"Starting Pipeline for: {name}")
    print(f"==================================================================")
    
    gen_cmd = [
        "python", "generate_samples.py",
        f"--model_id={model_id}",
        f"--output_dir={output_dir}",
        f"--eval_data_dir={eval_data_dir}"
    ]
    if controlnet_id:
        gen_cmd.append(f"--controlnet_id={controlnet_id}")
    if subfolder:
        gen_cmd.append(f"--subfolder={subfolder}")
        
    subprocess.run(gen_cmd)
    
    print(f"\nEvaluating Metrics for {name}...")
    eval_cmd = [
        "python", "evaluate_metrics.py",
        f"--gen_dir={output_dir}",
        f"--eval_dir={eval_data_dir}"
    ]
    subprocess.run(eval_cmd)

def main():
    parser = argparse.ArgumentParser(description="Evaluate All Configured Models against Eval Set")
    parser.add_argument("--eval_data_dir", default="data/eval", help="Path to evaluation dataset")
    parser.add_argument("--results_dir", default="results", help="Root directory to save generated outputs")
    args = parser.parse_args()

    sd15_id = "runwayml/stable-diffusion-v1-5"
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 1. Baseline A: Prompt-Only
    run_evaluation(
        name="Baseline A (Prompt-Only SD 1.5)",
        model_id=sd15_id,
        controlnet_id=None,
        subfolder=None,
        output_dir=os.path.join(args.results_dir, "baseline_a_prompt_only"),
        eval_data_dir=args.eval_data_dir
    )
    
    # 2. Public Substitute Baseline (Doguilmak)
    run_evaluation(
        name="Public Model Substitute (Doguilmak ControlNet)",
        model_id=sd15_id,
        controlnet_id="doguilmak/cityscapes-controlnet-sd15",
        subfolder="full_pipeline/controlnet",
        output_dir=os.path.join(args.results_dir, "public_substitute_doguilmak"),
        eval_data_dir=args.eval_data_dir
    )
    
    # 3. Custom Standard ControlNet (If trained)
    custom_standard = "models/standard_controlnet"
    if os.path.exists(custom_standard):
        run_evaluation(
            name="Baseline B (Trained Standard ControlNet)",
            model_id=sd15_id,
            controlnet_id=custom_standard,
            subfolder=None,
            output_dir=os.path.join(args.results_dir, "baseline_b_standard"),
            eval_data_dir=args.eval_data_dir
        )
    
    print(f"\nAll evaluations complete. Check `evaluation_results.txt` for the compiled report.")

if __name__ == "__main__":
    main()
