#!/usr/bin/env python3
"""
Integrated Execution Pipeline for Sim-Lingo.
Runs inference (Action & Text modes) and 5 visualization methods,
organizing outputs into a structured directory hierarchy.
"""

import argparse
import datetime
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Import visualization runners
from experiment.ours import GenericAttentionActionVisualizer
from experiment.generic_attention_baseline import GenericAttentionTextVisualizer
from experiment.vit_attention_rollout import VisionAttentionRollout
from experiment.vit_attention_flow import VisionAttentionFlow
from experiment.vit_raw_attention import VisionRawAttention


def run_command(cmd: list[str], cwd: Optional[Path] = None) -> None:
    print(f"[Pipeline] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        print(f"[Pipeline] Command failed with return code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Sim-Lingo Integrated Pipeline")
    parser.add_argument("scenario_path", type=Path, help="Path to the scenario directory (e.g., data/sample_small/01)")
    args = parser.parse_args()

    scenario_path = args.scenario_path.resolve()
    if not scenario_path.exists():
        print(f"Error: Scenario path {scenario_path} does not exist.")
        sys.exit(1)

    scenario_name = scenario_path.name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path("experiment_outputs/integrated") / f"{scenario_name}_{timestamp}"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Pipeline] Starting integrated pipeline for {scenario_name}")
    print(f"[Pipeline] Output directory: {base_output_dir}")

    # Define temporary directories for inference outputs
    temp_action_dir = base_output_dir / "temp_inference_action"
    temp_text_dir = base_output_dir / "temp_inference_text"

    # 1. Run Inference (Action Mode)
    print("\n[Pipeline] Step 1: Running Inference (Action Mode)...")
    cmd_action = [
        sys.executable, "-m", "experiment.simlingo_inference_baseline",
        "--scene_dir", str(scenario_path),
        "--output_dir", str(temp_action_dir),
        "--target_mode", "auto",
        "--explain_mode", "action",
        "--kinematic_metric", "curv_energy",
        "--image_size", "224",
        "--max_patches", "2"
    ]
    run_command(cmd_action)

    # 2. Run Inference (Text Mode)
    print("\n[Pipeline] Step 2: Running Inference (Text Mode)...")
    cmd_text = [
        sys.executable, "-m", "experiment.simlingo_inference_baseline",
        "--scene_dir", str(scenario_path),
        "--output_dir", str(temp_text_dir),
        "--target_mode", "auto",
        "--explain_mode", "text",
        "--text_token_strategy", "max",
        "--text_token_index", "-1",
        "--image_size", "224",
        "--max_patches", "2"
    ]
    run_command(cmd_text)

    # Helper to find the actual output directory (simlingo creates a subdir based on config)
    def find_inference_subdir(base_dir: Path) -> Path:
        # It usually creates a subdir like 'data_sample_small_01_action_...'
        # We take the most recent one or the only one
        subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if not subdirs:
            raise RuntimeError(f"No output subdirectory found in {base_dir}")
        # Sort by modification time just in case
        subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return subdirs[0]

    action_inference_dir = find_inference_subdir(temp_action_dir)
    text_inference_dir = find_inference_subdir(temp_text_dir)
    
    print(f"[Pipeline] Action inference output: {action_inference_dir}")
    print(f"[Pipeline] Text inference output: {text_inference_dir}")

    # Define methods and their configs
    methods = [
        {
            "name": "ours",
            "runner_cls": GenericAttentionActionVisualizer,
            "inference_dir": action_inference_dir,
            "kwargs": {"colormap": "JET", "alpha": 0.5}
        },
        {
            "name": "generic_text",
            "runner_cls": GenericAttentionTextVisualizer,
            "inference_dir": text_inference_dir,
            "kwargs": {"text_token_strategy": "max", "colormap": "JET", "alpha": 0.5}
        },
        {
            "name": "vit_rollout",
            "runner_cls": VisionAttentionRollout,
            "inference_dir": action_inference_dir,
            "kwargs": {"start_layer": 0, "residual_alpha": 0.5, "colormap": "JET", "alpha": 0.5}
        },
        {
            "name": "vit_flow",
            "runner_cls": VisionAttentionFlow,
            "inference_dir": action_inference_dir,
            "kwargs": {"discard_ratio": 0.9, "residual_alpha": 0.5, "colormap": "JET", "alpha": 0.5}
        },
        {
            "name": "vit_raw",
            "runner_cls": VisionRawAttention,
            "inference_dir": action_inference_dir,
            "kwargs": {"layer_index": -1, "head_strategy": "mean", "colormap": "JET", "alpha": 0.5}
        }
    ]

    # 3. Run Visualizations and Organize
    print("\n[Pipeline] Step 3: Running Visualizations...")
    
    for method in methods:
        name = method["name"]
        print(f"\n[Pipeline] Processing method: {name}")
        
        method_dir = base_output_dir / name
        pt_dir = method_dir / "pt"
        raw_dir = method_dir / "raw_heatmap"
        final_dir = method_dir / "final_heatmap"
        
        method_dir.mkdir(parents=True, exist_ok=True)
        pt_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        # Copy PT files
        src_pt_dir = method["inference_dir"] / "pt"
        if src_pt_dir.exists():
            print(f"[Pipeline] Copying PT files from {src_pt_dir} to {pt_dir}")
            for pt_file in src_pt_dir.glob("*.pt"):
                shutil.copy2(pt_file, pt_dir / pt_file.name)
        else:
            print(f"[Pipeline] Warning: No PT directory found at {src_pt_dir}")

        # Instantiate Runner
        # Note: We pass the COPIED pt_dir as payload_root (or scene_dir)
        # But wait, the runners expect payload_root to be the directory containing .pt files
        # AND they might need images.
        # The images are in the original scenario_path.
        # We can pass scene_dir=scenario_path and payload_root=pt_dir.
        
        runner_cls = method["runner_cls"]
        kwargs = method["kwargs"]
        
        # Initialize runner
        # Note: Some runners might have different init args, but we standardized payload_root/scene_dir handling somewhat.
        # However, init usually takes config_path, device, etc.
        # We rely on defaults for config_path.
        
        try:
            runner = runner_cls(
                scene_dir=scenario_path, # Pass scene_dir to init if supported (ViT scripts support it now)
                payload_root=pt_dir,     # Use the copied PT dir
                **kwargs
            )
        except TypeError:
            # Fallback for Generic runners which might not take scene_dir in init (they take it in generate)
            # ours/generic init: (config_path, device, ..., payload_root)
            runner = runner_cls(
                payload_root=pt_dir,
                **kwargs
            )

        # Run generation
        # generate_scene_heatmaps(scene_dir, output_dir, suffix, raw_output_dir)
        # We pass final_dir as output_dir, and raw_dir as raw_output_dir.
        # scene_dir is scenario_path (where images are).
        
        runner.generate_scene_heatmaps(
            scene_dir=scenario_path,
            output_dir=final_dir,
            suffix=name,
            raw_output_dir=raw_dir
        )
        
        # Cleanup: The runners create a subdirectory inside output_dir based on scene name.
        # We want the files directly in final_dir/raw_dir?
        # The user said: "each method directory is divided into pt directory, original heatmap directory and final result heatmap directory. Within these three directories, pt files or images have the same name as the images in the input scenario."
        # Currently, `generate_scene_heatmaps` creates a subdirectory `ScenarioName_suffix`.
        # We should move the files up and remove the subdir.
        
        def flatten_output(target_dir: Path):
            # Find the subdir created by runner
            # It usually starts with scenario_name
            for subdir in target_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith(scenario_name):
                    print(f"[Pipeline] Flattening {subdir} to {target_dir}")
                    for f in subdir.glob("*"):
                        shutil.move(str(f), str(target_dir / f.name))
                    subdir.rmdir()
        
        flatten_output(final_dir)
        flatten_output(raw_dir)
        
        # Rename files to match original image names (remove suffix)
        # e.g. 00001_ours.png -> 00001.png
        def rename_files(target_dir: Path, suffix_str: str):
            for f in target_dir.glob(f"*_{suffix_str}.png"):
                new_name = f.name.replace(f"_{suffix_str}", "")
                # Also remove _raw if present (for raw dir)
                # But raw files are saved as {stem}.png in my refactoring!
                # Wait, let's check my refactoring.
                # ours.py: raw_path = raw_output_dir / f"{image_path.stem}.png" -> This is already correct!
                # But final output: output_path = output_dir / f"{image_path.stem}_{suffix}.png"
                
                # So for final_dir, we need to rename.
                # For raw_dir, it should be fine if I implemented it correctly.
                
                if f.name != new_name:
                    shutil.move(str(f), str(target_dir / new_name))

        rename_files(final_dir, name)
        # Check raw dir files
        # In my refactoring: raw_path = raw_output_dir / f"{image_path.stem}.png"
        # So raw files should already be named correctly.
        
        # However, `_prepare_output_subdir` was used for raw dir too in my refactoring?
        # Yes: scenario_raw_output_dir = self._prepare_output_subdir(raw_output_dir, scene_dir, "raw")
        # So raw files are in `raw_heatmap/ScenarioName_raw/`.
        # And they are named `{stem}.png`.
        # So I just need to flatten raw_dir.

    # Cleanup temp dirs
    print("\n[Pipeline] Cleaning up temporary directories...")
    shutil.rmtree(temp_action_dir)
    shutil.rmtree(temp_text_dir)

    print(f"\n[Pipeline] Done! Results saved to {base_output_dir}")


if __name__ == "__main__":
    main()
