#!/usr/bin/env python3
"""
Integrated Execution Pipeline for Sim-Lingo (Batch Processing).
Runs inference (Action & Text modes) and 5 visualization methods in batches of 100 frames.
"""

import argparse
import datetime
import shutil
import sys
import gc
from pathlib import Path
from typing import List, Dict, Any
import torch
from tqdm import tqdm

# Add project root to sys.path to allow imports from experiment module
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import runners directly
from experiment.simlingo_inference_baseline import SimLingoInferenceBaseline
from experiment.ours import GenericAttentionActionVisualizer
from experiment.generic_attention_baseline import GenericAttentionTextVisualizer
from experiment.vit_attention_rollout import VisionAttentionRollout
from experiment.vit_attention_flow import VisionAttentionFlow
from experiment.vit_raw_attention import VisionRawAttention


def get_image_paths(scene_dir: Path, frames_subdir: str = "video_garmin") -> List[Path]:
    # Try frames subdir first
    candidate = scene_dir / frames_subdir
    if not candidate.exists():
        candidate = scene_dir / "images"
    
    if not candidate.exists():
        raise FileNotFoundError(f"No images found in {scene_dir} (checked {frames_subdir} and images)")
        
    return sorted([p for p in candidate.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def chunk_list(data: List[Any], chunk_size: int):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def main():
    parser = argparse.ArgumentParser(description="Sim-Lingo Integrated Pipeline (Batch)")
    parser.add_argument("scenario_path", type=Path, help="Path to the scenario directory")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of frames per batch")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    scenario_path = args.scenario_path.resolve()
    if not scenario_path.exists():
        print(f"Error: Scenario path {scenario_path} does not exist.")
        sys.exit(1)

    scenario_name = scenario_path.name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path("experiment_outputs/integrated") / f"{scenario_name}_{timestamp}"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Pipeline] Starting batch pipeline for {scenario_name}")
    print(f"[Pipeline] Output directory: {base_output_dir}")

    # 1. Initialize Inference Runner (Shared)
    # We will toggle explain_mode between 'action' and 'text'
    print("[Pipeline] Initializing Inference Model...")
    inference_runner = SimLingoInferenceBaseline(
        device=args.device,
        target_mode="auto",
        kinematic_metric="curv_energy",
        image_size=224,
        max_patches=2,
        text_token_strategy="max", # for text mode
        text_token_index=-1,
        skip_backward=False
    )

    # Prepare persistent inference output directories
    inference_action_dir = base_output_dir / "inference_action"
    inference_text_dir = base_output_dir / "inference_text"
    
    # 2. Initialize Visualizers
    # They need payload_root. Since we generate payloads on the fly, we point them to the
    # expected location of PT files.
    # SimLingo creates subdirs: output_dir / f"{scene_name}_{mode}" / "pt"
    
    # We need to know the exact path SimLingo will use.
    # It uses `_prepare_output_subdir` which does: output_dir / f"{scene_dir.name}_{suffix}"
    # So:
    pt_action_root = inference_action_dir / f"{scenario_name}_action" / "pt"
    pt_text_root = inference_text_dir / f"{scenario_name}_text" / "pt"
    
    # Ensure these exist (or at least the parent) so visualizers don't crash on init if they check
    # Visualizers check for existence of payload_root in __init__
    print(f"[Pipeline] Pre-creating PT directories: {pt_action_root}, {pt_text_root}")
    pt_action_root.mkdir(parents=True, exist_ok=True)
    pt_text_root.mkdir(parents=True, exist_ok=True)
    
    print("[Pipeline] Initializing Visualizers...")
    # Group visualizers by mode dependency
    action_visualizers = [
        {
            "name": "ours",
            "runner": GenericAttentionActionVisualizer(
                device=args.device,
                colormap="JET",
                alpha=0.5,
                payload_root=pt_action_root # Will be populated
            ),
            "pt_source": pt_action_root
        },
        {
            "name": "vit_rollout",
            "runner": VisionAttentionRollout(
                device=args.device,
                start_layer=0,
                residual_alpha=0.5,
                colormap="JET",
                alpha=0.5,
                payload_root=pt_action_root # Use action PTs
            ),
            "pt_source": pt_action_root
        },
        {
            "name": "vit_flow",
            "runner": VisionAttentionFlow(
                device=args.device,
                discard_ratio=0.9,
                residual_alpha=0.5,
                colormap="JET",
                alpha=0.5,
                payload_root=pt_action_root
            ),
            "pt_source": pt_action_root
        },
        {
            "name": "vit_raw",
            "runner": VisionRawAttention(
                device=args.device,
                layer_index=-1,
                head_strategy="mean",
                colormap="JET",
                alpha=0.5,
                payload_root=pt_action_root
            ),
            "pt_source": pt_action_root
        }
    ]

    text_visualizers = [
        {
            "name": "generic_text",
            "runner": GenericAttentionTextVisualizer(
                device=args.device,
                colormap="JET",
                alpha=0.5,
                payload_root=pt_text_root
            ),
            "pt_source": pt_text_root
        }
    ]

    # 3. Batch Loop
    all_images = get_image_paths(scenario_path)
    print(f"[Pipeline] Found {len(all_images)} images. Batch size: {args.batch_size}")

    for batch_idx, batch_files in enumerate(chunk_list(all_images, args.batch_size)):
        print(f"\n[Pipeline] === Processing Batch {batch_idx + 1} ({len(batch_files)} frames) ===")
        
        # Helper function to run visualizers
        def run_visualizers_for_batch(viz_list, batch_files):
            for viz in viz_list:
                name = viz["name"]
                runner = viz["runner"]
                pt_source = viz["pt_source"]
                
                print(f"[Pipeline]   Method: {name}")
                
                method_dir = base_output_dir / name
                pt_dir = method_dir / "pt"
                raw_dir = method_dir / "raw_heatmap"
                final_dir = method_dir / "final_heatmap"
                
                pt_dir.mkdir(parents=True, exist_ok=True)
                raw_dir.mkdir(parents=True, exist_ok=True)
                final_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy PT files for this batch to the method's PT dir
                for img_file in batch_files:
                    pt_file = pt_source / f"{img_file.stem}.pt"
                    if pt_file.exists():
                        shutil.copy2(pt_file, pt_dir / pt_file.name)
                
                # Update runner's payload_root and re-index
                runner.payload_root = pt_dir
                if hasattr(runner, "_index_payloads"):
                    runner._payload_index = runner._index_payloads(pt_dir)
                
                # Run generation for this batch
                runner.generate_scene_heatmaps(
                    scene_dir=scenario_path,
                    output_dir=final_dir,
                    suffix=name,
                    raw_output_dir=raw_dir,
                    target_files=batch_files
                )
                
                # Flatten output directories
                def flatten_and_rename(target_dir: Path, suffix_str: str):
                    for subdir in target_dir.iterdir():
                        if subdir.is_dir() and subdir.name.startswith(scenario_name):
                            for f in subdir.glob("*"):
                                new_name = f.name
                                if f.name.endswith(f"_{suffix_str}.png"):
                                    new_name = f.name.replace(f"_{suffix_str}.png", ".png")
                                dest = target_dir / new_name
                                if not dest.exists():
                                    shutil.move(str(f), str(dest))
                            subdir.rmdir()

                flatten_and_rename(final_dir, name)
                flatten_and_rename(raw_dir, "raw")
                
                # Explicit cleanup
                gc.collect()
                torch.cuda.empty_cache()

        # --- Inference Action ---
        print("[Pipeline] Running Inference (Action)...")
        inference_runner.explain_mode = "action"
        inference_runner.run_batch(batch_files, inference_action_dir, scenario_path)
        
        # --- Visualizations (Action) ---
        print("[Pipeline] Running Visualizations (Action-dependent)...")
        run_visualizers_for_batch(action_visualizers, batch_files)
        
        # --- Inference Text ---
        print("[Pipeline] Running Inference (Text)...")
        inference_runner.explain_mode = "text"
        inference_runner.run_batch(batch_files, inference_text_dir, scenario_path)
        
        # --- Visualizations (Text) ---
        print("[Pipeline] Running Visualizations (Text-dependent)...")
        run_visualizers_for_batch(text_visualizers, batch_files)

    print(f"\n[Pipeline] Done! Results saved to {base_output_dir}")
    
    # Cleanup persistent inference dirs if desired?
    # User might want to inspect them. Let's keep them or delete?
    # "Organize all generated outputs... into a structured directory hierarchy under experiment_outputs/integrated/"
    # The user didn't explicitly ask to delete intermediate inference outputs, but they are redundant if copied.
    # I'll leave them for debugging or delete them to save space.
    # Given OOM concerns, maybe delete? But disk space is usually cheap.
    # I'll leave them but print a message.
    print(f"[Pipeline] Intermediate inference outputs kept in {inference_action_dir} and {inference_text_dir}")


if __name__ == "__main__":
    main()
