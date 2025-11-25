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
from experiment.simlingo_vit_qwen_attn_grad import SimLingoInferenceBaseline
from experiment.ours import GenericAttentionActionVisualizer
from experiment.generic_attention_baseline import GenericAttentionTextVisualizer
from experiment.vit_attention_rollout import VisionAttentionRollout
from experiment.vit_attention_flow import VisionAttentionFlow
from experiment.vit_raw_attention import VisionRawAttention


def summarize_pt_dir(pt_dir: Path, log_path: Path, label: str) -> None:
    """Summarize presence of attention/interleaver fields in PT payloads for later verification."""
    if not pt_dir.exists():
        return
    pt_files = sorted([p for p in pt_dir.iterdir() if p.suffix == ".pt"])
    if not pt_files:
        return
    total = len(pt_files)
    attn_ok = inter_ok = mlp1_grad_ok = pix_grad_ok = 0
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"[SUMMARY-{label}] PT dir: {pt_dir}, count={total}\n")
        for pt in pt_files:
            try:
                payload = torch.load(pt, map_location="cpu")
            except Exception as exc:
                lf.write(f"  {pt.name}: load_failed ({exc})\n")
                continue
            attention = payload.get("attention")
            inter = payload.get("interleaver")
            if attention:
                attn_ok += 1
            if inter:
                inter_ok += 1
                mlp1 = inter.get("mlp1_output") if isinstance(inter, dict) else None
                pix = inter.get("pixel_shuffle_out") if isinstance(inter, dict) else None
                if isinstance(mlp1, dict) and mlp1.get("grad") is not None:
                    mlp1_grad_ok += 1
                if isinstance(pix, dict) and pix.get("grad") is not None:
                    pix_grad_ok += 1
                tokens_per_patch = inter.get("tokens_per_patch")
                num_patches_list = inter.get("num_patches_list")
                lf.write(
                    f"  {pt.name}: interleaver tokens_per_patch={tokens_per_patch}, num_patches_list={num_patches_list}, "
                    f"mlp1_grad={'yes' if mlp1_grad_ok else 'no'}, pixel_shuffle_grad={'yes' if pix_grad_ok else 'no'}\n"
                )
            else:
                lf.write(f"  {pt.name}: interleaver missing\n")
        lf.write(
            f"[SUMMARY-{label}] attention {attn_ok}/{total}, interleaver {inter_ok}/{total}, "
            f"mlp1_grad {mlp1_grad_ok}/{total}, pixel_shuffle_grad {pix_grad_ok}/{total}\n"
        )

class _PipelineTee:
    """간단한 tee: stdout/stderr를 파일과 콘솔에 동시에 기록."""

    def __init__(self, log_path: Path):
        self.log_file = open(log_path, "a", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def write(self, data):
        self._stdout.write(data)
        self.log_file.write(data)

    def flush(self):
        self._stdout.flush()
        self.log_file.flush()

    def close(self):
        try:
            self.log_file.close()
        except Exception:
            pass
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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


def process_scenario(scenario_path: Path, batch_size: int, device: str, base_output_root: Path, delete_pt: bool = False):
    """Process a single scenario through the integrated pipeline."""
    scenario_name = scenario_path.name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = base_output_root / f"{scenario_name}_{timestamp}"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = base_output_dir / "pipeline.log"

    with _PipelineTee(log_file):
        print(f"\n{'='*80}")
        print(f"[Pipeline] Starting batch pipeline for {scenario_name}")
        print(f"[Pipeline] Output directory: {base_output_dir}")
        print(f"[Pipeline] Log file: {log_file}")
        print(f"{'='*80}\n")

        # 1. Initialize Inference Runner (Shared)
        # We will toggle explain_mode between 'action' and 'text'
        print("[Pipeline] Initializing Inference Model...")
        inference_runner = SimLingoInferenceBaseline(
            device=device,
            target_mode="auto",
            kinematic_metric="curv_energy",
            image_size=224,
            max_patches=2,
            text_token_strategy="max", # for text mode
            text_token_index=-1,
            skip_backward=False
        )

        # Pre-create PT directories for inference
        inference_action_dir = base_output_dir / "inference_action"
        inference_text_dir = base_output_dir / "inference_text"
        
        action_pt_dir = inference_action_dir / f"{scenario_name}_action" / "pt"
        text_pt_dir = inference_text_dir / f"{scenario_name}_text" / "pt"
        
        action_pt_dir.mkdir(parents=True, exist_ok=True)
        text_pt_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[Pipeline] Pre-creating PT directories: {action_pt_dir}, {text_pt_dir}")

        # 2. Initialize Visualizers
        print("[Pipeline] Initializing Visualizers...")
        
        action_visualizers = [
            {
                "name": "ours",
                "runner": GenericAttentionActionVisualizer(
                    device=device,
                    colormap="JET",
                    alpha=0.5,
                    trajectory_overlay_root=None,
                    payload_root=action_pt_dir
                ),
                "pt_source": action_pt_dir,
            },
        ]

        vit_visualizers = [
            {
                "name": "vit_rollout",
                "runner": VisionAttentionRollout(
                    device=device,
                    start_layer=0,
                    residual_alpha=0.5,
                    colormap="JET",
                    alpha=0.5,
                    trajectory_overlay_root=None,
                    payload_root=action_pt_dir
                ),
                "pt_source": action_pt_dir,
            },
            {
                "name": "vit_flow",
                "runner": VisionAttentionFlow(
                    device=device,
                    discard_ratio=0.9,
                    residual_alpha=0.5,
                    colormap="JET",
                    alpha=0.5,
                    trajectory_overlay_root=None,
                    payload_root=action_pt_dir
                ),
                "pt_source": action_pt_dir,
            },
            {
                "name": "vit_raw",
                "runner": VisionRawAttention(
                    device=device,
                    layer_index=-1,
                    head_strategy="mean",
                    colormap="JET",
                    alpha=0.5,
                    trajectory_overlay_root=None,
                    payload_root=action_pt_dir
                ),
                "pt_source": action_pt_dir,
            },
        ]

        text_visualizers = [
            {
                "name": "generic_text",
                "runner": GenericAttentionTextVisualizer(
                    device=device,
                    text_token_strategy="max",
                    text_token_index=-1,
                    colormap="JET",
                    alpha=0.5,
                    trajectory_overlay_root=None,
                    payload_root=text_pt_dir
                ),
                "pt_source": text_pt_dir,
            },
        ]

        # 3. Get image paths
        image_paths = get_image_paths(scenario_path)
        print(f"[Pipeline] Found {len(image_paths)} images. Batch size: {batch_size}\n")

        # 4. Process batches
        batches = list(chunk_list(image_paths, batch_size))
        
        for batch_idx, batch_files in enumerate(batches):
            print(f"[Pipeline] === Processing Batch {batch_idx + 1} ({len(batch_files)} frames) ===")
            
            # Helper function to run visualizers
            def run_visualizers_for_batch(visualizers: List[Dict], batch_files: List[Path]):
                for viz in visualizers:
                    name = viz["name"]
                    runner = viz["runner"]
                    pt_source = viz["pt_source"]
                    
                    print(f"[Pipeline]   Method: {name}")
                    
                    method_dir = base_output_dir / name
                    method_dir.mkdir(parents=True, exist_ok=True)
                    
                    raw_dir = method_dir / "raw_heatmap"
                    final_dir = method_dir / "final_heatmap"
                    
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    final_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Point runner directly to source PT directory (no copying)
                    runner.payload_root = pt_source
                    if hasattr(runner, "_index_payloads"):
                        runner._payload_index = runner._index_payloads(pt_source)
                    
                    # Run generation for this batch - pass method_dir as output_dir
                    # The visualizer will create pt_log.txt there and save to raw_dir/final_dir
                    runner.generate_scene_heatmaps(
                        scene_dir=scenario_path,
                        output_dir=method_dir,  # Changed: pass method_dir instead of final_dir
                        suffix=name,
                        raw_output_dir=raw_dir,
                        final_output_dir=final_dir,  # Added: explicit final_dir
                        target_files=batch_files
                    )
                    
                    # Explicit cleanup
                    gc.collect()
                    torch.cuda.empty_cache()

        # --- Inference Action ---
        print("[Pipeline] Running Inference (Action)...")
        inference_runner.explain_mode = "action"
        # Capture the actual output directory (which includes timestamp)
        actual_action_dir = inference_runner.run_batch(batch_files, inference_action_dir, scenario_path)
        # Log PT summary for action pass
        summarize_pt_dir(actual_action_dir / "pt", log_file, label="action")
            
        # Update pt_source for action visualizers
        # The PT files are in actual_action_dir / "pt"
        pt_action_source = actual_action_dir / "pt"
        for viz in action_visualizers:
            viz["pt_source"] = pt_action_source
        for viz in vit_visualizers:
            viz["pt_source"] = pt_action_source

        # --- Visualizations (Action) ---
        print(f"[Pipeline] Running Visualizations (Action-dependent)... PT Source: {pt_action_source}")
        run_visualizers_for_batch(action_visualizers + vit_visualizers, batch_files)
            
        # --- Inference Text ---
        print("[Pipeline] Running Inference (Text)...")
        inference_runner.explain_mode = "text"
        actual_text_dir = inference_runner.run_batch(batch_files, inference_text_dir, scenario_path)
        summarize_pt_dir(actual_text_dir / "pt", log_file, label="text")
            
        # Update pt_source for text visualizers
        pt_text_source = actual_text_dir / "pt"
        for viz in text_visualizers:
            viz["pt_source"] = pt_text_source
            
        # --- Visualizations (Text) ---
        print(f"[Pipeline] Running Visualizations (Text-dependent)... PT Source: {pt_text_source}")
        run_visualizers_for_batch(text_visualizers, batch_files)
        
        # --- Cleanup PT Files ---
        if delete_pt:
            print(f"[Pipeline] Cleaning up PT files for batch {batch_idx + 1}...")
            if pt_action_source.exists():
                shutil.rmtree(pt_action_source)
                print(f"[Pipeline]   Deleted: {pt_action_source}")
            if pt_text_source.exists():
                shutil.rmtree(pt_text_source)
                print(f"[Pipeline]   Deleted: {pt_text_source}")

        print(f"\n[Pipeline] Done! Results saved to {base_output_dir}")
        print(f"[Pipeline] Intermediate inference outputs kept in {inference_action_dir} and {inference_text_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sim-Lingo Integrated Pipeline (Batch)")
    parser.add_argument("input_path", type=Path, help="Path to scenario directory or dataset directory")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of frames per batch")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--delete_pt", action="store_true", help="Delete intermediate PT payloads after visualizations")
    args = parser.parse_args()

    input_path = args.input_path.resolve()
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist.")
        sys.exit(1)

    base_output_root = Path("experiment_outputs/integrated")
    base_output_root.mkdir(parents=True, exist_ok=True)

    # Check if input_path is a dataset directory (contains subdirectories) or a single scenario
    # Heuristic: if it contains subdirectories that look like scenarios, process all
    # Otherwise, treat it as a single scenario
    
    potential_scenarios = []
    if input_path.is_dir():
        # Check for subdirectories that could be scenarios
        subdirs = [d for d in input_path.iterdir() if d.is_dir()]
        
        # Try to detect if this is a dataset directory:
        # If subdirectories contain image folders (video_garmin, images, etc.), they are scenarios
        for subdir in subdirs:
            # Check if subdir has image folders
            if (subdir / "video_garmin").exists() or (subdir / "images").exists() or (subdir / "input_images").exists():
                potential_scenarios.append(subdir)
        
        # If we found scenarios, process all of them
        if potential_scenarios:
            # Create dataset-level output directory
            dataset_name = input_path.name
            dataset_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_output_root = base_output_root / f"{dataset_name}_{dataset_timestamp}"
            dataset_output_root.mkdir(parents=True, exist_ok=True)
            
            print(f"[Pipeline] Detected dataset directory: {dataset_name}")
            print(f"[Pipeline] Found {len(potential_scenarios)} scenarios")
            print(f"[Pipeline] Scenarios: {[s.name for s in potential_scenarios]}")
            print(f"[Pipeline] Dataset output directory: {dataset_output_root}\n")
            
            for scenario in sorted(potential_scenarios):
                process_scenario(scenario, args.batch_size, args.device, dataset_output_root, delete_pt=args.delete_pt)
        else:
            # Treat input_path itself as a scenario
            print(f"[Pipeline] Processing single scenario: {input_path.name}")
            process_scenario(input_path, args.batch_size, args.device, base_output_root, delete_pt=args.delete_pt)
    else:
        print(f"Error: Input path {input_path} is not a directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
