#!/usr/bin/env python3
"""Sim-Lingo Direct Visualization (Alternative Implementation).

This script implements a direct visualization pipeline for Sim-Lingo, bypassing
intermediate .pt file storage. It runs forward and backward passes in-memory
to generate attention heatmaps and trajectory projections using methods like
Generic Attention, Rollout, and Raw Attention.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoProcessor

# Add external/simlingo to path
REPO_ROOT = Path(__file__).resolve().parents[1]
SIMLINGO_SRC = REPO_ROOT / "external" / "simlingo"
if SIMLINGO_SRC.exists() and str(SIMLINGO_SRC) not in sys.path:
    sys.path.insert(0, str(SIMLINGO_SRC))

# Use locally patched InternVL2 module cache
LOCAL_HF_MODULES = REPO_ROOT / "experiment" / "InternVL2-1B"
if LOCAL_HF_MODULES.exists():
    os.environ["HF_MODULES_CACHE"] = str(LOCAL_HF_MODULES.resolve())

sys.path.append(str(REPO_ROOT))

from simlingo_training.models.driving import DrivingModel
from simlingo_training.models.encoder import internvl2_model as ivl
from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess
from team_code.simlingo_utils import get_camera_extrinsics, get_camera_intrinsics, project_points
from experiment.simlingo_patches import patch_simlingo

# Re-use kinematic metrics and spline logic from baseline (copied for self-containment or imported)
# For simplicity and stability, we will copy the essential parts here to avoid circular imports
# if we were to import from the experiment folder (which is not a package).

patch_simlingo()

# --- Constants & Configuration Defaults ---
DEFAULT_CONFIG_PATH = Path("checkpoints/simlingo/simlingo/.hydra/config.yaml")
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt")
DEFAULT_SCENE_DIR = Path("data/sample/01")
DEFAULT_OUTPUT_DIR = Path("experiment_outputs/simlingo_vis_alt")
DEFAULT_IMAGE_SIZE = 224
DEFAULT_MAX_PATCHES = 2
DEFAULT_FRAMES_SUBDIR = "video_garmin"
DEFAULT_SPEED_SUBDIR = "video_garmin_speed"

EPS = 1e-6
DELTA_T = 0.25

# --- Helper Functions (Kinematics & Spline) ---
# (Simplified versions of what was in simlingo_inference_baseline.py)

def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    return x.float()

def compute_curvature_energy(route: torch.Tensor) -> torch.Tensor:
    route = _ensure_batch(route)
    diffs = route[:, 1:, :] - route[:, :-1, :]
    seglen = diffs.norm(dim=-1).clamp_min(EPS)
    tangent = diffs / seglen.unsqueeze(-1)
    headings = torch.atan2(tangent[..., 1], tangent[..., 0])
    heading_diff = headings[:, 1:] - headings[:, :-1]
    ds = (seglen[:, 1:] + seglen[:, :-1]) * 0.5
    curvature = heading_diff / ds.clamp_min(EPS)
    return (curvature ** 2).sum()

def compute_acceleration_energy(speed_wps: torch.Tensor, delta_t: float = DELTA_T) -> torch.Tensor:
    speed_wps = _ensure_batch(speed_wps)
    diffs = speed_wps[:, 1:, :] - speed_wps[:, :-1, :]
    base_dir = diffs[:, :1, :]
    norm = base_dir.norm(dim=-1, keepdim=True).clamp_min(EPS)
    direction = base_dir / norm
    velocities = (diffs * direction).sum(dim=-1) / max(delta_t, EPS)
    accelerations = (velocities[:, 1:] - velocities[:, :-1]) / max(delta_t, EPS)
    return (accelerations ** 2).sum()

KINEMATIC_METRICS = {
    "curv_energy": {"source": "route", "fn": compute_curvature_energy},
    "acc_energy": {"source": "speed", "fn": compute_acceleration_energy},
}

# --- Visualization Logic ---

class SimLingoVisualizer:
    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        device: Optional[str] = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
        max_patches: int = DEFAULT_MAX_PATCHES,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.image_size = image_size
        self.max_patches = max_patches
        self.frames_subdir = DEFAULT_FRAMES_SUBDIR
        self.speed_subdir = DEFAULT_SPEED_SUBDIR
        
        self.cfg = OmegaConf.load(self.config_path)
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model.vision_model.variant, trust_remote_code=True
        )
        if hasattr(self.processor, "tokenizer"):
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor
            
        # Add special tokens
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<WAYPOINTS>", "<WAYPOINTS_DIFF>", "<ORG_WAYPOINTS_DIFF>",
                "<ORG_WAYPOINTS>", "<WAYPOINT_LAST>", "<ROUTE>",
                "<ROUTE_DIFF>", "<TARGET_POINT>",
            ]
        })
        self.tokenizer.padding_side = "left"
        self.transform = build_transform(input_size=self.image_size)
        self.T = 1
        self.num_image_token = self._compute_num_image_tokens(
            self.cfg.model.vision_model.variant, image_size_override=self.image_size
        )
        
        self.model = self._build_model()
        self._speed_cache: Dict[str, Dict[str, float]] = {}

    def _compute_num_image_tokens(self, encoder_variant: str, image_size_override: Optional[int] = None) -> int:
        cfg = AutoConfig.from_pretrained(encoder_variant, trust_remote_code=True)
        image_size = image_size_override or cfg.force_image_size or cfg.vision_config.image_size
        patch_size = cfg.vision_config.patch_size
        downsample = getattr(cfg, "downsample_ratio", getattr(cfg.vision_config, "downsample_ratio", 1.0))
        return int((image_size // patch_size) ** 2 * (downsample ** 2))

    def _build_model(self) -> DrivingModel:
        cache_dir = Path("pretrained") / self.cfg.model.vision_model.variant.split("/")[1]
        cache_dir = Path(to_absolute_path(str(cache_dir)))
        
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        
        model = hydra.utils.instantiate(
            self.cfg.model,
            cfg_data_module=self.cfg.data_module,
            processor=self.processor,
            cache_dir=str(cache_dir),
            _recursive_=False,
        ).to(self.device)
        
        torch.set_default_dtype(default_dtype)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Enable gradients for explanation
        for param in model.parameters():
            param.requires_grad = True
            
        # Verify requires_grad
        print(f"Model Parameter requires_grad check: {next(model.parameters()).requires_grad}")
            
        # Configure model to output attentions
        model.language_model.model.config.output_attentions = True
        model.vision_model.image_encoder.model.config.output_attentions = True
        
        # IMPORTANT: Disable gradient checkpointing to ensure we get gradients for all layers
        if hasattr(model.language_model.model, "gradient_checkpointing_disable"):
             model.language_model.model.gradient_checkpointing_disable()
        if hasattr(model.vision_model.image_encoder.model, "gradient_checkpointing_disable"):
             model.vision_model.image_encoder.model.gradient_checkpointing_disable()
             
        # Also manually check config
        model.language_model.model.config.gradient_checkpointing = False
        model.vision_model.image_encoder.model.config.gradient_checkpointing = False
        
        # Force Eager Attention (Disable Flash Attention)
        # Flash Attention kernels often do not compute gradients w.r.t attention weights (only Q,K,V).
        # We need vanilla attention to get d(Loss)/d(AttnWeights).
        print("Forcing Eager Attention Implementation to capture attention gradients...")
        if hasattr(model.language_model.model.config, "attn_implementation"):
            model.language_model.model.config.attn_implementation = "eager"
        if hasattr(model.vision_model.image_encoder.model.config, "attn_implementation"):
            model.vision_model.image_encoder.model.config.attn_implementation = "eager"
            
        # Some HF models use _attn_implementation
        if hasattr(model.language_model.model.config, "_attn_implementation"):
            model.language_model.model.config._attn_implementation = "eager"
        if hasattr(model.vision_model.image_encoder.model.config, "_attn_implementation"):
            model.vision_model.image_encoder.model.config._attn_implementation = "eager"
        
        # Force model to train mode to ensure gradients are tracked (sometimes eval disables specific hooks)
        # But we want deterministic behavior, so we can set eval() but ensure grad is enabled.
        # PyTorch eval() does NOT disable gradient calculation, torch.no_grad() does.
        # However, some custom models might have specific flags.
        model.eval() 

        return model

    def _preprocess_image(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        use_global_img = getattr(self.cfg.model.vision_model, "use_global_img", True)
        images = dynamic_preprocess(
            image,
            image_size=self.image_size,
            use_thumbnail=use_global_img,
            max_num=self.max_patches,
        )
        pixel_values = torch.stack([self.transform(img) for img in images])
        pixel_values = pixel_values.unsqueeze(0)
        num_patches = pixel_values.shape[1]
        C, H, W = pixel_values.shape[2], pixel_values.shape[3], pixel_values.shape[4]
        processed_image = pixel_values.view(1, self.T, num_patches, C, H, W)
        return processed_image, num_patches, (image.height, image.width)

    def _prepare_driving_input(self, image_path: Path, current_speed: float = 0.0) -> Tuple[DrivingInput, Dict[str, Any]]:
        processed_image, num_patches, orig_hw = self._preprocess_image(image_path)
        prompt_str = f"Current speed: {current_speed:.1f} m/s. What should the ego vehicle do next? Provide a short commentary."
        
        # Build dummy conversation for prompt
        conv_module = self._load_conversation_template_module()
        template = conv_module.get_conv_template("internlm2-chat")
        template.append_message(template.roles[0], "<image>\n" + prompt_str)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        image_tokens = "<img>" + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + "</img>"
        query = query.replace("<image>", image_tokens, 1)
        
        prompt_tokenized = self.tokenizer(
            [query], padding=True, return_tensors="pt", add_special_tokens=False
        )
        input_ids = prompt_tokenized["input_ids"].to(self.device)
        
        # Identify image token indices
        # We need the ID of IMG_CONTEXT_TOKEN
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        # input_ids is [1, SeqLen]
        seq_ids = input_ids[0].tolist()
        image_token_indices = [i for i, tid in enumerate(seq_ids) if tid == img_context_token_id]
        
        # Create DrivingInput
        camera_intrinsics = get_camera_intrinsics(orig_hw[1], orig_hw[0], 110).unsqueeze(0).unsqueeze(0)
        camera_extrinsics = get_camera_extrinsics().unsqueeze(0).unsqueeze(0)
        
        # Create a dummy LanguageLabel
        ll = LanguageLabel(
            phrase_ids=input_ids,
            phrase_valid=(input_ids != self.tokenizer.pad_token_id),
            phrase_mask=(input_ids != self.tokenizer.pad_token_id),
            placeholder_values=[[]],
            language_string=[query],
            loss_masking=None
        )

        driving_input = DrivingInput(
            camera_images=processed_image.to(self.device).bfloat16(),
            image_sizes=torch.tensor([[orig_hw[0], orig_hw[1]]], dtype=torch.float32).to(self.device),
            camera_intrinsics=camera_intrinsics.to(self.device),
            camera_extrinsics=camera_extrinsics.to(self.device),
            vehicle_speed=torch.tensor([[current_speed]], dtype=torch.float32, device=self.device),
            target_point=torch.zeros(1, 2, dtype=torch.float32, device=self.device),
            prompt=ll,
            prompt_inference=ll,
        )
        
        meta = {
            "original_height": orig_hw[0],
            "original_width": orig_hw[1],
            "num_patches": num_patches,
            "image_token_indices": image_token_indices,
            "seq_len": input_ids.shape[1]
        }
        return driving_input, meta

    def _load_conversation_template_module(self):
        cache_dir = Path("pretrained") / self.cfg.model.vision_model.variant.split("/")[1]
        cache_dir = Path(to_absolute_path(str(cache_dir)))
        model_path = cache_dir / "conversation.py"
        if not model_path.exists():
            # Fallback or error
            pass
        spec = importlib.util.spec_from_file_location("get_conv_template", model_path)
        conv_module = importlib.util.module_from_spec(spec)
        sys.modules["get_conv_template"] = conv_module
        spec.loader.exec_module(conv_module)
        return conv_module

    def _load_speed_table(self, speed_dir: Path) -> Dict[str, float]:
        key = str(speed_dir.resolve())
        if key in self._speed_cache:
            return self._speed_cache[key]
        table: Dict[str, float] = {}
        for txt in sorted(speed_dir.glob("*.txt")):
            try:
                val = float(txt.read_text().strip().split()[0])
                table[txt.stem] = val
            except Exception:
                continue
        self._speed_cache[key] = table
        return table

    def _register_hooks(self, method="all"):
        self.hooks = []
        self.attn_maps = {}
        self.grad_maps = {}

        def get_hook(name, save_grad=False):
            def hook(module, input, output):
                # Debug: Inspect output structure for the first layer
                if "layer_0" in name and "vision" in name:
                    # Print Module Structure to find inner attention
                    print(f"DEBUG: Hook {name} Module Type: {type(module)}")
                    # Only print structure once
                    if not hasattr(self, "_printed_structure"):
                        print(f"DEBUG: Module Structure:\n{str(module)}")
                        self._printed_structure = True
                        
                    print(f"DEBUG: Hook {name} Output Type: {type(output)}")
                    if isinstance(output, tuple):
                        print(f"DEBUG: Hook {name} Output Length: {len(output)}")
                        for idx, item in enumerate(output):
                            if hasattr(item, "shape"):
                                print(f"  - Item {idx} Shape: {item.shape}")
                            else:
                                print(f"  - Item {idx} Type: {type(item)}")
                
                # Try to extract attn
                if isinstance(output, tuple):
                    context = output[0]
                    if len(output) > 1:
                        attn = output[1] # [B, H, S, S]
                    else:
                        attn = None
                else:
                    context = output
                    attn = None
                
                # Debug: Check context gradients (Layer Output)
                if save_grad and hasattr(context, "requires_grad") and context.requires_grad:
                    def context_grad_hook(grad):
                        if "layer_0" in name and "vision" in name:
                            print(f"DEBUG: Hook {name} Context (Output) Grad Mean: {grad.float().mean():.6e}")
                    context.register_hook(context_grad_hook)

                if attn is None: 
                    if "layer_0" in name and "vision" in name:
                        print(f"DEBUG: Hook {name} - Attn is None!")
                    return

                # Debug: Check if attn is attached to graph
                if "layer_0" in name and "vision" in name:
                     print(f"Hook {name} captured attn. Shape: {attn.shape}, Requires Grad: {attn.requires_grad}")

                self.attn_maps[name] = attn.detach()
                
                if save_grad:
                    if attn.requires_grad:
                        def grad_hook(grad):
                            self.grad_maps[name] = grad.detach()
                            if "layer_0" in name and "vision" in name:
                                print(f"DEBUG: Hook {name} Attn Grad Mean: {grad.float().mean():.6e}")
                        attn.register_hook(grad_hook)
                    else:
                        if "layer_0" in name and "vision" in name:
                            print(f"WARNING: Hook {name} cannot register grad hook because attn.requires_grad is False")
            return hook

        # 1. Vision Encoder Hooks (for All methods)
        vision_model = self.model.vision_model.image_encoder.model.vision_model
        for i, layer in enumerate(vision_model.encoder.layers):
            # Hook the Layer/Block itself, not the attention submodule.
            # Standard HF layers return (hidden_states, attentions) if output_attentions=True.
            target = layer
            
            # Generic needs grad, others don't (but we capture grad if we run generic)
            save_grad = (method in ["generic", "all", "ours"])
            self.hooks.append(target.register_forward_hook(get_hook(f"vision_layer_{i}", save_grad=save_grad)))

        # 2. Language Model Hooks (For Ours/Generic if requested)
        if method in ["generic", "all", "ours"]:
            lm = self.model.language_model.model
            layers = getattr(lm, "layers", getattr(getattr(lm, "model", None), "layers", []))
            for i, layer in enumerate(layers):
                # Hook the Layer/Block itself
                target = layer
                
                self.hooks.append(target.register_forward_hook(get_hook(f"language_layer_{i}", save_grad=True)))

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    # --- Visualization Methods ---

    def generate_visualization(
        self, 
        image_path: Path, 
        output_dir: Path, 
        methods: List[str] = ["generic", "rollout", "raw", "flow", "ours"],
        metric: str = "curv_energy",
        current_speed: float = 0.0,
        explain_mode: str = "action" # "action" or "text"
    ):
        self.model.zero_grad()
        driving_input, meta = self._prepare_driving_input(image_path, current_speed)
        
        # Register Hooks
        run_generic_or_ours = "generic" in methods or "ours" in methods or "all" in methods
        self._register_hooks(method="all" if run_generic_or_ours else "vision")

        # Forward
        outputs = self.model(driving_input)
        pred_speed_wps, pred_route, _ = outputs
        
        # Debug: Check output gradients
        print(f"Pred Route Requires Grad: {pred_route.requires_grad}")
        if not pred_route.requires_grad:
            print("CRITICAL WARNING: Model output does not require gradients. Check model configuration.")
        
        # Backward (if needed for Generic/Ours)
        if run_generic_or_ours:
            if explain_mode == "action":
                metric_cfg = KINEMATIC_METRICS.get(metric, KINEMATIC_METRICS["curv_energy"])
                if metric_cfg["source"] == "route":
                    target = metric_cfg["fn"](pred_route)
                else:
                    target = metric_cfg["fn"](pred_speed_wps)
            else:
                # Text mode placeholder
                target = pred_route.sum() * 0 
                pass

            if target.requires_grad:
                self.model.zero_grad()
                target.backward()
                
                # Debug: Check if gradients are flowing
                print(f"Target Value: {target.item():.6f}, Requires Grad: {target.requires_grad}")
                if self.grad_maps:
                    first_key = list(self.grad_maps.keys())[0]
                    last_key = list(self.grad_maps.keys())[-1]
                    print(f"Grad Map [{first_key}] Mean: {self.grad_maps[first_key].float().mean():.6e}, Max: {self.grad_maps[first_key].max():.6e}")
                    print(f"Grad Map [{last_key}] Mean: {self.grad_maps[last_key].float().mean():.6e}, Max: {self.grad_maps[last_key].max():.6e}")
                else:
                    print("WARNING: No gradients captured in grad_maps!")
            else:
                print("CRITICAL: Target does not require grad, cannot backward.")
        
        # Generate Visualizations
        for method in methods:
            # Create method-specific subdirectory
            method_dir = output_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            
            if method == "generic":
                heatmap = self._get_generic_relevance(self.attn_maps, self.grad_maps)
            elif method == "rollout":
                heatmap = self._get_rollout_relevance(self.attn_maps)
            elif method == "flow":
                heatmap = self._get_flow_relevance(self.attn_maps)
            elif method == "raw":
                heatmap = self._get_raw_attention(self.attn_maps)
            elif method == "ours":
                heatmap = self._get_ours_relevance(self.attn_maps, self.grad_maps, meta)
            else:
                continue
            
            # Debug: Print heatmap stats
            if heatmap.numel() > 0:
                print(f"[{method}] Heatmap Stats - Min: {heatmap.min():.4f}, Max: {heatmap.max():.4f}, Mean: {heatmap.mean():.4f}")
                
            # 1. Create Heatmap Overlay (Original + Heatmap)
            overlay_bgr = self._create_heatmap_overlay(heatmap, image_path, meta)
            if overlay_bgr is None: continue
            
            # 2. Convert to PIL for Trajectory Drawing
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            overlay_pil = Image.fromarray(overlay_rgb)
            
            # 3. Draw Trajectory on top of Heatmap
            # Save as {stem}.png inside the method folder
            self._project_and_draw_points(
                pred_route,
                (meta["original_height"], meta["original_width"]),
                (0, 0, 255, 255), # Red
                3,
                method_dir / f"{image_path.stem}.png",
                background_image=overlay_pil
            )
        
        self._remove_hooks()
        self.attn_maps = {}
        self.grad_maps = {}

    def _get_generic_relevance(self, attn_maps, grad_maps):
        # Filter for Vision Layers only for the Image Heatmap
        # (Even if we hooked LLM, the image heatmap comes from ViT layers)
        vision_maps = {k: v for k, v in attn_maps.items() if "vision" in k}
        
        num_layers = len(vision_maps)
        if num_layers == 0: return torch.zeros(1)
        
        # Sort by layer index
        sorted_keys = sorted(vision_maps.keys(), key=lambda x: int(x.split("_")[-1]))
        
        first_map = vision_maps[sorted_keys[0]]
        B, H, S, S = first_map.shape
        R = torch.eye(S, device=self.device).unsqueeze(0).expand(B, S, S)
        
        for name in sorted_keys:
            attn = vision_maps[name]
            grad = grad_maps.get(name, torch.zeros_like(attn))
            
            cam = attn * grad
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
            
        # FIX: InternVL ignores CLS token. We must aggregate relevance w.r.t ALL patch tokens.
        # R[-1] is the last batch item (Global View).
        # R[-1, 1:, 1:] is the Patch-to-Patch relevance matrix.
        # Summing over dim 0 (rows) gives total relevance of each patch to the entire output embedding set.
        if S <= 1: return torch.zeros(1) # Ensure there are patches beyond CLS token
        return R[-1, 1:, 1:].sum(dim=0)

    def _get_ours_relevance(self, attn_maps, grad_maps, meta):
        # "Ours" = Generic Attention on Language Model Layers
        # Filter for Language Layers
        lm_maps = {k: v for k, v in attn_maps.items() if "language" in k}
        
        num_layers = len(lm_maps)
        if num_layers == 0: return torch.zeros(1)
        
        sorted_keys = sorted(lm_maps.keys(), key=lambda x: int(x.split("_")[-1]))
        
        first_map = lm_maps[sorted_keys[0]]
        B, H, S, S = first_map.shape
        
        # Initialize Relevance with Identity
        R = torch.eye(S, device=self.device).unsqueeze(0).expand(B, S, S)
        
        for name in sorted_keys:
            attn = lm_maps[name]
            grad = grad_maps.get(name, torch.zeros_like(attn))
            
            # Chefer Rule: E_h [ (A * G)^+ ]
            cam = attn * grad
            cam = cam.clamp(min=0).mean(dim=1) # [B, S, S]
            
            # R = R + CAM * R
            R = R + torch.bmm(cam, R)
            
        # Extract Relevance of Image Tokens w.r.t Target
        # Target is usually the last token (or the token where classification happens)
        # In SimLingo action mode, we pool or use the last token.
        # Let's assume the last token in the sequence is the "source" of relevance.
        # R[0, last_token_idx, :] -> Relevance of all tokens to the last token.
        
        # Note: R is [B, S, S]. R[b, i, j] is relevance of token j to token i.
        # We want relevance OF image tokens TO the target token.
        # So we look at row 'target_idx'.
        
        target_idx = S - 1 # Last token
        
        # Get image token indices
        image_indices = meta.get("image_token_indices", [])
        if not image_indices:
            print("[Ours] Warning: No image tokens found in prompt!")
            return torch.zeros(1)
            
        # FIX: InternVL concatenates tokens as [Tile1, Tile2, ..., GlobalView].
        # To visualize the whole image properly, we should focus on the Global View tokens.
        # These are the LAST 'num_patches' tokens in the image sequence.
        num_patches = meta.get("num_patches", 256) # Default to 256 if not found, but usually passed in meta
        
        # If we have more tokens than num_patches, it means we have tiles. Take the last ones.
        if len(image_indices) > num_patches:
            print(f"[Ours] Selecting last {num_patches} tokens (Global View) from {len(image_indices)} total image tokens.")
            image_indices = image_indices[-num_patches:]
        else:
            print(f"[Ours] Found {len(image_indices)} tokens (likely just Global View).")
            
        # Extract scores
        # For LLM, B is usually 1 (unless batching multiple prompts).
        # We take batch 0.
        scores = R[0, target_idx, image_indices] # [Num_Image_Tokens]
        
        return scores

    def _get_rollout_relevance(self, attn_maps):
        vision_maps = {k: v for k, v in attn_maps.items() if "vision" in k}
        sorted_keys = sorted(vision_maps.keys(), key=lambda x: int(x.split("_")[-1]))
        
        if not sorted_keys: return torch.zeros(1)

        first_map = vision_maps[sorted_keys[0]]
        B, H, S, S = first_map.shape
        R = torch.eye(S, device=self.device).unsqueeze(0).expand(B, S, S)
        
        for name in sorted_keys:
            attn = vision_maps[name]
            attn_mean = attn.mean(dim=1) # [B, S, S]
            attn_mean = attn_mean + torch.eye(S, device=self.device).unsqueeze(0)
            attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
            R = torch.bmm(attn_mean, R)
            
        # FIX: Aggregate all patches
        if S <= 1: return torch.zeros(1) # Ensure there are patches beyond CLS token
        return R[-1, 1:, 1:].sum(dim=0)

    def _get_flow_relevance(self, attn_maps):
        vision_maps = {k: v for k, v in attn_maps.items() if "vision" in k}
        sorted_keys = sorted(vision_maps.keys(), key=lambda x: int(x.split("_")[-1]))
        
        if not sorted_keys: return torch.zeros(1)

        first_map = vision_maps[sorted_keys[0]]
        B, H, S, S = first_map.shape
        R = torch.eye(S, device=self.device).unsqueeze(0).expand(B, S, S)
        
        for name in sorted_keys:
            attn = vision_maps[name]
            attn_mean = attn.mean(dim=1) # [B, S, S]
            # No residual connection for pure flow usually, but let's check standard impl.
            # Often it's just chain multiplication.
            R = torch.bmm(attn_mean, R)
            
        # FIX: Aggregate all patches
        if S <= 1: return torch.zeros(1) # Ensure there are patches beyond CLS token
        return R[-1, 1:, 1:].sum(dim=0)

    def _get_raw_attention(self, attn_maps):
        vision_maps = {k: v for k, v in attn_maps.items() if "vision" in k}
        sorted_keys = sorted(vision_maps.keys(), key=lambda x: int(x.split("_")[-1]))
        
        if not sorted_keys: return torch.zeros(1)

        last_name = sorted_keys[-1]
        attn = vision_maps[last_name]
        # FIX: Aggregate all patches
        if attn.shape[2] <= 1: return torch.zeros(1) # Ensure there are patches beyond CLS token
        return attn.mean(dim=1)[-1, 1:, 1:].sum(dim=0)

    def _create_heatmap_overlay(self, heatmap, image_path, meta):
        if heatmap.numel() == 0: return None
        
        n_patches = heatmap.shape[0]
        grid_size = int(np.sqrt(n_patches))
        
        # Ensure heatmap is detached and float before normalization
        heatmap = heatmap.float().detach()

        # Handle mismatch (e.g. if patches != square)
        if grid_size * grid_size != n_patches:
            # If not a perfect square, try to reshape to a 1D array for resizing
            # or handle cases where it's not easily reshaped to 2D.
            # For now, if it's not square, we'll just treat it as a 1D array
            # and let cv2.resize handle the interpolation.
            heatmap_2d = heatmap.view(1, n_patches).cpu().numpy()
        else:
            heatmap_2d = heatmap.view(grid_size, grid_size).cpu().numpy()
            
        orig_h, orig_w = meta["original_height"], meta["original_width"]
        
        # Robust Normalization
        # If max == min, normalize returns 0. Avoid this.
        if heatmap_2d.max() == heatmap_2d.min():
            heatmap_norm = np.zeros_like(heatmap_2d, dtype=np.uint8)
        else:
            heatmap_norm = cv2.normalize(heatmap_2d, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        heatmap_img = cv2.resize(heatmap_norm, (orig_w, orig_h))
        heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        
        orig_img = cv2.imread(str(image_path))
        if orig_img is None: return None
            
        overlay = cv2.addWeighted(orig_img, 0.5, heatmap_img, 0.5, 0)
        return overlay

    def _project_and_draw_points(
        self,
        points: Optional[torch.Tensor],
        image_hw: Tuple[int, int],
        color: Tuple[int, int, int, int],
        radius: int,
        output_path: Path,
        background_image_path: Optional[Path] = None,
        background_image: Optional[Image.Image] = None,
    ) -> None:
        if points is None: return
        points = points.detach().to("cpu")
        if points.dim() == 3: points = points.squeeze(0)
        if points.dim() != 2 or points.shape[1] != 2: return
        points = points.float()
        H, W = image_hw
        K = get_camera_intrinsics(W, H, 110)
        if torch.is_tensor(K): K_np = K.detach().cpu().numpy()
        else: K_np = np.asarray(K)
        try: projected = project_points(points.numpy(), K_np)
        except Exception: return
            
        # Load background
        if background_image is not None:
            base = background_image.convert("RGBA").resize((W, H))
        elif background_image_path and background_image_path.exists():
            base = Image.open(background_image_path).convert("RGBA")
            base = base.resize((W, H))
        else:
            base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            
        draw = ImageDraw.Draw(base)
        for coord in projected:
            x, y = float(coord[0]), float(coord[1])
            if not np.isfinite([x, y]).all(): continue
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                draw.ellipse((xi - radius, yi - radius, xi + radius, yi + radius), fill=color)
        base.save(output_path)

    def run_scene(self, scene_dir: Path, output_dir: Path):
        scene_dir = Path(scene_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = scene_dir / self.frames_subdir
        if not images_dir.exists(): images_dir = scene_dir / "images"
        if not images_dir.exists(): raise FileNotFoundError(f"No images found in {scene_dir}")
            
        speed_dir = scene_dir / self.speed_subdir
        speed_table = {}
        if speed_dir.exists(): speed_table = self._load_speed_table(speed_dir)
            
        image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg"}])
        
        for img_path in tqdm(image_paths, desc="Visualizing"):
            speed = speed_table.get(img_path.stem, 0.0)
            # Run ALL methods
            self.generate_visualization(
                img_path, 
                output_dir, 
                methods=["generic", "rollout", "raw", "flow", "ours"],
                metric="curv_energy",
                current_speed=speed
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    # Method argument removed as we run all
    args = parser.parse_args()
    
    vis = SimLingoVisualizer()
    vis.run_scene(args.scene_dir, args.output_dir)

if __name__ == "__main__":
    main()
