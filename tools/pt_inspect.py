#!/usr/bin/env python3
"""
Quickly inspect the structure of a .pt file saved with torch.save.
Prints a shallow tree of keys, types, tensor shapes, and lengths.
Sim-Lingo 추론 결과(.pt)도 감지해 주요 필드를 한국어로 요약합니다.
"""
import argparse
import itertools
import os
import sys
from collections.abc import Mapping, Sequence

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a .pt file and print a compact summary."
    )
    parser.add_argument("path", help="Path to the .pt file")
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="How deep to recurse into nested structures (default: 8)",
    )
    parser.add_argument(
        "--items",
        type=int,
        default=200,
        help="Max mapping entries to show per level (default: 200)",
    )
    parser.add_argument(
        "--list-items",
        type=int,
        default=200,
        help="Max sequence items to show per level (default: 200)",
    )
    parser.add_argument(
        "--save-log",
        action="store_true",
        help="Save inspection output to tools/pt_inspect.log (appends)",
    )
    return parser.parse_args()


def format_tensor(tensor: torch.Tensor) -> str:
    shape = tuple(tensor.shape)
    return f"Tensor shape={shape} dtype={tensor.dtype}"


def _format_entry(obj) -> str:
    if isinstance(obj, torch.Tensor):
        return f"tensor shape={tuple(obj.shape)} dtype={obj.dtype}"
    return f"{type(obj).__name__}"


def _maybe_len(obj) -> str:
    try:
        return str(len(obj))
    except Exception:
        return "알 수 없음"


def _shape_or_len(obj) -> str:
    if isinstance(obj, torch.Tensor):
        return f"tensor {tuple(obj.shape)}"
    if isinstance(obj, Mapping):
        return f"dict len={len(obj)}"
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return f"list/seq len={len(obj)}"
    return type(obj).__name__


def _is_simlingo_payload(payload) -> bool:
    return (
        isinstance(payload, Mapping)
        and "attention" in payload
        and "outputs" in payload
        and "meta" in payload
    )


def describe_simlingo_payload(payload) -> None:
    """Sim-Lingo 추론 저장 포맷에 맞춘 요약을 한국어로 출력."""
    print("## Sim-Lingo 추론 결과 요약 (단일 프레임 기준)")
    tag = payload.get("tag")
    if tag:
        print(f"- 태그: {tag}")
    image_path = payload.get("image_path")
    if image_path:
        print(f"- 입력 이미지 경로: {image_path}")
    speed = payload.get("input_speed_mps")
    if speed is not None:
        print(f"- 입력 속도(m/s): {speed}")
    mode = payload.get("mode")
    if mode:
        print(f"- 설명 모드: {mode} (action/text)")

    meta = payload.get("meta") or {}
    if isinstance(meta, Mapping):
        oh = meta.get("original_height")
        ow = meta.get("original_width")
        if oh and ow:
            print(f"- 원본 해상도(HxW): {oh} x {ow}")
        num_patches = meta.get("num_patch_views")
        num_tokens = meta.get("num_total_image_tokens")
        if num_patches is not None or num_tokens is not None:
            print(f"- 패치/토큰: {num_patches}개 패치, 총 이미지 토큰 {num_tokens}")

    target_scalar = payload.get("target_scalar")
    target_info = payload.get("target_info") or {}
    if target_scalar is not None:
        if isinstance(target_scalar, torch.Tensor):
            print(f"- 타깃 스칼라: tensor shape={tuple(target_scalar.shape)}")
        else:
            print(f"- 타깃 스칼라: {type(target_scalar).__name__}")
    if target_info:
        if target_info.get("type") == "action":
            desc = target_info.get("description") or target_info.get("kinematic_metric")
            head = target_info.get("head")
            print(f"- 타깃 메타(액션): {desc} / head={head}")
        elif target_info.get("type") == "text":
            token_str = target_info.get("token_string", "")
            print(
                f"- 타깃 메타(텍스트): 전략={target_info.get('token_strategy')} "
                f"idx={target_info.get('token_index')} 토큰='{token_str}'"
            )

    outputs = payload.get("outputs") or {}
    if outputs:
        ps = outputs.get("pred_speed_wps")
        pr = outputs.get("pred_route")
        lang = outputs.get("language")
        if ps is not None:
            print(f"- 예측 속도(wps) 텐서: {tuple(ps.shape)}")
        if pr is not None:
            print(f"- 예측 경로 텐서: {tuple(pr.shape)}")
        if lang is not None:
            print(f"- language 출력 타입: {type(lang).__name__}")

    text_out = payload.get("text_outputs")
    if text_out:
        tokens = text_out.get("token_strings") or []
        print(f"- 텍스트 토큰: {len(tokens)}개, 예시 앞 5개: {tokens[:5]}")
        scores = text_out.get("token_scores")
        ids = text_out.get("token_ids")
        if scores is not None:
            print(f"  · token_scores 길이: {_maybe_len(scores)}")
        if ids is not None:
            print(f"  · token_ids 길이: {_maybe_len(ids)}")

    inter = payload.get("interleaver")
    if inter:
        print("- 인터리버 요약")
        tokens_per_patch = inter.get("tokens_per_patch")
        num_patches_list = inter.get("num_patches_list")
        selected_mask = inter.get("selected_mask")
        if tokens_per_patch is not None:
            print(f"  · tokens_per_patch: {tokens_per_patch}")
        if num_patches_list is not None:
            print(f"  · num_patches_list: {num_patches_list}")
        if isinstance(selected_mask, torch.Tensor):
            print(f"  · selected_mask: shape={tuple(selected_mask.shape)} sum={float(selected_mask.sum().item())}")

        def _shape_grad(entry, name: str):
            if entry is None:
                return f"{name}: 없음"
            tensor = entry.get("value") if isinstance(entry, dict) else entry
            grad = entry.get("grad") if isinstance(entry, dict) else None
            shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor).__name__
            grad_info = (
                f"grad shape={tuple(grad.shape)}" if isinstance(grad, torch.Tensor) else "grad 없음"
            )
            return f"{name}: {shape}, {grad_info}"

        for key in ["pixel_shuffle_out", "mlp1_input", "mlp1_output", "vit_embeds"]:
            print(f"  · {_shape_grad(inter.get(key), key)}")
        if inter.get("mlp1_weight") is not None:
            w = inter.get("mlp1_weight")
            print(f"  · mlp1_weight: {tuple(w.shape)}")
        if inter.get("mlp1_bias") is not None:
            b = inter.get("mlp1_bias")
            print(f"  · mlp1_bias: {tuple(b.shape)}")

    attention = payload.get("attention") or {}
    if attention:
        print("- 어텐션 맵 상세")
        for name in sorted(attention):
            stack = attention[name] or []
            print(f"  · {name}: {len(stack)}개")
            for idx, entry in enumerate(stack):
                attn_tensor = entry.get("attn")
                grad_tensor = entry.get("grad")
                shape = entry.get("shape")
                if shape is None and isinstance(attn_tensor, torch.Tensor):
                    shape = tuple(attn_tensor.shape)
                grad_info = "있음" if grad_tensor is not None else "없음"
                print(f"    [{idx}] attn_shape={shape} grad={grad_info}")

    print("")  # spacer


def describe(obj, name: str, indent: int, depth: int, args: argparse.Namespace) -> None:
    pad = "  " * indent
    if depth < 0:
        print(f"{pad}{name}: (depth limit reached)")
        return

    if isinstance(obj, torch.Tensor):
        print(f"{pad}{name}: {format_tensor(obj)}")
        return

    if isinstance(obj, Mapping):
        items = list(itertools.islice(obj.items(), args.items))
        print(f"{pad}{name}: {type(obj).__name__} len={len(obj)}")
        for key, value in items:
            describe(value, f"[{key!r}]", indent + 1, depth - 1, args)
        if len(obj) > len(items):
            print(f"{pad}  ... ({len(obj) - len(items)} more entries)")
        return

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        items = list(itertools.islice(obj, args.list_items))
        print(f"{pad}{name}: {type(obj).__name__} len={len(obj)}")
        for idx, value in enumerate(items):
            describe(value, f"[{idx}]", indent + 1, depth - 1, args)
        if len(obj) > len(items):
            print(f"{pad}  ... ({len(obj) - len(items)} more items)")
        return

    if isinstance(obj, (str, int, float, bool, type(None))):
        print(f"{pad}{name}: {obj!r}")
        return

    print(f"{pad}{name}: {type(obj).__name__}")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.path):
        print(f"File not found: {args.path}", file=sys.stderr)
        return 1

    log_fp = None
    if args.save_log:
        log_dir = Path(__file__).resolve().parent
        log_fp = open(log_dir / "pt_inspect.log", "a", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_fp)

    try:
        payload = torch.load(args.path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to load {args.path}: {exc}", file=sys.stderr)
        if log_fp:
            log_fp.close()
        return 1

    print(f"# {args.path}")
    print(f"type: {type(payload).__name__}")
    if _is_simlingo_payload(payload):
        describe_simlingo_payload(payload)
    describe(payload, "<root>", 0, args.depth, args)

    if log_fp:
        log_fp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
