# Task Notes: Interleaver Relevance Logging

**Goal:** End-to-end relevance tracing from Qwen2 top to ViT patches by logging interleaver tensors (image-token mask, mlp1 outputs/grads, patch ordering) and enabling Transformer-MM-Explainability style propagation across the projection step.

- **Guideline:** Implement code changes in `experiment/simlingo_vit_qwen_attn_grad.py` (do not edit `external/`; copy to `experiment/` if needed and adjust imports).
- **Guideline:** Never modify files under `external/`. If changes are needed, copy to `experiment/`, modify there, and update import paths accordingly.

- [x] Identify exact tensors to log at interleaver:
  - [x] `selected` mask for image-context token positions (B×N boolean).
  - [x] `vit_embeds` right before insertion (output of pixel_shuffle + mlp1) and its grad.
  - [x] Optional: pixel_shuffle output (pre-mlp1) and `num_patches_list` to map patch order.
- [x] Decide logging format and payload keys for new tensors in PT saves (reuse `attention` dict or add `interleaver` section).
- [x] Implement hooks/logging in `InternVLChatModel.forward` (or wrapper) to store tensors and masks per frame/batch.
- [ ] Verify saved PT payload contains new interleaver metadata for a test batch.
- [ ] Define relevance backprop step for interleaver (linear layer) consistent with Transformer-MM-Explainability: use `mlp1` weights + grad to map Qwen image-token relevance back to ViT patches.
- [ ] Write a small post-processing script/notebook to reconstruct mapping (token index ↔ patch) and run end-to-end relevance propagation on one sample.
- [ ] Document usage steps for future runs (how to enable logging, where outputs are saved, how to run post-processing).

## Next: integrate interleaver relevance in ours.py
- Plan:
  - Parse `interleaver` section from PT payload (selected_mask, tokens_per_patch, num_patches_list, pixel_shuffle_out/grad, mlp1_input/grad, mlp1_output/grad, vit_embeds/grad).
  - Add LLM→interleaver backprop: take image-token relevance, map through mlp1 (weights/grad) back to pixel_shuffle grid, using selected_mask/tokens_per_patch for alignment.
  - Run ViT relevance rollout (A ⊙ grad) from patched grid down to image patches; skip CLS.
  - Keep changes in `experiment/ours.py` only; add safety guards/logs for shape mismatch.
  - Validate on a small batch: check interleaver presence, token-to-patch counts, final patch relevance sum/heatmap.

## To-do: precise heatmap alignment (ViT–Qwen interleaver) in ours.py
- [x] Compute per-image token offsets using `selected_mask`, `tokens_per_patch`, `num_patches_list`; handle multi-image concat.
- [x] Split image-token relevance by image, average tokens_per_patch → reshape to (H×W) using `pixel_shuffle_out`; log/pad/trim on mismatch.
- [x] Support multiple images: produce per-image heatmaps (suffix `_img{idx}`) for all slices; document behavior.
- [x] After vision relevance, upsample patch grid to original image size (meta) and verify size matches loaded image; warn if not.
- [x] Add robust error handling/logs for shape/count mismatches; avoid hard crashes where possible.

## Gaps to address for end-to-end relevance (Transformer-MM style)
- [x] Interleaver step: incorporate mlp1/pixel_shuffle grad weighting (using recorded grads; still heuristic, not full Chefer).
- [x] Add cross-step relevance direction clarity: only `llm_to_vision` supported and validated.
- [x] Handle multi-image cases explicitly (all slices).
- [x] Vision relevance: use A⊙grad with residual_alpha blending (cam@R + residual_alpha * R).
- [x] Align final heatmap with exact image coords using stored H×W and any crop/resize applied during preprocessing; add verification step.

## Remaining validation/documentation tasks
- [ ] Validate PT payloads after a test run: confirm `interleaver` fields (mlp1 weight/bias/grad, pixel_shuffle grad) and multi-image outputs are present.
- [ ] Add post-processing sample (script/notebook) to reload one PT and run end-to-end relevance (LLM→Interleaver→ViT) with intermediate logs (token/patch counts, grid sizes).
- [ ] Document usage: how to enable logging, `propagation_mode` limitation (`llm_to_vision`), multi-image output naming (`_img{idx}`), heuristic nature of grad weighting.
- [ ] Run smoke tests (single/multi-image batches) and review `pipeline.log` to ensure no crashes and inspect warnings.
- [x] Add logging hooks so that, after integrated pipeline runs, a summarized verification log (checks on interleaver fields, attention presence) is saved alongside `pipeline.log` for easier remote review.

## Heatmap quality fixes (new)
- [x] Normalize/clamp relevance before heatmap (e.g., min-max or percentile clip) to avoid washed-out overlays; add toggle in visualizers.
- [x] Validate tokens_per_patch/grid mapping in interleaver; on mismatch, skip/adjust with clear warnings.
- [x] Add rollout tuning options: residual_alpha, optional softmax/temperature on cams, and layer depth selection.
- [ ] Log target scalar/token choice per frame; warn if target selection is empty or unexpected.
- [ ] Add a simple PT QA script: load one payload, print relevance stats (max/mean/var), and render a quick heatmap for manual QA.

