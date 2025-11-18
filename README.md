# Sim-Lingo Action-to-Vision Explainability Repo

이 레포는 pretrained Sim-Lingo InternVL2 모델로 장면 이미지를 추론하고, 이후 Vision-Language-Action 설명 기법(Transformer-MM, ViT attention 등)을 개발하기 위한 공통 데이터를 생성합니다. 핵심 목표는 **정책 텍스트가 아니라 실제 행동 궤적(trajectory)을 기반으로 한 action-to-vision 히트맵**을 얻는 것입니다.

## 추론 스크립트 개요

`experiment/simlingo_inference_baseline.py`는 입력 장면 디렉토리의 모든 이미지를 순차적으로 추론하고, 각 프레임에 대한 Attention/Gradient/운동학 스칼라/텍스트 정보를 `.pt` 파일로 저장합니다. 이 `.pt`가 이후 모든 설명 메소드의 입력이 됩니다.

### 입력과 디폴트 설정

```bash
python experiment/simlingo_inference_baseline.py \
  [--config checkpoints/simlingo/simlingo/.hydra/config.yaml] \
  [--checkpoint checkpoints/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt] \
  [--scene_dir data/scene01] \
  [--output_dir experiment_outputs/simlingo_inference] \
  [--explain_mode action|text] \
  [--kinematic_metric curv_energy|...] \
  [--text_token_strategy max|last|index] \
  [--text_token_index N]
```

- `--config`, `--checkpoint`는 기본값이 체크포인트에 포함된 `.hydra/config.yaml`과 `epoch=013.ckpt/pytorch_model.pt`로 세팅되어 있어 추가 인자 없이 실행할 수 있습니다.
- `--scene_dir`는 `scene01`처럼 이미지가 들어 있는 디렉토리를 가리킵니다.
- `--output_dir`는 결과 루트 디렉토리입니다. 실행 시점에 `scene명_모드_YYMMDD_HHMM` 형식의 하위 폴더가 자동 생성되어 그 안에 `.pt`가 저장됩니다.

### 런타임 동작 과정 (요약)

1. Hydra config/ckpt를 로드해 Sim-Lingo DrivingModel을 구성하고 모든 ViT/LLaMA 블록에 attention 훅을 건다.
2. `scene_dir` 내 모든 이미지를 정렬해 순회하며, Sim-Lingo의 `DrivingInput`을 생성 → 모델 forward → 텍스트/액션 출력을 얻는다.
3. `explain_mode`에 따라:
   - `action`: `pred_route` 또는 `pred_speed_wps`에서 운동학 함수(`curv_energy`, `longitudinal_progress`, `acc_energy`, `jerk_energy` 등)를 계산해 스칼라 \(y_t\)를 만든다.
   - `text`: 생성된 텍스트 토큰 로짓 중 전략(max/last/index)에 맞는 값을 \(y_t\)로 선택한다.
4. \(y_t\)에 대해 `backward()`를 호출해 모든 attention tensor에 gradient를 남긴 뒤, 이미지 1장당 하나의 `.pt` payload로 저장한다.

## 출력 규칙

- 저장 경로: `experiment_outputs/simlingo_inference/<scene>_<mode>_<YYMMDD_HHMM>_[suffix]/`.
  - 이미 동일한 폴더명이 존재하면 `_1`, `_2`, ...가 붙습니다.
- 파일명: `<이미지스탬>_<mode>.pt` (예: `scene01_0001_action.pt`).

## `.pt` 파일 구조

```python
payload = {
  "tag": "scene01_0001",
  "image_path": ".../scene01/scene01_0001.png",
  "mode": "action" | "text",
  "meta": {...},          # 원본 해상도, 이미지 토큰 수 등
  "target_scalar": Tensor,
  "target_info": {...},   # 사용한 kinematic metric 또는 텍스트 토큰 정보
  "outputs": {
      "pred_speed_wps": Tensor | None,
      "pred_route": Tensor | None,
      "language": List[str],
  },
  "text_outputs": {
      "token_ids": List[int],
      "token_scores": List[float],
      "token_strings": List[str],
      "decoded_text": str,
  } | None,
  "attention": {
      "vision_block_0": [{"attn": Tensor, "grad": Tensor|None, "shape": ...}, ...],
      "language_block_0": [...],
      ...
  },
}
```

- `meta`: `original_height/width`, `num_total_image_tokens`, `frames` 등 설명 기법이 필요로 하는 정보를 담고 있습니다.
- `target_info`: action 모드이면 `{"type": "action", "kinematic_metric": "curv_energy", "head": "route"}` 식으로 남으며, text 모드이면 특정 토큰 인덱스/문자열이 기록됩니다.
- `text_outputs`는 기본적으로 모든 모드에서 생성 토큰 정보를 저장합니다. (텍스트가 전혀 생성되지 않은 경우에만 `None`일 수 있습니다.)
- `attention` 딕셔너리는 각 블록마다 여러 헤드의 `[attn, grad]`를 저장한 리스트입니다.

> **주의:** `.pt` 파일에는 실행 당시 선택된 스칼라(`target_scalar`)에 대한 gradient만 포함됩니다. 예를 들어 텍스트 기반 실험을 새 토큰/전략으로 수행하려면, 해당 모드/옵션으로 `simlingo_inference_baseline.py`를 다시 실행해 새로운 `.pt`를 생성해야 합니다.

### 정보 접근 방법

1. Python에서 `payload = torch.load(".../scene01_0001_action.pt")`.
2. 예시:

```python
meta = payload["meta"]
attention = payload["attention"]["vision_block_0"][0]["attn"]     # 특정 블록의 어텐션
grad = payload["attention"]["vision_block_0"][0]["grad"]          # 대응 그래디언트
target_info = payload["target_info"]
metric_name = target_info.get("kinematic_metric", "unknown")
text_info = payload.get("text_outputs")
```

3. `meta["num_total_image_tokens"]`와 `meta["original_height"]`를 이용하면 토큰 relevance → 이미지 히트맵으로 투영할 수 있습니다.

## 팀원 참고

- `.pt` 파일은 추론 시점의 model state/attention/gradient를 그대로 담고 있으므로 **절대 편집할 필요가 없습니다**. 읽을 때는 `torch.load`로 메모리로만 가져오면 됩니다.
- 운동학 함수는 `--kinematic_metric` 인자로 손쉽게 바꿀 수 있으므로, 여러 action-to-vision 실험을 자동화할 때 CLI 옵션만 달리하면 됩니다.
