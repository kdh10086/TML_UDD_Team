# Sim-Lingo 시각화 실행 가이드 (원격 GPU) 

이 문서는 원격 GPU 서버에서 Sim-Lingo 모델의 시각화 코드를 실행하는 방법을 초보자도 쉽게 따라 할 수 있도록 설명합니다.

이 코드는 `.pt` 파일을 따로 저장하지 않고, 한 번의 실행으로 모델의 추론과 시각화(Heatmap)를 동시에 수행하여 결과를 이미지로 저장합니다.

## 1. 준비 사항

터미널을 열고 프로젝트 폴더(`TML_UDD_Team`)로 이동해 주세요. 필요한 파이썬 가상환경이 켜져 있어야 합니다.

```bash
cd TML_UDD_Team
# (필요하다면) conda activate [가상환경이름]
```

## 2. 실행 방법

아래 명령어를 그대로 복사해서 터미널에 붙여넣고 엔터(Enter)를 누르세요.

```bash
python3 experiment_alt/simlingo_visualizer.py \
    --scene_dir data/sample/01 \
    --output_dir experiment_outputs/simlingo_vis_alt
```

*   **`--scene_dir`**: 이미지가 들어있는 폴더 경로입니다. (예: `data/sample/01`)
*   **`--output_dir`**: 결과 이미지가 저장될 폴더 경로입니다.

> **참고**: 위 명령어를 실행하면 `generic`, `ours`, `rollout`, `flow`, `raw` 5가지 시각화 방식이 모두 한 번에 수행됩니다.

## 3. 결과 확인하기

실행이 완료되면 `experiment_outputs/simlingo_vis_alt` 폴더에 이미지 파일들이 생성됩니다. 각 파일의 의미는 다음과 같습니다.

| 파일명 예시 | 설명 | 비고 |
| :--- | :--- | :--- |
| `frame0001_generic.png` | **Vision 모델**이 중요하게 본 영역 | Chefer et al. 방식 |
| `frame0001_ours.png` | **LLM(언어 모델)**이 중요하게 본 영역 | Ours (LLM Generic) |
| `frame0001_rollout.png` | 정보가 어떻게 퍼지는지 보여줌 | Attention Rollout |
| `frame0001_flow.png` | 레이어 간의 연결 강도 흐름 | Attention Flow |
| `frame0001_raw.png` | Vision 모델 마지막 층의 단순 시선 | Raw Attention |
| `frame0001_traj.png` | 모델이 예측한 **주행 경로** (빨간 점) | Trajectory Overlay |

모든 이미지는 원본 사진 위에 붉은색/푸른색 히트맵(Heatmap)이나 점으로 겹쳐서(Overlay) 저장되므로, 직관적으로 확인할 수 있습니다.

## 4. 자주 발생하는 문제 (Troubleshooting)

*   **메모리 부족 오류 (CUDA Out of Memory)**
    *   GPU 메모리가 부족해서 실행이 멈춘다면, 코드 내의 `max_patches` 값을 줄여야 할 수 있습니다.
    *   (개발자에게 문의하거나 코드를 수정하여 `max_patches=1` 등으로 낮춰보세요.)

*   **이미지를 찾을 수 없음 (FileNotFoundError)**
    *   `--scene_dir`로 지정한 폴더 안에 `video_garmin` 또는 `images`라는 하위 폴더가 있는지, 그리고 그 안에 `.png`나 `.jpg` 파일이 있는지 확인해 주세요.
