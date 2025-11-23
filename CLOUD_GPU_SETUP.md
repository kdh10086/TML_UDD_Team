# Cloud GPU Setup Guide

이 레포를 윈도우 기반 우분투 VDI/클라우드 GPU 환경에서 바로 돌리기 위한 빠른 부트업 안내입니다.

## 1) 필수 전제
- NVIDIA 드라이버: CUDA 12.1 호환 드라이버(GPU 필수).
- 디스크 여유: 모델 체크포인트 및 HF 모델 다운로드 공간(수 GB 이상).
- 인터넷 접근: 처음 실행 시 HuggingFace 모델(InternVL2 등) 자동 다운로드 필요.

## 2) 리포지토리 클론 + 체크포인트
```bash
git clone --recursive https://github.com/kdh10086/TML_UDD_Team.git && cd TML_UDD_Team
git lfs install
git lfs pull  # checkpoints/simlingo/.../pytorch_model.pt 등 대형 파일 받기
# 또는 HuggingFace에서 모델 직접 클론 후 이 레포의 checkpoints/ 아래에 배치:
# git lfs clone https://huggingface.co/RenzKa/simlingo checkpoints/simlingo/simlingo/checkpoints/
```

## 3) 필수 시스템 패키지(우분투)
```bash
sudo apt-get update
sudo apt-get install -y libgl1 ffmpeg
```

## 3-1) Codex CLI 설치
```bash
# 1) 패키지 업데이트 + curl 설치
sudo apt update && sudo apt install -y curl

# 2) Node.js LTS 저장소 추가
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -

# 3) Node.js + npm 설치
sudo apt install -y nodejs

# 4) Codex CLI 전역 설치
sudo npm install -g @openai/codex

# 5) 버전 및 설치 확인
node -v
npm -v
codex --help
```

## 4) 파이썬 의존성
```bash
pip install -r requirements.txt  # CUDA 12.1 빌드(torch 2.3.1+cu121 등) 포함
```

## 5) 데이터 배치
- 전처리된 구조 예:
```
data/<dataset>/<scenario>/
  ├─ video_garmin/          # 4fps 프레임 PNG
  ├─ video_saliency/        # 4fps 히트맵 PNG
  └─ video_garmin_speed/    # m/s 속도 txt (프레임 스템 동일)
```
- 기본 scene_dir: `data/DREYEVE_DATA_preprocessed/01`
- 서브디렉토리명이 다르면 `--frames_subdir/--speed_subdir`로 지정.

## 6) Sim-Lingo 추론 실행
```bash
python experiment/simlingo_inference_baseline.py \
  --scene_dir data/DREYEVE_DATA_preprocessed/01 \
  --output_dir experiment_outputs/simlingo_inference
```
- tqdm로 시나리오 단위 진행률 표시.
- 입력 속도는 `video_garmin_speed`의 m/s를 자동 주입. 없으면 0 m/s 폴백.

## 7) 뷰어 (선택)
```bash
python tools/overlay_viewer.py
# data/<name> 또는 전체/시나리오 경로 입력 → a/d/좌/우로 탐색, q/Esc 종료
```

## 8) 기타
- FlashAttention2 미설치 시 경고만 출력, 동작에는 문제 없음.
- HF 모델 캐시 경로를 커스텀하려면 환경변수 `HF_HOME` 설정.
