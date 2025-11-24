# Cloud GPU Setup Guide

이 레포를 윈도우 기반 우분투 VDI/클라우드 GPU 환경에서 바로 돌리기 위한 빠른 부트업 안내입니다.

# 원격 GPU 노드 접속 매뉴얼
### 📢 [UDD 팀] GPU 서버 접속 가이드

우리 프로젝트 GPU 서버(Theta EdgeCloud) 접속 설정 방법입니다.
보안을 위해 각자의 로컬에서 키를 만들고 등록하는 방식으로 진행합니다.

#### 1단계: 본인의 SSH 키 생성
터미널(Mac/Ubuntu/WSL)을 열고 아래 명령어를 입력하세요.
**중요:** `-C` 뒤에 본인의 **영어 이니셜 혹은 닉네임(예: kdh, yjh)**을 적어주세요.

```bash
# 1. 키 생성
ssh-keygen -t ed25519 -C "본인이니셜 혹은 닉네임" -f ~/.ssh/theta_udd
```

> **⚠️ 필독 (비밀번호 설정 시 주의사항)**
> 키 생성 도중 `Enter passphrase`라고 물어볼 때:
> * **비밀번호 없이** 쓰려면: 그냥 엔터(Enter)를 두 번 치세요. (편리함/권장)
> * **비밀번호를 설정**하려면: 입력한 암호를 **반드시 메모장이나 텍스트 파일 등에 따로 적어두세요.**
> * (※ 접속할 때마다 이 암호를 묻습니다. 잊어버리면 **절대 복구 불가**하며 키를 다시 만들어야 합니다.)

#### 2단계: 접속 정보 미리 설정 (Config)
매번 IP와 포트를 입력하지 않도록 설정 파일을 미리 만들어둡니다.

1. 터미널에서 설정 파일 열기:
   `nano ~/.ssh/config`

2. 아래 내용을 맨 아래에 빈 줄을 만들고 그대로 붙여넣기:

```text
Host remote_gpu
    HostName 35.199.51.171
    User root
    Port 30096
    IdentityFile ~/.ssh/theta_udd
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```
Port 번호는 매번 달라질 수 있으므로, 웹페이지 접속해서 확인하고 수정해주세요.
3. 저장하고 나오기 (`Ctrl + O` 엔터 -> `Ctrl + X`)

#### 3단계: 공개키 전송 및 등록 대기
이제 만들어진 열쇠(공개키)를 김도형에게 보내주세요.

1. 아래 명령어로 공개키 내용 출력:
```bash
cat ~/.ssh/theta_udd.pub
```

2. **출력된 긴 문자열(`ssh-ed25519 ...`) 전체를 복사해서 김도형에게 보내주세요.**
3. **"서버 등록 완료"** 연락을 받을 때까지 잠시 대기합니다.

#### 4단계: 접속 테스트
등록이 완료되었다면, 터미널에서 아래 명령어만 치면 바로 접속됩니다.

```bash
ssh remote_gpu
```

※ 혹시 `WARNING: UNPROTECTED PRIVATE KEY FILE!` 에러가 뜨면 아래 명령어를 입력하고 다시 접속하세요.
`chmod 600 ~/.ssh/theta_udd`



# 원격 GPU 노드 접속 이후
## 1) 필수 전제
- NVIDIA 드라이버: CUDA 12.1 호환 드라이버(GPU 필수).
- 디스크 여유: 모델 체크포인트 및 HF 모델 다운로드 공간(수 GB 이상).
- 인터넷 접근: 처음 실행 시 HuggingFace 모델(InternVL2 등) 자동 다운로드 필요.

## 2) 원클릭 부트스트랩 (클론+의존성 설치 한 번에)
이미 SSH 접속한 상태에서 아래 한 줄로 클론 → LFS → 의존성 설치까지 실행:
```bash
git clone --recursive https://github.com/kdh10086/TML_UDD_Team.git && cd TML_UDD_Team && chmod +x tools/cloud_bootstrap.sh && bash tools/cloud_bootstrap.sh
```
스크립트가 libgl1/ffmpeg/git-lfs 설치, git lfs pull, requirements 설치, 지정된 공개키(ssh-ed25519 …hyun, ryu) 등록까지 수행하고,
`git config --global credential.helper store` 설정으로 토큰 캐싱까지 완료합니다. 남은 수동 작업(모델 HF clone, 데이터 배치 등)을 안내합니다.

## 3) 필수 시스템 패키지(우분투)
```bash
# sudo가 없으면 sudo를 빼고 실행
apt-get update
apt-get install -y libgl1 ffmpeg
```

## 3-1) Codex CLI 설치
# Codex CLI가 필요할 때 (sudo가 없으면 sudo 생략)
```bash
apt-get update && apt-get install -y curl
curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
apt-get install -y nodejs
npm install -g @openai/codex
# 확인: node -v && npm -v && codex --help
```

## Git 토큰 캐싱이 안 될 때 수동 설정
```bash
git config --global credential.helper store
git config --global user.name "<your name>"
git config --global user.email "<your email>"
# 이후 최초 git/pull/push 시 한 번 토큰/패스워드를 입력하면 ~/.git-credentials에 저장됩니다.
```

## 4) 체크포인트 및 샘플 데이터 다운로드
### SimLingo 체크포인트(HuggingFace)
```bash
cd checkpoints/ && git lfs clone https://huggingface.co/RenzKa/simlingo
```

### Persistent Storage에 저장된 데이터셋을 /UDD_TML_Team/data/로 복사, 압축해제
```bash
#샘플 데이터셋 복사
unzip /mnt/data1/new_sample_dataset.zip -d /root/TML_UDD_Team/data/
#원본 데이터셋 복사 (옵션)
unzip /mnt/data1/DREYEVE_DATA_filtered.zip -d /root/TML_UDD_Team/data/
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
- 단일 GPU 실행 예:
  ```bash
  PYTHONPATH=. python experiment/simlingo_inference_baseline.py \
    --scene_dir data/sample/01 \
    --output_dir experiment_outputs/simlingo_inference \
    --target_mode auto \
    --explain_mode action \
    --text_token_strategy max \
    --text_token_index -1 \
    --kinematic_metric curv_energy \
    --image_size 224 \
    --max_patches 2
  ```
### kinematic_metric 이름 -> 사용 토큰(source)/함수/설명 매핑
KINEMATIC_METRICS = {
    "curv_energy": {"source": "route", "fn": compute_curvature_energy, "description": "곡률 제곱합"},
    "curv_diff": {
        "source": "route",
        "fn": compute_curvature_diff,
        "description": "곡률 변화 제곱합",
    },
    "longitudinal_progress": {
        "source": "speed",
        "fn": compute_longitudinal_progress,
        "description": "종방향 전진 거리",
    },
    "forward_speed": {
        "source": "speed",
        "fn": compute_forward_speed,
        "description": "평균 전진 속도",
    },
    "acc_energy": {
        "source": "speed",
        "fn": compute_acceleration_energy,
        "description": "종방향 가속도 에너지",
    },
    "brake_energy": {
        "source": "speed",
        "fn": compute_brake_energy,
        "description": "제동(감속) 에너지",
    },
    "jerk_energy": {
        "source": "speed",
        "fn": compute_jerk_energy,
        "description": "종방향 jerk 에너지",
    },
    "none": {"source": None, "fn": None, "description": "절댓값 합을 사용하는 예비 설정"},
}
TEXT_TOKEN_STRATEGIES = ("max", "last", "index")

  - `--gpu_ids`를 주면 GPU별 별도 프로세스를 띄워 `--scene_dirs`를 균등 배분합니다.
- tqdm로 시나리오 단위 진행률 표시.
- 입력 속도는 `video_garmin_speed`의 m/s를 자동 주입. 없으면 0 m/s 폴백.

## 6-1) ViT 어텐션 시각화만 실행 (모델 재실행 없이 캐시 재사용)
- 전제: `simlingo_inference_baseline`로 생성된 `.pt`에 비전 어텐션이 포함돼 있어야 함(`experiment_outputs/simlingo_inference/.../pt/*.pt`).
- `--payload_root`는 해당 `.pt` 디렉터리(또는 상위). `.pt`에 이미지 경로가 없으면 `scene_dir`(선택)이나 `payload_root/input_images/<tag>.png`로 복구.
- 루트에서 실행 예시(샘플 데이터, 액션 모드 캐시):
```bash
# Raw attention
python -m experiment.vit_raw_attention \
  --output_dir experiment_outputs/vit_raw \
  --payload_root experiment_outputs/simlingo_inference/TML_UDD_Team_data_sample_scene_action_curv_energy_251123_2207 \
  --scene_dir data/sample_scene \
  --layer_index -1 --head_strategy mean --colormap JET --alpha 0.5

# Attention rollout
 python -m experiment.vit_attention_rollout \
  --output_dir experiment_outputs/vit_rollout \
  --payload_root experiment_outputs/simlingo_inference/TML_UDD_Team_data_sample_scene_action_curv_energy_251123_2207 \
  --scene_dir data/sample_scene \
  --residual_alpha 0.5 --start_layer 0 --colormap JET --alpha 0.5

# Attention flow
 python -m experiment.vit_attention_flow \
  --output_dir experiment_outputs/vit_flow \
  --payload_root experiment_outputs/simlingo_inference/TML_UDD_Team_data_sample_scene_action_curv_energy_251123_2207 \
  --scene_dir data/sample_scene \
  --residual_alpha 0.5 --discard_ratio 0.0 --colormap JET --alpha 0.5
```
- 캐시가 없으면 실행 불가(모델 재실행 없음). `.pt`에 비전 어텐션이 포함돼 있는지 먼저 확인하세요.

## 6-2) Generic Attention (텍스트/액션) — 캐시 전용 실행
- 전제: `simlingo_inference_baseline`로 생성된 `.pt`에 언어 블록 attn/grad가 포함돼 있어야 함(`text_outputs`/`attention` 존재). 이미지 경로는 `.pt`→`scene_dir`→`payload_root/input_images/<tag>.png` 순으로 복구.
- 루트에서 실행 예시:
```bash
# 텍스트 모드 Generic (캐시 전용)
 python -m experiment.generic_attention_baseline \
  --payload_root experiment_outputs/simlingo_inference/TML_UDD_Team_data_sample_scene_text_max_XXXX \
  --output_dir experiment_outputs/generic_text \
  --scene_dir data/sample_scene \
  --text_token_strategy max --text_token_index -1 \
  --colormap JET --alpha 0.5

# 액션 모드 Generic (캐시 전용, ours.py)
 python -m experiment.ours \
  --payload_root experiment_outputs/simlingo_inference/TML_UDD_Team_data_sample_scene_action_curv_energy_251123_2207 \
  --output_dir experiment_outputs/generic_action \
  --scene_dir data/sample_scene \
  --colormap JET --alpha 0.5
```
- `--payload_root`는 필수이며, 모델을 다시 돌리지 않습니다.

## 압축/전송(참고, zip 기준)
- 압축(현재 경로에 폴더가 있을 때): `zip -r <압축할파일이름>.zip <폴더이름>`  
  예: `zip -r experiment_outputs.zip experiment_outputs` → `./experiment_outputs.zip` 생성

- 압축 해제: `unzip <input.zip> -d <output_dir>`  
  예: `unzip sim_outputs.zip -d ./experiment_outputs/` → `./experiment_outputs/`에 압축해제
  
- scp 다운로드(로컬에서 실행): `scp -P <PORT> <user>@<host>:<remote_path.zip> <local_dest_dir>/`  
  예: `scp -P 30002 root@202.39.40.153:/root/TML_UDD_Team/experiment_outputs/sim_outputs.zip ~/home/컴퓨터이름/TML_UDD_Team/experiment_outputs/cloud_outputs/`

## 7) 기타
- FlashAttention2 미설치 시 경고만 출력, 동작에는 문제 없음.
- HF 모델 캐시 경로를 커스텀하려면 환경변수 `HF_HOME` 설정.

### Persistent Storage에 데이터셋 다운로드(Google Drive)
```bash
python3 -m pip install --upgrade gdown
cd /mnt/data1/
#샘플 데이터셋
gdown --fuzzy 'https://drive.google.com/file/d/1CfmRcnSZepCG0k9J4n5lkQZXd_tTxr7B/view?usp=drive_link' -O new_sample_dataset.zip
#DREYEVE_DATA_filtered 데이터셋
gdown --fuzzy 'https://drive.google.com/file/d/1-VgGkHAf5WNOCEISZXjNazaaEn3vE9r0/view?usp=sharing' -O DREYEVE_DATA_filtered.zip
```
