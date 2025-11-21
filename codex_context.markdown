# codex_context.markdown

이 파일은 Git 레포지토리 내에서 코드 어시스턴트(Codex 등)에게 제공할 **프로젝트 맥락 / 구현 규칙 / 작업 계획**을 정리한 문서입니다.  
코드를 작성하거나 리팩터링할 때, 이 문서의 내용을 항상 전제로 삼아야 합니다.

---

## 0. 프로젝트 개요 (요약)

- 우리는 **Sim-Lingo InternVL2 기반 VLA 모델**을 **추가 파인튜닝 없이(pretrained 그대로)** 사용합니다.
- 목표는 **정책 텍스트가 아니라 실제 액션/경로(trajectory)를 기준으로** 한 **action-to-vision 설명**을 만드는 것입니다.
- 파이프라인 핵심 아이디어:
  1. Sim-Lingo의 **액션 헤드 출력 → 20×2(or 10×2) 경로 좌표**로 디코딩
  2. 경로 좌표에서 **운동학적 함수(곡률, 가속도, jerk, 위험도 등)** 를 정의하여 **스칼라 \(y_t\)** 로 요약
  3. 이 \(y_t\)를 Chefer의 **Transformer-MM-Explainablity 메소드** 방식에서의 target output으로 사용
  4. Transformer 내부 attention 구조를 따라 relevance를 역전파하여
  5. 최종적으로 **“이번에 생성된 실제 경로(행동)에 인과적으로 기여한 시각 패치/영역”** 를 나타내는 히트맵을 얻음
  6. 위 과정을 횡방향 제어 행렬 p,  종방향 제어 행렬 w에 각각 적용하여 각각의 히트맵을 얻음.
- 비교군으로는
  - **텍스트 토큰 기반 Transformer-MM-Explainablity 메소드**
  - **Chefer의 Transformer Attribution (텍스트 토큰 기반)**
  - **ViT Attention / Attention Rollout (ViT 내 시각화)**
  를 구현합니다.
  이때, 텍스트 토큰 기반 메소드들은 output 텍스트 중 동사 단어의 토큰의 가장 값이 큰 logit을 y_t로 사용한다.
  그리고 ViT 내 시각화 메소드들은, ViT 시각 인코더 내에서만 훅을 걸어서 동작하도록 한다. (시각->시각)
- 최종 목표:
  - **원터치(one-touch)** 실행으로 모든 장면·모든 메소드의 히트맵을 자동 생성
  - 이후 기말고사 이후에 정량 평가에 사용할 수 있는 히트맵 결과를 확보

---

## 0.1 현재 구현 상태 / 부트업 체크리스트

- 데이터: `data/DADA-2000-Core/sorted_index/` 아래에 DADA-2000-Core 시나리오들이 준비됨(테스트는 여기 시나리오 사용). Scene 5개 선정·크롭 규약 확정은 여전히 필요.
- 추론: `experiment/simlingo_inference_baseline.py` 완성. action/text 모드에서 **추론 시점에 kinematic/text 스칼라를 만들고 `backward()`까지 수행**하여 각 블록 어텐션+gradient를 `.pt` payload로 저장. 기본 경로는 `checkpoints/simlingo/simlingo/.hydra/config.yaml` / `checkpoints/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`(약 9GB, LFS).
- action 타깃: 액션 토큰이 아니라 **액션 MLP 출력( pred_route 20×2, pred_speed_wps 10×2 )을 곡선으로 근사 → 운동학 함수 → 스칼라 \(y_t\)**. kinematic metric은 플러그인(curv_energy/acc_energy/progress/brake/jerk 등).
- text 타깃: 생성 텍스트 로짓에서 전략(max/last/index)으로 스칼라를 선택해 `backward` 수행.
- payload 구조: `target_scalar`, `target_info`, `attention`(attn+grad per block), `meta`(원본 H/W, 이미지 토큰 수), `text_outputs`(token ids/scores/strings/decoded) 등이 포함됨. Generic/다른 메소드에서 재추론 없이 활용 가능해야 함.
- 입력 규약: `scene_dir`는 시나리오 루트(`.../sorted_index/<city>/<scenario>`)를 가리키며, 스크립트가 하위 `images/`에서 프레임을 읽는다. 프롬프트는 commentary 모드 고정(`Current speed: {speed} m/s. What should the ego vehicle do next? Provide a short commentary.`). 기본은 모든 프레임에 0 m/s를 입력·표기하며, `--use_prev_speed`를 켜면 이전 프레임 `pred_speed_wps`에서 속도를 근사(0~30 m/s 클램프, 3구간 평균)해 다음 프롬프트/`vehicle_speed`에 주입한다.
- 출력 포맷: scene 단위로 `sorted_index(or original_index)_<city>_<scenario>_{mode}_{detail}_{YYMMDD_HHMM}/` 생성되고, 중복 시 `_1` 등 suffix로 덮어쓰기 방지. 여기서 `{detail}`은 action 모드면 kinematic metric 이름, text 모드면 text_token_strategy 이름. 그 안에 파일 유형별 서브디렉토리(`pt/`, `route_overlay/`, `speed_overlay/`, `text_output/`, `pred_route/`, `pred_speed_wps/`, `input_images/`)가 있으며, 각 서브디렉토리에는 입력 이미지 스템 이름을 딴 파일이 저장됨(예: `pt/frame001.pt`, `route_overlay/frame001.png`, `speed_overlay/frame001.png`, `text_output/frame001.txt`, `pred_route/frame001.txt`, `pred_speed_wps/frame001.txt`, `input_images/frame001.png`). PNG는 투명 배경 위에 투영점만 표시.
- 전처리: `dynamic_preprocess` 호출 시 `use_global_img`가 config에 없을 경우 기본값으로 `True`를 사용하도록 방어 로직 추가.
- 메모리 튜닝: CLI에서 `--image_size`(기본 448), `--max_patches`(기본 2)를 내려서 GPU 메모리를 줄일 수 있음. 필요 시 `--image_size 336 --max_patches 1` 등으로 실행.
- 기본 해상도: 8GB VRAM 노트북 환경을 고려해 기본 `--image_size`는 224로 낮춘 상태이며, 필요 시 인자로 올려서 사용.
- 의존성: 추론 실행에 필요한 패키지는 루트 `requirements.txt`에 정리되어 있음. 의존성 오류가 나면 `pip install --user -r requirements.txt`로 설치.
- 오버레이/텍스트: pred_route/pred_speed_wps는 float32로 변환 후 투영해 오버레이 PNG를 생성. text_output은 모델 생성 텍스트(`text_outputs`가 있으면 이를, 없으면 `language` 필드 fallback)만 기록하며, 추가 LLM 후처리나 수치 템플릿 요약은 하지 않는다.
- **속도 입력 문제(중대)**: SimLingo는 입력으로 이미지 + 현재 속도를 요구하나 DADA-2000에는 속도 데이터가 없음. 현재는 기본 0 m/s(또는 `--use_prev_speed`로 이전 예측 속도 주입)로 추론하지만, 이는 실제 상황과 불일치해 심각한 오판을 유발할 수 있음. 팀 차원 결정 필요. 후보:
  1) 시나리오 영상을 Gemini 등 LLM에 보내 초반 속도만 추정, 이후는 pred_speed_wps 기반 재귀 주입(초기값 검토 필요, 누적 오차 위험).
  2) 시나리오 영상을 LLM에 보내 프레임별 속도 추정 후 각 프레임에 주입(실험 변수로 외부 모델 예측이 들어감, 비용/지연·정확도 리스크).
  3) 속도 포함된 다른 데이터셋으로 전환(사고/GT 히트맵 여부는 재고, 속도+이미지 우선).
  4) CARLA에서 사고 시나리오 자체 생성 후 추론(속도 정확, SimLingo 학습 분포와 일치, 변인 통제 우수하지만 구축 비용/VRAM·환경 설정 부담).
- Generic Attention: 현재 텍스트 모드 구현본이 `experiment/generic_attention_baseline.py`에 있으며, **앞으로 action/text 공용으로 `.pt`를 입력 받아 Chefer rule 5/6로 relevance만 누적→히트맵 저장**하도록 리팩터링 필요.
- ViT 시각화: `experiment/vit_raw_attention.py`, `experiment/vit_attention_rollout.py`, `experiment/vit_attention_flow.py`가 구현 완료(현재는 직접 추론 실행 방식).
- 통합 실행/데이터 루프: `run_all_methods.py` 등 통합 스크립트와 scene 데이터 준비는 미완.

---

## 1. 실험 계획 (Experiment Plan)

1.  기말 전까지 우선 목표:
   - 각 메소드별 히트맵 결과 이미지들을 **자동으로 생성/모으는 코드**를 완성
   - 단일 실행으로,
     - Sim-Lingo 추론 수행
     - 5개 설명 메소드 모두 실행
     - 모든 Scene 구간에 대해 히트맵 결과를 디렉토리에 정리해 저장
   - 즉 **“실행 버튼 한 번 → 전 메소드 히트맵 결과 생성”** 구조를 만든다.

---

## 2. 실험 구현 단계 (a–f)

### a. Scene 구간 5개 찾기 (Phase 1, 최우선)

- 시나리오를 여러 번 돌려보면서 **실험에 사용할 “Scene 구간” 5개**를 선정한다.
- 현재 테스트 데이터 루트: `data/DADA-2000-Core/sorted_index/` (DADA-2000-Core 정렬 시나리오). 여기에서 Scene 5개를 선정하고 크롭/리사이즈 규약을 확정한다.
- 각 Scene 구간은:
  - **히트맵이 시각적으로 의미 있고 아름답게** 나와야 한다.
  - **사고 직전 일반 상태**가 잠깐 포함되어야 한다.
  - **사고 과정 전체**가 포함되어야 한다.
  - **사고 이후 차량 정지(혹은 이에 준하는 상태 종료)** 가 포함되어야 한다.
  - 길이는 가변적이어도 된다. (이미지 개수 N은 Scene마다 달라도 괜찮음)
- Scene 1~5 각각에 대해:
  - 어떤 기준으로 프레임을 자를지(초 단위 / 이벤트 단위 등)를 정리
  - 이미지 파일명 규칙 예:
    - `data/scene01/scene01_0001.png`
    - …
  - 디렉토리 구조 예:
    - `data/scene01/*.png`
    - `data/scene02/*.png`
  - 각 Scene의 전개를 **간단한 텍스트 설명**으로 기록
- 이 구조가 이후 b, c, d, e에서 **그대로 재사용되는 데이터 I/F 규약**이므로 중요하다.

---

### b. Sim-Lingo 추론 베이스라인 코드 (Phase 1, 최우선)

- **Sim-Lingo 추론 전용 베이스라인 코드**를 만든다.
- 특징:
  - “진짜 추론만 하는 코드”
  - 모든 설명 메소드는 **반드시 이 코드만 사용**하여 추론 결과(토큰, trajectory, attention 등)를 가져온다.
- 입력:
  - `scene_dir: str` — Scene 이미지들이 들어있는 디렉토리 경로 (N개)
- 출력(예시 딕셔너리 구조, 필요에 따라 확장 가능):

  - `"images"`: 원본 이미지를 텐서/리스트 형태로
  - `"vision_tokens"`: ViT patch 토큰
  - `"text_tokens"`: 텍스트/정책 토큰 (필요 시)
  - `"action_tokens"`: 액션 헤드 토큰
  - `"trajectory"`: 디코딩된 경로, 예: `(N, T, 2)` (T=20 or 10)
  - `"attention_maps"`: 각 레이어/헤드의 self-attention (필요 시)

- 이 모듈은 **전 메소드의 공통 기반**이므로, 모델 로딩/전처리/후처리를 중복 구현하지 않도록 한다.
- Sim-Lingo에서 요구하는 입력 해상도에 맞게 **이미지 크롭/리사이즈**도 이 단계에서 처리한다. (Scene 선정 후 적용)

---

### c. Generic Attention 논문 방식 조상(ancestor) 베이스라인 (2개 메소드)

- Chefer의 **Generic Attention Explainability** 방식을 기반으로 한 조상 메소드를 구현한다.
- 이 조상 코드를 토대로:

  1. **Ours (action-based \(y_t\))**
  2. **텍스트 토큰 기반 \(y_t\)**

  두 메소드를 파생시킨다.

#### c-1. Generic Attention 조상 코드에 필요한 변경

Generic Attention 논문 코드(공개 구현 등)를 가져와 분석 후 다음을 수행한다.

1. **입력은 Sim-Lingo 추론 출력 `.pt`**  
   - `simlingo_inference_baseline.py`가 저장한 payload(`attention`의 attn+grad, `target_info`, `meta`, `text_outputs`)를 로딩한다.  
   - 이미 추론 시 `y_t`를 만들고 `backward()`를 수행했으므로, 모델 재실행이나 스칼라 재계산을 하지 않는다.
2. **Chefer rule 5/6로 relevance 누적 후 히트맵 저장**  
   - 언어/비전 블록별 attn·grad를 활용해 relevance matrix를 만들고, 이미지 토큰 위치로 투영 후 PNG 저장.  
   - action/text를 동일 코드 경로로 처리하되, target/토큰 위치 정보는 payload에 기록된 `mode`/`target_info`/`text_outputs`/`meta`를 사용한다.
3. **입력 세트 반복 처리**  
   - `.pt` 폴더 내 모든 파일에 대해 relevance→히트맵을 생성한다.

#### c-2. 고정된 입출력 규칙

모든 메소드에서 통일해야 하는 규칙:

- **입력(Input)**  
  - `pt_dir: str` — `simlingo_inference_baseline.py`가 생성한 `.pt` 파일들이 들어있는 디렉토리 (한 scene에 대한 추론 결과)
- **출력(Output)**  
  - 히트맵 이미지들이 들어있는 **결과 디렉토리 생성 (N개)**  
  - `output_dir: str` — 필요한 경우 자동 생성

메소드 시그니처 예시:

- `generate_heatmaps_generic_ours(pt_dir: str, output_dir: str, ...) -> None`
- `generate_heatmaps_generic_text(pt_dir: str, output_dir: str, ...) -> None`

#### c-3. 두 메소드

1. **Ours (action-based \(y_t\))**
   - Sim-Lingo 액션 헤드 MLP 출력(pred_route \(p\), pred_speed_wps \(w\)) → 경로 곡선 근사 → 운동학 스칼라 함수 → \(y_t\)
   - 운동학 스칼라 함수는 **플러그인 방식**으로 붙였다 뗐다 가능해야 한다.
     - 여러 후보 함수(아래 “운동함수 후보” 참조)를 빠르게 교체하며 실험 가능해야 함.

2. **텍스트 토큰 기반 Generic Attention**
   - 기존 연구들과 동일하게 텍스트/정책 토큰 기반의 \(y_t\) 정의
   - 예: 특정 텍스트 토큰의 로짓이나 score 등을 스칼라 target으로 사용
   - 나머지 파이프라인은 Ours와 동일 (동일 입출력 규약)

---

### d. Transformer Attribution 논문 방식 베이스라인 (1개 메소드)

- **Transformer Attribution** (예: “Transformer Interpretability Beyond Attention Visualization”) 방식을 사용한 텍스트 토큰 기반 메소드 1개를 구현한다.

- 입출력 규칙은 동일:

  - 입력: `scene_dir` (N개 이미지)
  - 출력: 히트맵 이미지들이 들어 있는 결과 디렉토리 생성 (N개)

- 메소드 예시 시그니처:
  - `generate_heatmaps_transformer_attr(scene_dir: str, output_dir: str, ...) -> None`

---

### e. ViT Attention / Attention Rollout (2개 메소드)

- ViT 내부에서만 동작하는 시각화 메소드 2개를 구현한다.

1. **Raw Attention 방식**
   - ViT의 self-attention map을 직접 이용한 시각화
   - 마지막 레이어(혹은 여러 레이어 평균)의 CLS→patch attention을 사용해 이미지 히트맵 생성

2. **Attention Rollout 방식**
   - 여러 레이어의 attention을 roll-out (연속 곱)하여 CLS→patch 글로벌 relevance를 구한 후 히트맵 생성

- 두 메소드 모두에서 동일 입출력 규칙 사용:

  - 입력: `scene_dir` (N개 이미지)
  - 출력: 결과 디렉토리(각 N개 히트맵) 생성

- 예시 시그니처:
  - `generate_heatmaps_vit_attention(scene_dir: str, output_dir: str, ...) -> None`
  - `generate_heatmaps_vit_rollout(scene_dir: str, output_dir: str, ...) -> None`

---

### f. 통합 실행 코드 (Phase 3, 우선순위 가장 뒤)

- 최종적으로 **5개의 출력 메소드를 한 번에 호출**하여,
  - 모든 Scene 디렉토리에 대해 반복 실행하는 통합 스크립트를 작성한다.

- 예시 스크립트:
  - `run_all_methods.py`

- 기능:
  - 인자:
    - `--scenes_root`: `data/` 처럼 scene01~scene05가 들어있는 루트 디렉토리
    - `--output_root`: `results/` 루트
  - 내부 동작:
    - `data/scene01`, `data/scene02`, … 순회
    - 각 Scene에 대해:
      - 5개 메소드의 `generate_heatmaps_*` 호출
      - 결과는 예를 들어 다음과 같이 저장:
        - `results/generic_ours/scene01/`
        - `results/generic_text/scene01/`
        - `results/transformer_attr/scene01/`
        - `results/vit_attention/scene01/`
        - `results/vit_rollout/scene01/`

---

## 3. 운동학 스칼라 함수 후보 (Longitudinal / Lateral)

### 3.1 종방향(롱기튜드) 제어 \(w \in \mathbb{R}^{10 \times 2}\) — 2.5초, Δt = 0.25s

- 설정:
  - \( w_t \in \mathbb{R}^2, \ t = 0, \dots, 9 \): 시간 t에서의 위치
  - 시간 간격: \( \Delta t = 0.25 \,s \)
  - 위치 변화:
    - \( \Delta w_t = w_t - w_{t-1}, \ t = 1, \dots, 9 \)

#### (1) 종방향 방향축 정의

- 첫 step의 진행 방향을 종방향 축으로 정의:

  - \( e_\text{long} = \frac{\Delta w_1}{\|\Delta w_1\|_2 + \varepsilon} \)

#### (2) 속도, 가속도, jerk

- 종방향 속도:
  - \( v_t = \frac{\Delta w_t \cdot e_\text{long}}{\Delta t}, \ t = 1, \dots, 9 \)
- 종방향 가속도:
  - \( a_t = \frac{v_t - v_{t-1}}{\Delta t}, \ t = 2, \dots, 9 \)
- 종방향 jerk:
  - \( j_t = \frac{a_t - a_{t-1}}{\Delta t}, \ t = 3, \dots, 9 \)

#### 종방향 스칼라 후보들

1. **평균 전진 속도**

   - \( g_\text{speed}(w) = \frac{1}{9} \sum_{t=1}^{9} \text{ReLU}(v_t) \)
   - 전진 정도(후진은 무시)에 대한 지표

2. **누적 전진 거리**

   - \( g_\text{progress}(w) = \sum_{t=1}^{9} \text{ReLU}(\Delta w_t \cdot e_\text{long}) \)
   - 2.5초 동안의 전진량

3. **가속도 에너지 (동적 강도 / comfort 저하)**

   - \( g_\text{acc-energy}(w) = \sum_{t=2}^{9} a_t^2 \)
   - 가속/감속의 크기를 제곱합한 값

4. **제동(감속) 에너지 (braking risk)**

   - \( g_\text{brake}(w) = \sum_{t=2}^{9} \text{ReLU}(-a_t)^2 \)
   - 감속(음의 가속도) 구간만 보는 제동 강도 지표

5. **jerk 기반 승차감/부드러움**

   - \( g_\text{jerk}(w) = \sum_{t=3}^{9} j_t^2 \)
   - acceleration의 변화량 제곱합 → 승차감/부드러움과 관련

> 종방향에서:
> - 전진/정지/저속에 관심 → \( g_\text{speed} \) 또는 \( g_\text{progress} \)
> - 제동/급제동 위험 → \( g_\text{brake} \)
> - 승차감/부드러운 주행 → \( g_\text{jerk}, g_\text{acc-energy} \)
> 를 각각 별도의 \(y_t\) 로 정의해 Generic Attention을 실행할 수 있다.

---

### 3.2 횡방향 제어 \(p \in \mathbb{R}^{20 \times 2}\) — 거리 기반 path

- 설정:
  - \( p_i \in \mathbb{R}^2, \ i=0,\dots,19 \) : path 상의 점
  - 샘플 간 거리 간격: \( \Delta s \approx 1m \) 정도로 가정

#### (0) 국소 방향 / 곡률 정의

1. 국소 tangent 벡터:
   - \( \Delta p_i = p_i - p_{i-1}, \ i=1,\dots,19 \)
   - \( t_i = \frac{\Delta p_i}{\|\Delta p_i\|_2 + \varepsilon} \)
2. heading 각도:
   - \( \theta_i = \text{atan2}(t_i^y, t_i^x) \)
3. 이산 곡률 근사:
   - \( \kappa_i = \frac{\theta_i - \theta_{i-1}}{\Delta s}, \ i=2,\dots,19 \)

#### 횡방향 스칼라 후보들

1. **곡률 에너지 — 조향 강도(steering magnitude)**

   - \( g_\text{curv}(p) = \sum_{i=2}^{19} \kappa_i^2 \)
   - 크게/자주 꺾이는 궤적일수록 값이 커짐

2. **곡률 변화량 — 조향 부드러움 / 횡방향 jerk**

   - \( g_\text{curv-diff}(p) = \sum_{i=3}^{19} (\kappa_i - \kappa_{i-1})^2 \)
   - 조향을 급하게 바꾸는 정도

3. **차선 유지 / lateral deviation (reference 경로가 있을 때)**

   - 기준 경로 \( r_i \), 기준 heading \( \theta_i^\text{ref} \) 가 있을 때,
   - 횡방향 단위벡터:
     - \( e_{\text{lat},i} = (-\sin \theta_i^\text{ref}, \cos \theta_i^\text{ref}) \)
   - 횡방향 오프셋:
     - \( d_i = (p_i - r_i) \cdot e_{\text{lat},i} \)
   - 차선 유지 지표:
     - \( g_\text{lat-dev}(p) = \sum_{i=0}^{19} d_i^2 \)
   - lane-change에 특화시키려면:
     - \( g_\text{lane-change}(p) = \sum_{i=15}^{19} d_i^2 \) 등

4. **횡방향 안전 여유 (장애물이 있을 때)**

   - 각 점에서 장애물/경계까지의 횡방향 거리 \( b_{i,k} \) 가 주어졌다고 가정
   - soft-min lateral margin:
     - \( m_i = -\frac{1}{\alpha} \log \sum_k \exp(-\alpha b_{i,k}) \)
   - 전체 lateral risk:
     - \( g_\text{lat-risk}(p) = \sum_i \text{ReLU}(m_\text{th} - m_i)^2 \)
       - \( m_\text{th} \): 안전 여유 임계값 (예: 0.5m)

> 횡방향에서:
> - 조향 강도 → \( g_\text{curv} \)
> - 조향 부드러움/jerk → \( g_\text{curv-diff} \)
> - 차선 유지/이탈 → \( g_\text{lat-dev}, g_\text{lane-change} \)
> - 횡방향 안전 여유 → \( g_\text{lat-risk} \)
> 등을 각각 별도의 \(y_t\) 로 두고 Generic Attention을 수행할 수 있다.

### 3.3 운동학 함수 그래디언트 소멸 방지 팁

- 문제: 곡률/가속도 에너지처럼 입력이 0이면 \( \partial g / \partial (\text{입력}) = 0 \) 이 되어 relevance가 죽는다. 상수 바이어스를 더해도 기울기는 동일하게 0이다.
- ReLU 완화: \(\text{ReLU}(x) \rightarrow \tau \log(1+\exp(x/\tau))\) (softplus)로 교체하면 0 근처에도 기울기가 남는다. 브레이크/진행량 메트릭에 적용.
- 작은 1차 항 추가: \(g(x)=x^2 + \lambda x\) 또는 \(x^2 + \lambda\,\text{softplus}(x)\) 형태로 정의해 \(x=0\)에서도 기울기 \(\approx \lambda\)를 확보한다. \(\lambda\)는 1e-4~1e-3 수준으로 작게.
- 보조 메트릭 혼합: 직진·정지 구간에서 주 메트릭이 0이면 \(g = \alpha g_\text{curv} + (1-\alpha) g_\text{progress}\) 처럼 진행/속도 등 보조 항을 섞어 기울기를 유지한다.

---

## 4. 작업 우선순위 및 역할 분담

### 4.1 Phase 1 – 최우선

1. **a. 장면 구간 5개 선정**
2. **b. Sim-Lingo 추론 베이스라인 코드 구현**

- a와 b는 서로 독립적이지만, 둘 다 이후 실험의 기반이므로 **동급 최우선**으로 본다.
- Scene이 고정되지 않으면 c–e 메소드의 성능/설명 차이를 공정하게 비교하기 어렵다.
- Sim-Lingo 추론 베이스라인이 없으면 어떤 설명 메소드도 실제로 동작할 수 없다.
- Sim-Lingo 입력 해상도에 맞는 크롭/리사이즈는 **Scene 선정 이후** 적용한다.

**일정 전략:**

- Phase 1에서 **a와 b를 최대한 병렬로 진행**
- a와 b가 모두 “동작 가능한 상태”가 되어야
  - c–e 메소드들을 안정적으로 개발/테스트할 수 있다.

---

### 4.2 역할 분담 (현재 기준)

> 이름은 예시이며, 실제 담당자는 팀 내에서 조정 가능.

#### 담당자 A – 장면 구간 5개 선정(a) + 데이터 구조 정의 (담당자 : 동현)

- 시나리오를 여러 번 돌려보고, 대표성이 좋고 난이도가 있는 Scene 5개 선정
- 위에서 정의한 Scene 조건(사고 전/중/후, 히트맵 시각적으로 의미 있게)을 만족하도록 선택
- 각 Scene에 대해:
  - 프레임 자르는 기준 (시간/이벤트)
  - 파일명 규칙, 디렉토리 구조
  - 간단한 텍스트 설명
  를 문서화
- 이로써 **데이터 I/F 규약**을 확정

#### 담당자 B – Sim-Lingo 추론 베이스라인(b) 리드 (담당자 : 도형)

- Sim-Lingo 모델 로딩, 전처리/후처리, forward pass 구현
- Scene 디렉토리를 입력으로 받아:
  - 이미지 로딩 → 리사이즈/크롭 → Sim-Lingo forward
  - 필요한 토큰/trajectory/attention 등을 일관된 포맷으로 반환
- 간단한 API 정의 예:
  - `run_inference(scene_dir) -> {images, vision_tokens, text_tokens, action_tokens, trajectory, ...}`

#### 담당자 C – 논문 코드/메소드 구조 사전 분석 (Phase 2 준비) (담당자 : 재혁)

- Generic Attention, Transformer Attribution, ViT Attention 관련 논문과 공개 코드 읽기
  - 코드 구조, 입력 형태, 레이어/토큰 접근 방식 정리
- 레포지토리에 사전 스켈레톤 생성:
  - `methods/generic_attention.py`
  - `methods/transformer_attribution.py`
  - `methods/vit_attention.py`
- 함수 시그니처만 먼저 정의해 두어, Phase 1 종료 후 구현을 바로 시작 가능하게 준비

## 참고 논문 링크 (arXiv)

- SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment  
  - https://arxiv.org/abs/2503.09594

- Generic Attention-Model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers  
  - https://arxiv.org/abs/2103.15679

- Transformer Interpretability Beyond Attention Visualization  
  - https://arxiv.org/abs/2012.09838

- Quantifying Attention Flow in Transformers (Attention Rollout / Attention Flow)  
  - https://arxiv.org/abs/2005.00928

- “Raw Attention”은 독립 논문 제목이 아니라, 위 논문들에서 gradient/rollout을 쓰지 않은 attention weight 자체를 가리키는 비교 기준 용어로 사용됨.

---

### 4.3 Phase 2 이후 요약

Phase 1이 완료되었다고 가정하면:

- **담당자 B + C**
  - Generic Attention 기반 Ours + 텍스트 방식 2개 메소드 (c) 구현
  - 공통 규약:
    - feed-forward 방향에서 각 헤드별 attention map 추출 방식을 통일
    - grid 단위 → 원본 이미지 픽셀로 히트맵 투영 시 interpolation 방식 통일
  - B, C가 실제 코딩 담당
  - A는 코드 리뷰 및 두 메소드 간 **일관성 유지**(설계/인터페이스 관점) 담당
- **담당자 A + B + C**
  - Transformer Attribution 1개(d), ViT Attention 2개(e) 구현과 통합 실행(f) 마무리를 공동으로 진행

---

## 5. 핵심 정리 (Codex가 특히 기억해야 할 것)

1. **우리는 “policy-to-vision”이 아니라 “action-to-vision” 설명을 목표로 한다.**
   - input 이미지 -> simlingo 추론 -> Sim-Lingo 액션 토큰 → trajectory → kinematic scalar \(y_t\) → Transformer-MM-Explainability → vision relevance map → heatmap
2. 추론/설명 I/F: `simlingo_inference_baseline.py`는 `scene_dir`를 입력받아 `.pt` payload를 만든다. Generic Attention 등은 이 `.pt`를 입력(`pt_dir`)으로 받아 히트맵을 만든다. ViT/다른 방법도 가능하면 공통 추론 출력을 재사용한다.
3. Sim-Lingo 추론 로직은 **한 곳(run_inference 등)에만 구현**하고, 모든 메소드는 이 모듈의 출력(토큰/trajectory/attention/grad/텍스트)을 사용한다.
4. 운동학 함수(종방향/횡방향)는 **플러그인/모듈형 구조**로 구현하되, 0 부근 기울기 소멸을 피하기 위해 softplus/작은 1차 항/보조 메트릭 혼합 등을 활용한다.
5. 최종적으로 `run_all_methods.py` 같은 스크립트에서 **원터치 실행**으로 5개 메소드의 히트맵을 모두 생성할 수 있도록 해야 한다.
6. 코드 변경 시, 본 문서(`codex_context.markdown`)에 적힌 맥락과 달라진 부분이 있으면 최신 맥락에 맞게 필요한 부분만 업데이트한다(내용 전체 덮어쓰기·삭제 등 급진적 변경은 피하고, 차이 나는 부분만 신중히 보정).

기본 언어는 한국어로 하며, 정중하고 신중한 말투를 사용해야한다.
이 파일의 내용은 **코드 어시스턴트에게 주입되는 상시 컨텍스트**로 사용되며,  
레포지토리 내 모든 구현은 이 맥락과 규약을 따르는 것을 기본 전제로 한다.
