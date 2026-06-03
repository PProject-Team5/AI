# Minit-ML 파이프라인 기술 문서

> 2단계 캐스케이드 추론 + 지식 증류(Knowledge Distillation) 구조

---

## 목차

1. [개요](#1-개요)
2. [전체 흐름 요약](#2-전체-흐름-요약)
3. [Stage 1 — 경량 모델 (1차 판단)](#3-stage-1--경량-모델-1차-판단)
4. [Suspicious Zone — 2차 라우팅 기준](#4-suspicious-zone--2차-라우팅-기준)
5. [Stage 2 — 정밀 모델 (2차 판단)](#5-stage-2--정밀-모델-2차-판단)
6. [지식 증류 — Stage 2 → Stage 1 피드백 루프](#6-지식-증류--stage-2--stage-1-피드백-루프)
7. [추론 시점의 동기/비동기 분리](#7-추론-시점의-동기비동기-분리)
8. [설정값 레퍼런스](#8-설정값-레퍼런스)
9. [소스 코드 맵](#9-소스-코드-맵)

---

## 1. 개요

Minit-ML은 파일 업로드 시 **NSFW 이미지**와 **PE 악성코드**를 탐지하는 콘텐츠 모더레이션 ML 파이프라인입니다.

핵심 아이디어는 세 가지입니다:

1. **빠른 1차 모델(Stage 1)**로 대부분의 파일을 즉시 판정한다.
2. 확신이 없는 **애매한 파일만** 무거운 **2차 모델(Stage 2)**로 보낸다.
3. 2차 모델의 판정 결과를 **증류(Distillation)**하여 1차 모델을 점진적으로 개선한다.

이 구조의 장점은 다음과 같습니다:

| 관점 | 효과 |
|------|------|
| **레이턴시** | 전체 트래픽의 대부분이 Stage 1에서 ~50ms 이내로 처리 |
| **정확도** | Stage 2의 정밀 판정이 애매한 케이스를 커버 |
| **자기 개선** | 시간이 지날수록 Stage 1이 Stage 2의 판정 능력을 흡수하여 Stage 2 호출 빈도가 감소 |

---

## 2. 전체 흐름 요약

```
파일 업로드
    │
    ▼
┌──────────────────────────────┐
│  파일 타입 감지               │  magic bytes로 image / pe / unknown 분류
│  (detect_file_type)          │
└──────────┬───────────────────┘
           │
     ┌─────┴─────┐
     │           │
  image?       pe?        ───→ unknown이면 즉시 pass
     │           │
     ▼           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: 경량 추론                           │
│                                                                     │
│  [이미지]                         [PE 실행파일]                      │
│  CLIP ViT-B/32 임베딩              pefile 임포트 테이블 추출          │
│  → 512차원 벡터                    → 상위 1000개 API 바이너리 벡터    │
│  → MLP Head (sigmoid)             → XGBoost (logistic)              │
│  → nsfw_prob 출력                  → malware_prob 출력               │
│                                                                     │
│  판정:                             판정:                             │
│  prob < 0.3  →  ✅ safe            prob < 0.2  →  ✅ safe           │
│  prob > 0.7  →  🚫 nsfw           prob > 0.8  →  🚫 malware        │
│  0.3~0.7     →  ⚠️ suspicious     0.2~0.8     →  ⚠️ suspicious     │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
              ✅ safe      ⚠️ suspicious   🚫 위험
                 │             │             │
           즉시 통과     Stage 2로 전달    즉시 차단
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        STAGE 2: 정밀 검증                           │
│                                                                     │
│  [이미지]                         [PE 실행파일]                      │
│  NudeNet (YOLOv8-nano ONNX)       EMBER PE Feature Extractor        │
│  → 18개 신체 부위 클래스 탐지       → 54차원 구조적 특징 추출          │
│  → EXPOSED 클래스 존재 여부         → LightGBM 분류                   │
│  → ~50ms/image                    → ~100ms/file                     │
│                                                                     │
│  결과: is_nsfw (bool)             결과: is_malware (bool)            │
│        nsfw_score (float)                malware_prob (float)        │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
               최종 판정             증류 버퍼에 수집
               (safe / block)        (embedding, label) 페어 저장
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     지식 증류 (Distillation)                         │
│                                                                     │
│  수집된 (Stage1 특징, Stage2 라벨) 쌍으로 Stage 1 모델을 재학습       │
│                                                                     │
│  [NSFW]   embedding + soft label → MLP Head 재학습 (BCE + α 가중)   │
│  [Malware] import features + pseudo label → XGBoost 재학습           │
│                                                                     │
│  → 학습 완료 후 Stage 1 모델 가중치를 교체 (hot-reload)              │
│  → 시간이 지날수록 Stage 1의 suspicious zone이 좁아짐               │
│  → Stage 2 호출 빈도가 점진적으로 감소                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 1 — 경량 모델 (1차 판단)

Stage 1의 역할은 **확실한 파일을 빠르게 걸러내는 것**입니다. 정확도보다 속도가 우선이며, "확신이 없으면 넘긴다"는 원칙으로 동작합니다.

### 3-1. NSFW 이미지 탐지

| 항목 | 내용 |
|------|------|
| **모델** | CLIP ViT-B/32 (frozen) + 학습 가능한 MLP Head |
| **입력** | 이미지 파일 |
| **처리** | CLIP 이미지 인코더로 512차원 임베딩 추출 → MLP Head (Linear→ReLU→Dropout ×5 → Linear) → Sigmoid |
| **출력** | `nsfw_prob` (0.0~1.0), `label` (safe/suspicious/nsfw) |
| **레이턴시** | ~41ms (임베딩 추출 ~40.9ms + 추론 ~0.5ms) |
| **코드** | `src/stage1/clip_nsfw.py` → `CLIPNSFWDetector.predict()` |

**MLP Head 구조** (`CLIPNSFWHead`):
```
512 → 1024 → 2048 → 1024 → 256 → 128 → 16 → 1
      (ReLU + Dropout 0.2 반복)
```

> Slim 변형(`CLIPNSFWHeadSlim`)도 존재하며, 512 → 256 → 64 → 1 로 더 가볍습니다.

### 3-2. PE 악성코드 탐지

| 항목 | 내용 |
|------|------|
| **모델** | XGBoost Booster (binary:logistic) |
| **입력** | PE 실행파일 (.exe, .dll) |
| **처리** | `pefile`로 임포트 테이블 파싱 → 상위 1000개 Windows API 함수의 존재 여부를 바이너리 벡터화 → XGBoost 추론 |
| **출력** | `malware_prob` (0.0~1.0), `label` (safe/suspicious/malware) |
| **레이턴시** | ~58ms (피처 추출 ~54.2ms + 추론 ~4.1ms) |
| **코드** | `src/stage1/pe_lite.py` → `PELiteDetector.predict()` |

**피처 벡터 예시** — `GetProcAddress`, `LoadLibraryA`, `VirtualAlloc`, `WriteProcessMemory` 등 악성코드에서 자주 사용되는 Windows API의 호출 여부를 0/1로 인코딩한 1000차원 벡터.

---

## 4. Suspicious Zone — 2차 라우팅 기준

Stage 1의 출력 확률이 **확신 영역**에 들지 않으면 `suspicious`로 분류되고, 이 파일만 Stage 2로 전달됩니다.

```
                    NSFW                              Malware
                                                      
  0.0 ──────── 0.3 ──────── 0.7 ──────── 1.0    0.0 ──── 0.2 ──────────── 0.8 ──── 1.0
  ├── safe ────┤            ├──── nsfw ───┤      ├ safe ──┤                ├ malware ┤
               └─ suspicious┘                            └── suspicious ──┘
               (→ Stage 2)                               (→ Stage 2)
```

| 도메인 | safe 범위 | suspicious 범위 | 위험 범위 |
|--------|----------|----------------|----------|
| NSFW | prob < **0.3** | 0.3 ≤ prob ≤ **0.7** | prob > 0.7 |
| Malware | prob < **0.2** | 0.2 ≤ prob ≤ **0.8** | prob > 0.8 |

이 임계값은 `config.yaml`의 `stage1.nsfw.suspicious_range`와 `stage1.malware.suspicious_range`에서 설정할 수 있습니다.

**왜 비대칭인가?**
- Malware의 suspicious zone (0.2~0.8)이 NSFW (0.3~0.7)보다 넓습니다.
- 악성코드의 경우 오탐(false negative)의 비용이 높기 때문에 더 보수적으로 설정합니다.

---

## 5. Stage 2 — 정밀 모델 (2차 판단)

Stage 2는 **정확도가 최우선**입니다. 느리지만 더 강력한 모델로 최종 판정을 내립니다.

### 5-1. NudeNet (NSFW 정밀 검증)

| 항목 | 내용 |
|------|------|
| **모델** | YOLOv8-nano (ONNX) |
| **방식** | 18개 신체 부위 클래스에 대한 객체 탐지 |
| **판정 기준** | EXPOSED 클래스(`BUTTOCKS_EXPOSED`, `FEMALE_BREAST_EXPOSED` 등 9종) 중 하나라도 탐지되면 `is_nsfw = True` |
| **신뢰도 임계값** | confidence ≥ 0.3 인 탐지만 유효 |
| **레이턴시** | ~50ms/image |
| **코드** | `src/stage2/nudenet_nsfw.py` → `NudeNetDetector.predict()` |

### 5-2. EMBER + LightGBM (Malware 정밀 검증)

| 항목 | 내용 |
|------|------|
| **모델** | LightGBM Booster |
| **방식** | PE 헤더, 섹션, 리소스, 임포트/익스포트 등에서 54차원 구조적 특징 추출 → LightGBM 분류 |
| **추출 특징** | FILE_HEADER, OPTIONAL_HEADER, 섹션 엔트로피, 임포트 DLL/함수 수, 리소스 엔트로피 등 |
| **레이턴시** | ~100ms/file |
| **코드** | `src/stage2/ember_malware.py` → `EmberMalwareDetector.predict()` |

### Stage 2 판정 후 동작

```python
# pipeline.py의 실제 로직 (NSFW 예시)
if stage2_result["is_nsfw"]:
    result.nsfw_prob = max(result.nsfw_prob, stage2_result["nsfw_score"])
    result.nsfw_label = "nsfw"
    result.blocked = True       # → 차단
else:
    result.nsfw_label = "safe"
    result.blocked = False      # → 통과
```

- Stage 2가 양성으로 판정하면 → **최종 차단** (prob은 Stage 1과 Stage 2 중 더 높은 값)
- Stage 2가 음성으로 판정하면 → **최종 통과** (1차 suspicious를 safe로 override)

---

## 6. 지식 증류 — Stage 2 → Stage 1 피드백 루프

이 파이프라인의 핵심 차별점입니다. Stage 2로 보내진 파일의 판정 결과를 **교사 신호(teacher signal)**로 사용하여 Stage 1 모델을 재학습시킵니다.

### 6-1. 증류 데이터 수집

Stage 2 판정이 완료될 때마다, 아래 데이터 페어를 증류 버퍼(JSONL 파일)에 자동으로 기록합니다:

| 도메인 | 저장되는 데이터 | 버퍼 파일 |
|--------|----------------|----------|
| NSFW | (CLIP 512d 임베딩, Stage2 nsfw_score) | `data/nsfw_distill_buffer.jsonl` |
| Malware | (PE import 1000d 벡터, EMBER malware_prob) | `data/malware_distill_buffer.jsonl` |

```python
# pipeline.py — Stage 2 결과 수집 시점
self._get_nsfw_distiller().collect_sample(
    embedding=stage1_result["embedding"],       # Stage 1이 추출한 특징
    stage1_prob=stage1_result["nsfw_prob"],      # Stage 1의 원래 예측
    stage2_result=stage2_result,                 # Stage 2의 정밀 판정
)
```

### 6-2. NSFW 증류 과정

```
              교사(Teacher): NudeNet
                     │
                     │ soft label (nsfw_score)
                     ▼
   ┌──────────────────────────────────┐
   │  증류 학습 (nsfw_distill.py)     │
   │                                  │
   │  입력: CLIP 임베딩 (512d)        │
   │  라벨: Stage 2 soft label        │
   │                                  │
   │  손실함수:                       │
   │   BCE(student_logit, soft_label) │
   │   × α (증류 샘플 가중치 0.5)     │
   │                                  │
   │  + 원본 학습 데이터 (있을 경우)   │
   │    × (1 - α)                     │
   │                                  │
   │  옵티마이저: Adam (lr=0.0001)    │
   │  에폭: 50                        │
   │  배치 크기: 64                   │
   └────────────────┬─────────────────┘
                    │
                    ▼
         학생(Student): MLP Head 가중치 업데이트
         → models/clip_nsfw_head.pth 저장
         → Stage 1 모델 hot-reload
```

**핵심 설계 포인트:**

- **Soft Label 사용**: Stage 2의 `nsfw_score`를 그대로 사용하여 hard 0/1 라벨보다 더 많은 정보를 전달합니다.
- **α 가중 혼합**: 증류 데이터(α=0.5)와 원본 학습 데이터(1-α=0.5)를 혼합하여 기존 지식 망각(catastrophic forgetting)을 방지합니다.
- **버퍼 초기화**: 증류 완료 후 버퍼를 비워서 동일 데이터의 반복 학습을 방지합니다.

### 6-3. Malware 증류 과정

```
              교사(Teacher): EMBER + LightGBM
                     │
                     │ pseudo label (malware_prob)
                     ▼
   ┌──────────────────────────────────┐
   │  증류 학습 (malware_distill.py)  │
   │                                  │
   │  입력: PE import 벡터 (1000d)    │
   │  라벨: EMBER malware_prob        │
   │                                  │
   │  원본 CSV 데이터 (weight=1.0)    │
   │  + 증류 데이터 (weight=0.3)      │
   │                                  │
   │  모델: XGBoost                   │
   │   - max_depth=6, eta=0.05        │
   │   - 300 rounds, early stopping   │
   │                                  │
   │  최소 샘플 요구량: 100개          │
   └────────────────┬─────────────────┘
                    │
                    ▼
         학생(Student): XGBoost 모델 교체
         → models/pe_lite_xgb.json 저장
         → Stage 1 모델 hot-reload
```

**NSFW 증류와의 차이점:**

| 비교 항목 | NSFW | Malware |
|----------|------|---------|
| 학생 모델 | PyTorch MLP (gradient descent) | XGBoost (전체 재학습) |
| 라벨 형태 | Soft label (continuous) | Pseudo label (continuous) |
| 가중치 전략 | α=0.5 (증류:원본 동일 비중) | 0.3 (증류 데이터를 낮은 가중치로) |
| 최소 데이터 | 제한 없음 | 100개 이상 필요 |
| 학습 방식 | 미니배치 SGD | 전체 데이터 XGBoost train |

### 6-4. 증류의 장기적 효과

```
  시간 경과에 따른 변화:

  [초기]                              [증류 N회 후]
  
  Stage 1 suspicious zone: 넓음       Stage 1 suspicious zone: 좁아짐
  Stage 2 호출 비율: ~30%             Stage 2 호출 비율: ~5%
  
  0.0 ── 0.3 ────── 0.7 ── 1.0       0.0 ──── 0.45 ── 0.55 ── 1.0
  safe   │suspicious│  nsfw           safe     │susp│    nsfw
         └──Stage 2─┘                         └─S2─┘
         (넓은 불확실 영역)                    (좁은 불확실 영역)
```

Stage 1이 Stage 2의 판정 패턴을 학습하면서:
- 이전에 `suspicious`로 분류되던 파일들이 `safe` 또는 `nsfw/malware`로 확정 판정됨
- Stage 2 호출 빈도가 최대 **85% 감소** (벤치마크 기준)
- 전체 파이프라인의 평균 레이턴시가 Stage 1 수준으로 수렴

---

## 7. 추론 시점의 동기/비동기 분리

실제 서비스에서는 사용자 업로드 지연을 최소화하기 위해 **Stage 1은 동기, Stage 2는 비동기**로 동작합니다.

```
클라이언트 업로드 요청
         │
         ▼
  POST /scan (동기, Stage 1 only)
         │
    ┌────┴────────────┬──────────────┐
    │                 │              │
 verdict=0         verdict=2      verdict=1
 (safe)            (suspicious)   (danger)
    │                 │              │
 즉시 승인       임시 통과 +       즉시 차단
                 비동기 큐 등록
                      │
                      ▼
          VerificationQueueWorker (백그라운드)
          → Stage 2 정밀 검증 실행
          → 결과를 verification_results.jsonl에 기록
          → 증류 버퍼에 자동 수집
                      │
                      ▼
          GET /verify/{file_hash} (BE 스케줄러가 폴링)
          → pending / completed 상태 반환
          → completed면 최종 verdict 기반으로 DB 업데이트
```

**핵심**: `/scan` 엔드포인트에서는 `stage2_enabled = False`로 강제 설정하여 Stage 1만 실행합니다. Stage 2는 `VerificationQueueWorker`가 백그라운드 스레드풀(max_workers=2)에서 비동기 처리합니다.

---

## 8. 설정값 레퍼런스

`config.yaml`에서 모든 주요 파라미터를 제어할 수 있습니다.

```yaml
pipeline:
  block_threshold_nsfw: 0.7       # NSFW 즉시 차단 임계값
  block_threshold_malware: 0.8    # Malware 즉시 차단 임계값
  stage2_enabled: true            # Stage 2 활성화 여부

stage1:
  nsfw:
    clip_model: "ViT-B/32"        # CLIP 백본 모델
    slim: false                   # Slim MLP Head 사용 여부
    suspicious_range: [0.3, 0.7]  # 2차로 넘기는 확률 구간
  malware:
    suspicious_range: [0.2, 0.8]  # 2차로 넘기는 확률 구간

stage2:
  nsfw:
    confidence_threshold: 0.3     # NudeNet 탐지 최소 신뢰도
  malware:
    feature_version: 2            # EMBER 피처 버전

distillation:
  nsfw:
    temperature: 3.0              # (예약됨) 증류 온도 파라미터
    alpha: 0.5                    # 증류 데이터 가중치 (0~1)
    learning_rate: 0.0001         # MLP Head 학습률
    epochs: 50                    # 증류 학습 에폭 수
    batch_size: 64                # 미니배치 크기
  malware:
    pseudo_label_weight: 0.3      # 증류 pseudo-label 가중치
    min_samples: 100              # 증류 실행 최소 샘플 수
```

---

## 9. 소스 코드 맵

```
minit-ml/
├── pipeline.py                          # 파이프라인 오케스트레이터 (진입점)
│   ├── CascadePipeline                  #   2단계 캐스케이드 추론 관리
│   ├── VerificationQueueWorker          #   Stage 2 비동기 큐 워커
│   └── create_fastapi_app()             #   REST API 서버 생성
│
├── src/
│   ├── stage1/                          # ── 1차 경량 모델 ──
│   │   ├── clip_nsfw.py                 #   CLIP + MLP Head NSFW 탐지
│   │   │   ├── CLIPNSFWHead             #     MLP Head 모델 정의
│   │   │   ├── CLIPNSFWHeadSlim         #     경량 MLP Head 변형
│   │   │   └── CLIPNSFWDetector         #     추론 인터페이스
│   │   └── pe_lite.py                   #   PE Import + XGBoost 악성코드 탐지
│   │       ├── extract_pe_imports()     #     PE 임포트 피처 추출
│   │       ├── PELiteDetector           #     추론 인터페이스
│   │       └── train_pe_lite()          #     초기 학습 함수
│   │
│   ├── stage2/                          # ── 2차 정밀 모델 ──
│   │   ├── nudenet_nsfw.py              #   NudeNet YOLOv8 NSFW 탐지
│   │   │   └── NudeNetDetector          #     ONNX 추론 인터페이스
│   │   └── ember_malware.py             #   EMBER PE 구조 분석 + LightGBM
│   │       └── EmberMalwareDetector     #     추론 인터페이스
│   │
│   └── distillation/                    # ── 지식 증류 ──
│       ├── nsfw_distill.py              #   NSFW soft-label 증류
│       │   ├── DistillationBuffer       #     샘플 수집/관리
│       │   └── NSFWDistiller            #     MLP Head 재학습
│       └── malware_distill.py           #   Malware pseudo-label 증류
│           ├── MalwareDistillBuffer     #     샘플 수집/관리 (중복 제거 포함)
│           └── MalwareDistiller         #     XGBoost 재학습
│
├── models/                              # 학습된 모델 가중치
│   ├── clip_nsfw_head.pth               #   Stage 1 NSFW MLP Head
│   ├── pe_lite_xgb.json                 #   Stage 1 Malware XGBoost
│   └── ember_lgbm.txt                   #   Stage 2 Malware LightGBM
│
├── data/                                # 데이터 및 버퍼
│   ├── nsfw_distill_buffer.jsonl        #   NSFW 증류 샘플 버퍼
│   ├── malware_distill_buffer.jsonl     #   Malware 증류 샘플 버퍼
│   ├── verification_queue/              #   Stage 2 대기 큐 (파일 복사본)
│   ├── verification_results.jsonl       #   Stage 2 검증 완료 결과
│   └── top_1000_pe_imports.csv          #   PE 임포트 피처 원본 데이터
│
├── config.yaml                          # 파이프라인 설정
├── ARCHITECTURE.md                      # 아키텍처 개요 (영문)
├── INTEGRATION.md                       # 백엔드 연동 규격서 (한글)
└── PIPELINE.md                          # ← 이 문서
```
