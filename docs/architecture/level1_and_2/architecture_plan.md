# [MC-nGNN for GoG] Level 1, 2 분리 및 아키텍처 리팩토링 기획안

## 1. Fraud-Detection 평가 지표 분석 및 방향성

### 1.1 F1-score
* **개념:** 정밀도(Precision)와 재현율(Recall)의 조화평균. 극단적인 모델(예: 모두 사기로 예측하여 Recall은 100%지만 Precision이 0%인 경우)에 페널티를 주어 소수 클래스(사기 거래) 탐지력을 단일 수치로 보여줌.
* **MC 기법 적용의 장점 (논문 Contribution):** 기존 GoG는 데이터 희소성(Sparsity)으로 인해 은닉된 사기 패턴 학습에 한계가 있어 Recall이 낮음. MC(Monte Carlo) 기반 샘플링은 사기 데이터의 복잡한 위상학적 패턴을 증폭시킴. "정상 거래 탐지력은 유지하면서 사기 거래 재현율을 대폭 끌어올려 소수 클래스(Minor Class) F1-score가 향상되었다"는 점을 강조.

### 1.2 ROC-AUC
* **개념:** 임계값과 무관하게 모델이 '정상'과 '이상'을 얼마나 잘 분별(Discrimination)하는지 측정. 1에 가까울수록 신뢰도가 높음.
* **MC 기법 적용의 장점:** 기존 모델은 애매한 경계의 노드에 잘못된 확신(Overconfidence)을 가짐. MC 샘플링 앙상블을 통해 정교하게 보정된 확률값(Calibrated Probability)을 얻고, 불확실성(Uncertainty)을 측정하여 오탐지(False Positive)를 줄임.

---

## 2. 기존 베이스라인(DOMINANT 등) 실험 결과 및 한계

### 2.1 DOMINANT 결과 분석
* AUC: 0.45 ~ 0.61 수준 / AP: 0.14 ~ 0.21 수준
* 하이퍼파라미터에 민감하며 불안정함. (가장 좋은 설정: `hid_dim=8, lr=0.005`)
* 모델 크기를 키울수록(hid_dim=16, 32) 오히려 성능 하락 (과적합/최적화 문제).
* 에포크(Epoch)를 늘린다고 지속적으로 개선되지 않으므로 Early Stopping 필수.

### 2.2 Unc=0.0000 (불확실성 추정 실패) 원인 진단
MC 샘플링 시 Uncertainty가 0이 나오는 문제에 대한 원인과 해결책:
| 모델 | 상태 및 원인 | 해결책 |
|---|---|---|
| **DOMINANT, GAE** | Dropout은 존재하나 `p=0.0`으로 비활성화됨 | Dropout 확률 수동 주입 및 직접 forward 패스 수행 |
| **DONE, COLA** | decision_function이나 BatchNorm 등에 의해 랜덤성(Variance)이 이미 정상 발생함 (`Unc > 0`) | 기존 decision_function 유지 |
| **AnomalyDAE** | Dropout도 없고 BatchNorm도 없음 | Dropout 수동 삽입 필수 |

---

## 3. 핵심 아키텍처 개편: Level 1과 Level 2의 이원화 (Hierarchical Fraud Stack)

기존 모델은 개별 서브그래프 내부 구조 분석과 그래프 간의 관계 분석이 뒤섞여 있어 학습이 모호했습니다. 이를 두 개의 독립적인 GNN 레벨로 분리합니다.

### 3.1 엣지(Edge)의 명확한 분리
두 계층에서 사용하는 Edge의 성격이 완전히 다름을 코드로 명확히 해야 합니다.

| 구분 | Level 1: graph_individual edge | Level 2: graph_of_graph edge |
|---|---|---|
| **연결 대상** | 노드 ↔ 노드 (한 서브그래프 내부) | 그래프 ↔ 그래프 (서브그래프 간) |
| **노드 의미** | 블록체인 주소, 스마트 컨트랙트, 트랜잭션 등 | 개별 서브그래프 자체 (Meta-node) |
| **목적** | 서브그래프 내부 구조(Fan-in/out 등) 학습 | 캠페인, 자금 세탁 등 그래프 간 관계 학습 |
| **메시지 단위** | Node-level Message Passing | Graph-level Message Passing |
| **Feature** | 노드 Feature | Level 1이 출력한 서브그래프 임베딩 벡터 |

### 3.2 아키텍처 데이터 흐름 (Data Flow Diagram)
```text
[Raw Blockchain Data]
       │
       ▼
[Preprocess: Temporal Split / Label Check / Normalization]
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
[Level 1 Dataset Builder]                [Level 2 Relation Builder]
       │                                 (Temporal Links between subgraphs)
       ▼                                          │
[Level 1 Encoder (Local GNN)]                     │
(Extracts individual subgraph embedding)          │
       │                                          │
       ├─────────────────┐                        │
       ▼                 ▼                        ▼
[Level 1 Fraud Head]   [Embedding Cache] ──▶ [Level 2 GoG Builder]
(Node/Graph score)                                │
       │                                          ▼
       │                                 [Level 2 Encoder (nGNN / Meta GNN)]
       │                                 (Relation-aware embedding)
       │                                          │
       │                                          ▼
       │                                 [Level 2 Fraud Head]
       │                                          │
       └─────────────────┬────────────────────────┘
                         ▼
                  [Fusion Module]
        (Score Fusion / Embedding Fusion)
                         │
                         ▼
             [Final Fraud Prediction]
```

## 4. 모듈별 상세 설계 원칙

### 4.1 Level 1: 내부 구조 인코더
* **입력:** 개별 Transaction subgraph, 노드/엣지 피처 level 1, 2 분리.pdf]
* **역할:** 서브그래프 내부의 이상 패턴(Fan-in, Fan-out, Hub 집중 등) 포착 level 1, 2 분리.pdf]
* **출력:** 서브그래프 임베딩(Embedding), 1차 Fraud Score level 1, 2 분리.pdf]
* **구현:** GIN, GraphSAGE, GAT 중 택 1 + Readout(mean/max pooling) level 1, 2 분리.pdf]

### 4.2 Level 2: 관계 맥락 보강기
* **입력:** Level 1에서 추출된 서브그래프 임베딩 + 시간적/메타적 릴레이션 엣지 level 1, 2 분리.pdf]
* **역할:** 조직적 캠페인, 자금 세탁 경로 등 서브그래프 간 상호작용 학습 level 1, 2 분리.pdf]
* **출력:** 맥락이 반영된(Context-aware) 임베딩 및 최종 2차 Fraud Score level 1, 2 분리.pdf]

### 4.3 Fusion 계층
* **Score Fusion:** L1 Score와 L2 Score를 단순 가중합(Weighted Sum) 또는 작은 MLP로 결합 (초기 MVP용) level 1, 2 분리.pdf]
* **Embedding Fusion:** L1 임베딩과 L2 임베딩을 Concat 후 최종 Prediction (가장 추천) level 1, 2 분리.pdf]

### 4.4 MC 및 nGNN 확장
* 핵심 코어 파이프라인(L1 -> L2 -> Fusion)이 고정된 후, **플러그인(Adapter/Strategy Pattern)** 형태로만 부착. 코드 내부 분기(`if use_mc:`) 금지. level 1, 2 분리.pdf]

---

## 5. 데이터 엔지니어링 및 최적화 (8GB GPU 제한 환경)

* **Wei 값 정규화 (필수):** Ethereum/Polygon의 트랜잭션 값은 단위가 `wei`($10^{18}$)이므로 Float32 범위를 초과해 `NaN/Inf` 에러를 유발함. 반드시 전처리 단계에서 `log1p` 변환을 적용해야 함. level 1, 2 분리.pdf]
* **오프라인 캐싱 (Pickle):** 원본 파싱 비용 절감을 위해 전처리된 그래프 객체를 `.pkl`로 캐싱. 모델 로직 안에서 파일 I/O 금지. level 1, 2 분리.pdf]
* **메모리 최적화:** L1 임베딩 오프라인 캐시 적용 (L2 학습 시 L1은 Frozen 상태로 사용), Gradient Accumulation, Mixed Precision(`autocast`) 적극 활용. level 1, 2 분리.pdf]

---

## 6. 디렉토리 구조 및 리팩토링 목표
기존 레거시 코드를 다음과 같이 재구성. (Legacy 모듈 직접 Import 절대 금지) level 1, 2 분리.pdf]

```text
MC_and_nGNN_for_GoG/
├── configs/                   # YAML 기반 실험 설정
├── data/                      # 원본 및 처리된 데이터
├── artifacts/                 # 저장된 임베딩(Cache), 모델 체크포인트
├── legacy/                    # 이전 원본/비교용 코드 (수정 안 함)
└── src/gog_fraud/             # 핵심 소스 코드
    ├── data/
    │   ├── preprocess/        # Temporal split, Data normalization (log1p)
    │   ├── level1/            # Level1 전용 Dataset/Loader
    │   └── level2/            # Level2 전용 GoG/Relation Builder
    ├── models/
    │   ├── level1/            # GNN Encoder, Structural Features
    │   ├── level2/            # Meta-GNN (GATv2 등)
    │   ├── fusion/            # Fusion Module
    │   └── extensions/        # MC 및 nGNN 플러그인 폴더
    ├── training/              # Level별 개별 Trainer 및 Joint Trainer
    └── pipelines/             # 실행 진입점 (train_level1.py, run_benchmark.py 등)

```