---

# 3) `mc_pipeline_and_experiment_plan.md`


# MC Pipeline and Experiment Plan

이 문서는 Monte Carlo(MC)를 기존 benchmark 및 refactoring 파이프라인에 어떻게 단계적으로 편입할지 설명합니다.

---

## 개발 전제

현재 완료된 기반:
- Legacy baseline 검증 가능
- Revision Level 1 검증 가능
- Revision Level 1 + Level 2 검증 가능
- Revision Full(Fusion) 검증 가능

이제 MC 개발은 새로운 모델을 처음부터 만드는 작업이 아니라,
**기존 파이프라인 위에 실험 가능한 확장 레이어를 추가하는 작업**으로 진행합니다.

---

## 단계별 실험 로드맵

PDF에서 제안된 실험 구분은 아래 4단계입니다.

| Phase | 이름 | edge dropout | model MC | 설명 |
|---|---|---:|---:|---|
| 1 | Base deterministic nGNN | ✗ | ✗ | clean baseline |
| 2 | nGNN + Model MC | ✗ | ✓ | evaluation 시 multi-forward |
| 3 | nGNN + Data MC | ✓ | ✗ | training 시 edge perturbation |
| 4 | nGNN + Data + Model MC | ✓ | ✓ | 최종 결합 버전 |

현재 상황에 맞게 이를 더 일반화하면 다음과 같이 적용할 수 있습니다.

| Phase | 이름 | 적용 범위 | 목적 |
|---|---|---|---|
| 1 | Base deterministic | L1 / L2 / Fusion | MC 없는 기준선 확보 |
| 2 | Model MC | 우선 L1, 이후 L2/Fusion | uncertainty 및 calibration 효과 측정 |
| 3 | Data MC | 우선 L1, 이후 relation graph 확장 | 구조 변형에 대한 강건성 확보 |
| 4 | Hybrid MC | L1 + L2 + Fusion | 최종 통합 성능 확인 |

---

## 추천 개발 순서

### Step 1. 평가 기준 이식 및 통일

PDF 제안:
- GoG 소스에서 사용하던 평가 지표 및 로그 저장 로직을 동일하게 맞춤
- F1, AUC 중심으로 정상 동작을 먼저 확인

현재 리팩토링 구조에서는 다음을 먼저 고정하는 것이 좋습니다.

- 공통 evaluator
- 공통 metrics 집계기
- 공통 artifact 저장 방식
- baseline / revision / MC 실험의 비교 가능 포맷

---

### Step 2. Base model 안정화 확인

MC를 붙이기 전 다음이 먼저 보장되어야 합니다.

- forward / backward 오류 없음
- dimension mismatch 없음
- Level 1 embedding cache 정상 생성
- Level 2 relation graph 입력 정상
- Fusion 입력 정합성 확인

이 단계가 불안정하면 MC 성능 해석이 무의미해집니다.

---

### Step 3. Model MC 먼저 연결

가장 먼저 구현할 것은 evaluation-time multi-forward입니다.

권장 대상 순서:
1. Level 1 only
2. Level 1 + Level 2
3. Full Fusion

이유:
- Level 1 영향력이 가장 큼
- 구현 난이도가 낮음
- uncertainty 변화 해석이 쉬움

---

### Step 4. Data MC 추가

학습 단계에서 edge perturbation 또는 edge dropout을 적용합니다.

우선순위:
1. Level 1 transaction subgraph
2. Level 2 relation graph
3. joint setting

주의사항:
- Level 2에서는 temporal/meta relation의 의미가 손상되지 않도록 dropout policy를 조심해야 함
- relation edge 전체를 무작위로 제거하기보다 중요도 기반 샘플링이 바람직함

---

### Step 5. 자동화 실험 파이프라인 구성

PDF 기준으로 전체 실험 자동화 진입점이 필요합니다.

권장 엔트리포인트:
- `train.py`
- `evaluate.py`
- `run_experiment.py`

리팩토링 버전에서는 예를 들어 아래 구성이 자연스럽습니다.

- `src/gog_fraud/pipelines/train_level1.py`
- `src/gog_fraud/pipelines/train_level2.py`
- `src/gog_fraud/pipelines/run_fraud_benchmark.py`
- `src/gog_fraud/pipelines/run_mc_benchmark.py`

---

## 구성 파일 권장안

### Configs

예시:
- `configs/base.yaml`
- `configs/benchmark/level1.yaml`
- `configs/benchmark/level2.yaml`
- `configs/mc/model_mc.yaml`
- `configs/mc/data_mc.yaml`
- `configs/mc/hybrid_mc.yaml`

핵심 파라미터:
- `mc_samples`
- `dropout_p`
- `edge_dropout_p`
- `adaptive_sampling`
- `uncertainty_metric`
- `enable_mc`
- `enable_data_mc`

---

## 실행 흐름 예시

### Model MC 실험 흐름

1. deterministic checkpoint 로드
2. evaluation mode 진입
3. dropout layer만 활성화
4. 동일 batch에 대해 다중 forward 수행
5. 평균 score 및 uncertainty 계산
6. F1 / AUC / AP와 uncertainty 통계 저장

### Data MC 실험 흐름

1. train loader에 stochastic graph transform 연결
2. edge perturbation/dropout 적용
3. deterministic 또는 hybrid 방식으로 학습
4. validation에서 성능 추적
5. 필요시 inference에서 model MC 추가 적용

---

## 데이터/캐시/아티팩트 관리

PDF의 방향성과 현재 refactoring 구조를 종합하면 다음이 중요합니다.

### 오프라인 캐싱

- 전처리된 graph 객체 캐시
- Level 1 embedding cache
- relation graph cache
- experiment result cache

목적:
- 재실행 시간 절감
- 디버깅 반복 비용 감소
- MC 샘플링 실험 반복 가능성 확보

### 모델 로직 안에서 파일 I/O 금지

파일 저장/로드는 파이프라인 또는 data layer에서 담당하고,
model forward 안에 I/O를 넣지 않는 것이 좋습니다.

---

## 8GB GPU 환경 고려사항

PDF에서 강조된 최적화 지점은 다음과 같습니다.

- dynamic batching
- multiprocessing pipeline
- Level 1 embedding offline cache
- Level 2 학습 시 Level 1 freeze
- gradient accumulation
- mixed precision
- 불필요한 full-graph exact computation 회피

추가로 MC 단계에서는 아래가 중요합니다.

- `mc_samples`를 과도하게 키우지 않기
- high-risk graph 우선 적용
- smoke mode / debug mode 유지
- sample-wise 결과를 모두 GPU에 쌓지 말고 CPU로 이동 후 집계

---

## 체인별 실험 전략

PDF에서는 Ethereum, Polygon, BSC 등 체인별 대규모 비교 실험을 염두에 두고 있습니다.

권장 실험 순서:
1. smoke set
2. 소규모 고정 split
3. 단일 체인 full run
4. 멀티 체인 비교 run

분석 포인트:
- class imbalance 차이
- relation graph density 차이
- MC 적용 시 false positive 감소 여부
- chain-specific overfitting 완화 여부

---

## 최종 실험 출력

각 실험은 최소한 아래 항목을 저장하도록 설계하는 것이 좋습니다.

- config snapshot
- checkpoint path
- F1
- ROC-AUC
- AP
- inference latency
- uncertainty summary
- chain name
- phase name
- notes / error flag

---

## 요약

MC 파이프라인은 다음 흐름으로 진행하는 것이 적절합니다.

1. deterministic 기준선 유지
2. Model MC 먼저 붙이기
3. Data MC 추가
4. Hybrid MC로 통합
5. 자동화된 benchmark 파이프라인으로 체인별 비교 실험 수행

핵심은 MC를 독립 실험 가능한 단계로 쪼개고,
각 단계에서 성능 향상과 원인을 분리해서 검증하는 것입니다.
