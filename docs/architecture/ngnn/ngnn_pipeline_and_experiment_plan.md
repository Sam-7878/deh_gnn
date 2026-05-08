
# nGNN Pipeline and Experiment Plan

이 문서는 nGNN(Nested GNN)을 기존 benchmark 및 refactoring 파이프라인에
어떻게 단계적으로 편입할지 설명합니다.

---

## 개발 전제

현재 완료된 기반:
- Legacy baseline 검증 가능
- Revision Level 1 검증 가능
- Revision Level 1 + Level 2 검증 가능
- Revision Full(Fusion) 검증 가능

따라서 nGNN 개발은 완전히 새로운 학습 시스템을 만드는 것이 아니라,
**현재 구조 위에서 Level 1 encoder backend를 확장하고 단계적으로 통합하는 작업**으로 보는 것이 적절합니다.

---

## 단계별 실험 로드맵

권장 실험 단계는 아래와 같습니다.

| Phase | 이름 | Level 1 backend | Level 2 | Fusion | 목적 |
|---|---|---|---|---|---|
| 1 | Base deterministic | standard GNN | optional | optional | 기준선 확보 |
| 2 | L1-nGNN | nGNN | ✗ | ✗ | nGNN의 직접 효과 검증 |
| 3 | L1-nGNN + L2 | nGNN | ✓ | ✗ | relation graph 결합 효과 확인 |
| 4 | Full nGNN Revision | nGNN | ✓ | ✓ | 최종 통합 성능 측정 |
| 5 | nGNN + MC | nGNN | optional | optional | 이후 uncertainty/robustness 결합 |

현재 시점에서는 **Phase 2와 Phase 3**가 가장 중요합니다.

---

## 추천 개발 순서

### Step 1. Level 1 교체 가능한 인터페이스 정리

먼저 Level 1 encoder backend를 명시적으로 선택 가능하게 만들어야 합니다.

예시 config:
- `encoder_backend: gnn`
- `encoder_backend: ngnn`

핵심 요구사항:
- downstream code에서 출력 shape와 key가 유지될 것
- train/eval/predict path 모두 동일 인터페이스 사용
- cache key가 backend별로 분리될 것

---

### Step 2. rooted subgraph extraction 구현

nGNN 구현의 실제 시작점은 model이 아니라 **입력 단위 재구성**입니다.

구현 우선순위:
1. root 선정 규칙 고정
2. hop 기반 nested subgraph 추출
3. 추출 결과 캐시 저장
4. smoke mode에서 작은 subgraph만 사용 가능하게 설정

주의:
- extraction policy가 바뀌면 실험 결과 비교가 어려워짐
- dataset split과 무관하게 재현 가능한 deterministic extraction 옵션이 필요

---

### Step 3. L1-nGNN 단독 학습/평가

이 단계는 가장 중요한 ablation입니다.

비교 대상:
- 기존 L1 only
- L1 only + nGNN

분석 포인트:
- Fraud F1이 실제로 오르는가
- ROC-AUC / AP가 함께 개선되는가
- latency와 memory 비용이 어느 정도 증가하는가
- embedding variance나 class separation이 나아지는가

이 단계에서 gain이 없으면 L2/Fusion까지 확장할 필요가 줄어듭니다.

---

### Step 4. L1-nGNN embedding cache 도입

nGNN은 Level 1 비용이 크므로, L2 연결 전에 embedding cache를 먼저 안정화하는 것이 좋습니다.

권장 방식:
1. train/val/test 각각에 대해 L1-nGNN embedding 생성
2. disk cache 또는 memory-mapped artifact 저장
3. L2 학습 시 캐시된 embedding만 사용

장점:
- L2 실험 반복 비용 감소
- Fusion 디버깅 쉬움
- 8GB GPU 환경에서 실용적

---

### Step 5. L1-nGNN + L2 통합

이 단계에서는 L2 자체를 바꾸기보다,
입력으로 들어가는 Level 1 embedding만 nGNN 버전으로 교체합니다.

목표:
- relation-aware graph가 더 좋은 L1 표현을 받았을 때 성능 향상 측정
- L1 성능 향상이 L2에도 전달되는지 검증

주의:
- dimension mismatch 방지
- legacy/refactoring interface mismatch 방지
- cache schema mismatch 방지

---

### Step 6. Full Fusion 연결

마지막으로 Full Revision에 nGNN을 결합합니다.

구조:
- Level 1(nGNN)
- Level 2(existing)
- Fusion(existing)

핵심 질문:
- L1-nGNN의 gain이 Full pipeline에서도 유지되는가
- Fusion이 L1 향상분을 실제로 활용하는가
- L2/Fusion이 L1 gain을 희석하지 않는가

---

## 권장 config 파일 구성

예시:

- `configs/ngnn/base_ngnn.yaml`
- `configs/ngnn/level1_ngnn.yaml`
- `configs/ngnn/level1_ngnn_level2.yaml`
- `configs/ngnn/full_ngnn.yaml`

핵심 파라미터:
- `encoder_backend`
- `num_hops`
- `max_nodes_per_subgraph`
- `nested_readout`
- `root_policy`
- `cache_nested_subgraphs`
- `cache_level1_embeddings`
- `batch_size`
- `grad_accum_steps`
- `mixed_precision`

---

## 실행 엔트리포인트 권장안

기존 파이프라인과 일관성을 유지하는 것이 좋습니다.

예시:
- `src/gog_fraud/pipelines/train_level1.py`
- `src/gog_fraud/pipelines/train_level2.py`
- `src/gog_fraud/pipelines/run_fraud_benchmark.py`
- `src/gog_fraud/pipelines/run_ngnn_benchmark.py`

권장 추가 기능:
- backend 선택
- nGNN 캐시 warmup
- smoke mode
- ablation mode
- artifact export

---

## 8GB GPU 환경 고려사항

nGNN은 특히 메모리 사용량이 증가하기 쉬우므로 다음 원칙이 중요합니다.

### 1. 작은 nested depth부터 시작

권장 초기값:
- `num_hops = 1` 또는 `2`

### 2. batch size를 공격적으로 줄인다

nGNN은 graph 하나당 처리 비용이 커지므로,
기존 Level 1 batch size를 그대로 쓰면 OOM 위험이 큽니다.

### 3. gradient accumulation 사용

실효 batch 크기를 유지하면서 GPU 메모리 사용량을 통제할 수 있습니다.

### 4. mixed precision 사용

가능하다면 학습/추론 모두 mixed precision을 사용합니다.

### 5. extraction과 encoding을 분리

subgraph extraction은 미리 해두고,
학습 시에는 tensorized artifact만 로드하는 편이 효율적입니다.

---

## 체인별 실험 전략

체인별 구조 특성이 다르기 때문에 nGNN 효과도 다르게 나타날 수 있습니다.

권장 순서:
1. smoke set
2. 단일 chain 소규모 split
3. 단일 chain full run
4. multi-chain 비교

분석 포인트:
- dense chain vs sparse chain에서 nGNN 효과 차이
- fraud pattern의 local structural complexity 차이
- Level 1 nested representation의 일반화 성능

---

## ablation 실험 권장안

nGNN의 효과를 분리하려면 다음 ablation이 필요합니다.

### A. backend ablation
- standard GNN
- nGNN

### B. hop ablation
- 1-hop
- 2-hop
- 3-hop

### C. readout ablation
- mean
- max
- attention

### D. integration ablation
- L1 only
- L1+nGNN + L2
- Full Fusion

---

## 최종 실험 출력

각 실험은 최소한 다음 정보를 저장하는 것이 좋습니다.

- config snapshot
- checkpoint path
- backend type
- nested extraction policy
- F1
- ROC-AUC
- AP
- latency
- peak memory
- cache hit ratio
- chain name
- experiment notes

---

## 요약

nGNN 파이프라인은 다음 흐름으로 진행하는 것이 적절합니다.

1. Level 1 backend selectable 구조 정리
2. rooted subgraph extraction 구현
3. L1-nGNN 단독 성능 검증
4. Level 1 embedding cache 안정화
5. L1-nGNN + L2 통합
6. Full Fusion으로 최종 평가

핵심은 nGNN을 먼저 Level 1 개선 기술로 검증하고,
그 효과가 L2와 Fusion에서 어떻게 증폭되는지를 단계적으로 측정하는 것입니다.
