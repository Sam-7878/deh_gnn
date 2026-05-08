# nGNN Metrics and Reporting

이 문서는 nGNN(Nested GNN) 적용 이후 어떤 지표를 중심으로 성능을 측정하고,
어떤 방식으로 결과를 해석해야 하는지 정리합니다.

---

## 핵심 평가 지표

nGNN 도입 이후에도 Fraud Detection 관점의 기본 지표는 그대로 유지합니다.

### 1. F1-score

의미:
- Precision과 Recall의 균형
- class imbalance 환경에서 fraud 탐지 품질을 요약

nGNN 관점에서 기대 효과:
- nested local structure를 더 잘 반영하여 fraud recall 향상
- false positive를 크게 늘리지 않으면서 minor class 성능 개선

특히 강조 포인트:
- L1 only에서 F1이 개선되는가
- Full pipeline에서도 gain이 유지되는가

---

### 2. ROC-AUC

의미:
- score ranking 품질
- threshold와 무관한 분별력

nGNN 관점에서 기대 효과:
- 정상/이상 샘플의 score 분리도 향상
- local structural anomaly가 더 선명하게 드러남

---

### 3. AP (Average Precision)

의미:
- precision-recall 곡선 기반 성능
- positive 비율이 낮은 fraud detection에서 중요

nGNN 관점에서 기대 효과:
- hard positive를 더 위쪽 순위로 끌어올림
- 희소 fraud 사례에 대한 탐지 품질 개선

---

## nGNN 전용 보고 지표

nGNN은 단순 성능 외에도 구조적/시스템적 지표를 함께 봐야 합니다.

### 1. Latency

의미:
- rooted subgraph extraction + encoding + prediction 전체 소요 시간

필수 이유:
- nGNN은 일반 GNN보다 처리 비용이 크기 때문
- 실험 성능 향상만 보고 실제 운용 가능성을 놓치면 안 됨

---

### 2. Peak Memory

의미:
- 학습 및 추론 시 최대 GPU 메모리 사용량

필수 이유:
- 현재 환경이 8GB GPU이므로 실험 가능성 자체를 좌우함
- 동일 성능 향상이라도 메모리 비용이 지나치면 채택이 어려움

---

### 3. Nested Extraction Statistics

예시:
- 평균 rooted subgraph 수
- 평균 subgraph node 수
- hop별 node 분포
- extraction cache hit ratio

활용:
- 성능 향상의 원인이 단순 데이터 양 증가인지,
  실제 nested encoding 효과인지 해석 가능
- 병목 지점 파악 가능

---

### 4. Level 1 Embedding Quality Proxy

가능한 보조 지표:
- class centroid distance
- intra-class variance
- inter-class separation
- t-SNE / UMAP 시각화용 embedding dump

활용:
- nGNN이 실제로 Level 1 representation을 더 분리 가능하게 만드는지 점검

---

## 권장 리포트 표

### 공통 성능 표

| Experiment | L1 Backend | L2 | Fusion | F1 | ROC-AUC | AP | Latency | Peak Memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline L1 | GNN | ✗ | ✗ | - | - | - | - | - |
| nGNN L1 | nGNN | ✗ | ✗ | - | - | - | - | - |
| nGNN L1 + L2 | nGNN | ✓ | ✗ | - | - | - | - | - |
| Full nGNN | nGNN | ✓ | ✓ | - | - | - | - | - |

---

### nested 구조 통계 표

| Experiment | num_hops | Avg Subgraphs | Avg Nodes/Subgraph | Cache Hit | Notes |
|---|---:|---:|---:|---:|---|
| nGNN L1 | 1 | - | - | - | |
| nGNN L1 | 2 | - | - | - | |
| nGNN L1 | 3 | - | - | - | |

---

### ablation 리포트 표

| Ablation | Setting | F1 | ROC-AUC | AP | Latency | Comment |
|---|---|---:|---:|---:|---:|---|
| Backend | GNN | - | - | - | - | |
| Backend | nGNN | - | - | - | - | |
| Hops | 1-hop | - | - | - | - | |
| Hops | 2-hop | - | - | - | - | |
| Readout | mean | - | - | - | - | |
| Readout | attention | - | - | - | - | |

---

## 성능 해석의 핵심 질문

nGNN 결과는 단순히 “오르냐 내리냐”보다 다음 질문으로 해석해야 합니다.

### 1. nGNN이 Level 1 단독 성능을 개선하는가

이 질문이 가장 중요합니다.
현재 구조 분석상 Level 1이 탐지율에 더 큰 영향을 주기 때문입니다.

### 2. L1 개선이 L2와 Fusion에도 전달되는가

만약 L1 only에서는 개선되는데 Full에서는 개선되지 않는다면,
문제는 nGNN이 아니라 L2 또는 Fusion 인터페이스에 있을 수 있습니다.

### 3. 성능 향상이 계산 비용 증가를 정당화하는가

nGNN은 비용이 큰 구조이므로,
성능 gain과 latency/memory cost를 함께 판단해야 합니다.

---

## 진단 포인트

nGNN 개발 과정에서 자주 발생할 수 있는 문제는 다음과 같습니다.

### 1. 성능이 오르지 않는 경우

가능한 원인:
- nested extraction policy가 너무 단순함
- root 선정 규칙이 task와 맞지 않음
- hop 수가 너무 작거나 큼
- readout이 정보를 과도하게 평균화함
- 실제 fraud signal이 local nested 구조보다 다른 feature에 더 있음

점검 항목:
- L1 only 기준선과 직접 비교
- hop별 ablation
- root-aware representation 유무 비교
- embedding 시각화

---

### 2. 메모리 사용량이 급증하는 경우

가능한 원인:
- subgraph 수 과다
- nested artifact를 GPU에 장시간 유지
- batch size 과대
- extraction 결과 미캐시

점검 항목:
- 평균 subgraph 수
- 평균 node 수
- CPU/GPU 이동 시점
- intermediate tensor 해제 여부

---

### 3. L2/Fusion 통합 후 성능이 떨어지는 경우

가능한 원인:
- L1 embedding dimension mismatch
- embedding scale 변화
- cache schema mismatch
- Fusion이 새 표현을 제대로 활용하지 못함

점검 항목:
- L1 출력 분포 확인
- L2 입력 normalization 점검
- Fusion input key / shape 검증
- 기존 deterministic path와 비교

---

## 논문/보고서 기여 포인트

nGNN 결과는 다음 메시지로 연결될 수 있어야 합니다.

### 1. Hierarchical structural representation 강화

- root-centered nested subgraph를 통해 local fraud pattern을 더 잘 포착했다.
- 기존 flat GNN 대비 더 discriminative한 Level 1 embedding을 얻었다.

### 2. GoG 파이프라인과의 자연스러운 결합

- nGNN을 Level 1 encoder replacement로 도입하여,
  Level 2 및 Fusion 구조를 유지한 채 성능을 개선했다.

### 3. 실용적 리팩토링 기반 확장

- legacy 구조를 직접 깨지 않고 src 기반 modular architecture 위에서 nGNN을 통합했다.
- caching, batching, memory optimization을 통해 제한된 GPU 환경에서도 실험 가능하게 만들었다.

---

## 로그 및 저장 권장 항목

각 실험 run마다 다음을 저장하는 것이 좋습니다.

- seed
- config yaml snapshot
- backend type
- rooted extraction policy
- num_hops
- nested readout type
- dataset split 정보
- chain 정보
- F1 / ROC-AUC / AP
- latency
- peak memory
- average nested graph stats
- cache hit ratio
- sample embedding dump path
- error flag / notes

---

## 요약

nGNN 평가는 단순한 분류 성능 비교만으로는 충분하지 않습니다.

중점 포인트:
- F1, ROC-AUC, AP
- Level 1 representation 개선 여부
- L2 / Fusion으로의 gain 전달 여부
- latency / peak memory
- nested extraction 비용과 cache 효율
- 체인 간 일반화 가능성

즉, nGNN은 단순한 새 모델이 아니라
**Level 1 구조 표현력을 강화하여 전체 GoG Fraud Detection 파이프라인의 성능 상한을 높이는 핵심 확장 계층**으로 보고 평가해야 합니다.
