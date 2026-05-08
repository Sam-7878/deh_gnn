# nGNN Development Strategy Overview

이 문서는 현재 완료된 Legacy vs Revision 4-way benchmark pipeline 이후, 다음 단계인 nGNN(Nested GNN) 개발을 어떤 목적과 순서로 진행할지 상위 전략 관점에서 정리합니다.

---

## 현재 위치

현재 실험 파이프라인은 다음 4개 축으로 정리되어 있습니다.

- Legacy baseline: DOMINANT / DONE / GAE / AnomalyDAE / CoLA
- Revision L1: Level 1 only
- Revision L1+L2: Level 1 + Level 2
- Revision Full: Level 1 + Level 2 + Fusion

이 과정에서 확인된 핵심 사실은 다음과 같습니다.

- Graph-of-Graphs 구조에서 Level 1은 node feature / subgraph embedding의 품질을 결정합니다.
- Level 2는 inter-graph relation을 반영하는 보강 계층입니다.
- 실제 Fraud Detection 탐지 성능에는 Level 1의 표현력이 더 직접적이고 크게 작용합니다.

따라서 nGNN은 처음 계획처럼 단순히 구조 변경 자체를 목표로 하기보다, **Level 1 representation을 강화하는 핵심 기술**로 재정의하는 것이 적절합니다.

---

## 왜 nGNN인가

기존 GoG 파이프라인은 개별 서브그래프 또는 로컬 거래 구조를 하나의 그래프로 보고 임베딩한 뒤,
이를 상위 relation graph(Level 2)로 연결하는 방식입니다.

그러나 Fraud Detection에서는 다음과 같은 문제가 남습니다.

- 동일한 전체 통계량을 가진 subgraph라도 내부 구조가 다를 수 있음
- root node 기준의 국소 패턴이 anomaly 판단에 더 중요할 수 있음
- 단순 message passing만으로는 nesting된 구조 차이를 충분히 반영하지 못할 수 있음

nGNN의 목적은 이러한 한계를 보완하는 것입니다.

즉, nGNN은 다음을 가능하게 합니다.

- root-centered local topology를 더 정교하게 표현
- 같은 크기의 subgraph라도 내부 nested pattern을 구분
- structural role 차이를 embedding에 반영
- Level 1에서 더 정보량이 높은 representation 생성

---

## nGNN의 역할 재정의

현재 구조에서 nGNN의 1차 역할은 **Level 1 encoder 고도화**입니다.

### 1. Level 1 표현력 강화

기존 Level 1 encoder는 다음 정보를 충분히 잃을 수 있습니다.

- root node와 주변 노드의 역할 차이
- hop별 구조 차이
- local motif / cycle / star / fan-in / fan-out 패턴
- laundering / peeling과 유사한 계층적 전개 형태

nGNN은 nested subgraph 단위로 representation을 만들기 때문에,
이러한 패턴 차이를 더 잘 보존할 수 있습니다.

### 2. Level 2 입력 품질 향상

Level 2는 결국 Level 1 embedding을 입력으로 받습니다.
따라서 Level 1 representation이 개선되면, Level 2 성능도 간접적으로 개선될 가능성이 큽니다.

즉, nGNN의 직접 효과는 Level 1에, 간접 효과는 L1+L2 전체에 나타납니다.

### 3. Fusion 단계의 정보 밀도 증가

Fusion은 최종적으로 Level 1 / Level 2 출력을 결합합니다.
nGNN 기반 Level 1 embedding은 Fusion 단계에서 더 discriminative한 신호를 제공할 수 있습니다.

---

## 개발 우선순위

nGNN은 아래 순서로 붙이는 것이 적절합니다.

### Stage 0. 현재 deterministic 파이프라인 고정

선행 조건:
- Level 1 단독 학습/평가 안정화
- Level 2 단독 학습/평가 안정화
- Level 1 frozen + Level 2 학습 가능
- Fusion 포함 최종 평가 가능

### Stage 1. Level 1에 nGNN 적용

가장 먼저 수행할 단계는 **Level 1 encoder replacement**입니다.

핵심 목표:
- 기존 L1 encoder 대비 nested representation의 효과를 분리 측정
- Fraud Detection에 실제로 유의미한 gain이 있는지 확인
- L2 없이도 성능 상승 여부 파악

이 단계가 가장 중요합니다.

### Stage 2. Level 1(nGNN) + Level 2 결합

다음은 nGNN 기반 Level 1 embedding을 Level 2 relation graph에 입력합니다.

핵심 목표:
- improved Level 1 embedding이 L2에서 얼마나 증폭되는지 측정
- inter-graph relation과 nested local structure의 결합 효과 검증

### Stage 3. Full Fusion에 nGNN 적용

최종적으로 Level 1(nGNN) + Level 2 + Fusion 구조를 완성합니다.

핵심 목표:
- end-to-end Fraud Detection 성능 비교
- 기존 Full Revision 대비 개선 폭 측정
- 이후 MC와 결합 가능한 구조 확보

---

## nGNN 개발의 핵심 원칙

### 원칙 1. nGNN은 Level 1 우선이다

현재까지의 분석상 탐지율에 더 직접적인 영향을 주는 것은 Level 1입니다.
따라서 nGNN도 먼저 Level 1 encoder에 적용해 성능 기여를 분명히 해야 합니다.

금지 사항:
- L1 효과 검증 없이 바로 L2/Fusion까지 동시에 바꾸기
- 원인 분리가 불가능한 대규모 구조 변경

### 원칙 2. 기존 L1/L2/Fusion 구조를 유지한 채 확장한다

nGNN은 기존 파이프라인을 대체하는 것이 아니라,
우선은 **L1 encoder의 drop-in replacement 또는 selectable backend**로 설계해야 합니다.

즉 다음 구조를 유지합니다.

- Level 1 encoder: GNN / nGNN 선택 가능
- Level 2 encoder: 기존 relation-aware model 유지
- Fusion head: 기존 인터페이스 유지

### 원칙 3. nested subgraph 생성 비용을 통제해야 한다

nGNN은 성능 향상 가능성이 높지만 계산량과 메모리 부담이 큽니다.
특히 8GB GPU 환경에서는 다음이 중요합니다.

- nested subgraph 사전 캐싱
- root-centered extraction 범위 제한
- hop 수와 nested depth 제한
- batch 크기 동적 조정
- Level 1 embedding offline cache 활용

### 원칙 4. MC보다 먼저 또는 동시에 엮기보다 독립적으로 검증한다

MC와 nGNN은 둘 다 성능 향상을 줄 수 있지만, 성격이 다릅니다.

- MC: uncertainty / approximation / robustness
- nGNN: representation power / nested structure encoding

따라서 nGNN은 먼저 독립적으로 효과를 검증한 뒤,
이후 MC와 결합하는 것이 가장 해석 가능성이 높습니다.

---

## 기대 효과

### 1. Local structural fraud pattern 포착력 향상

사기 거래는 단일 edge보다 local transaction neighborhood 안에서 패턴이 드러나는 경우가 많습니다.
nGNN은 root 기준 nested substructure를 더 세밀하게 반영할 수 있습니다.

기대 효과:
- 재현율 상승
- 구조적으로 미세한 fraud pattern 분리
- Level 1 anomaly score 품질 향상

### 2. Relation graph 입력 품질 개선

Level 2는 L1 embedding 품질에 의존합니다.
nGNN 기반 L1 embedding은 relation propagation 이전에 더 정교한 초기 표현을 제공합니다.

기대 효과:
- relation-aware classification 향상
- Fusion의 정보 밀도 증가

### 3. 향후 MC와의 결합 기반 확보

nGNN은 표현력 향상 축이고, MC는 불확실성/강건성 축입니다.
nGNN이 안정화되면 이후 다음 결합이 가능해집니다.

- nGNN + Model MC
- nGNN + Data MC
- nGNN + MC + Fusion

---

## 요약

현재 단계에서 nGNN은 다음과 같이 정의할 수 있습니다.

- 목적: Level 1 nested structural representation 강화
- 위치: 우선 Level 1 encoder replacement
- 우선순위: Level 1 nGNN -> L1+nGNN + L2 -> Full Fusion
- 전제 조건: 현재 L1/L2/Fusion deterministic 파이프라인 안정화
- 기대 효과: Fraud Detection F1, ROC-AUC, AP 개선 및 구조적 패턴 분별력 향상
