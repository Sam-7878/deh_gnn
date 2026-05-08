# nGNN Precompute Strategy Overview

이 문서는 nGNN(Nested GNN) 도입을 위해 필요한 rooted subgraph precompute 전략을 상위 관점에서 정리합니다.

---

## 왜 precompute가 필요한가

nGNN은 일반적인 Level 1 encoder보다 입력 준비 비용이 큽니다.
각 contract 또는 subgraph에 대해 추가적으로 다음 작업이 필요합니다.

- root 선정
- k-hop rooted subgraph 추출
- 노드 인덱스 remapping
- root indicator / hop metadata 생성
- 제한된 크기(max nodes per subgraph) 안으로 구조 정리
- nested input tensor화

이 작업을 DataLoader 내부에서 실시간으로 수행하면 다음 문제가 발생할 수 있습니다.

- epoch 시간 급증
- DataLoader worker 병목
- batch별 처리 시간 편차 확대
- 메모리 사용량 변동성 증가
- 재현성과 디버깅 난이도 상승

특히 현재 8GB GPU 환경에서는 학습 단계보다 입력 생성 단계에서 전체 파이프라인이 불안정해질 가능성이 큽니다.

따라서 초기 nGNN 개발 단계에서는 rooted subgraph를 **사전 계산(precompute)** 하고, 이를 디스크에 저장한 뒤 학습 시 재사용하는 전략이 적절합니다.

---

## precompute의 목표

nGNN precompute 파이프라인의 목표는 단순 캐시가 아닙니다.
다음 4가지 목적을 동시에 만족해야 합니다.

### 1. 재현성 확보

동일한 원본 graph와 동일한 config에 대해 항상 동일한 rooted subgraph를 생성해야 합니다.

### 2. 학습 안정성 확보

학습 시점에는 extraction이 아닌 loading과 batching에 집중하게 함으로써,
GPU 메모리와 epoch 시간을 안정적으로 통제합니다.

### 3. L1/L2/Fusion 반복 실험 지원

rooted subgraph를 한 번 생성해 두면,
동일한 nGNN 입력을 여러 L1 실험과 이후 L2/Fusion 실험에서 반복 사용할 수 있습니다.

### 4. 향후 확장성 유지

현재는 deterministic precompute가 기본이지만,
향후 online transform 또는 MC 기반 stochastic augmentation으로 확장할 수 있도록 구조를 열어 둡니다.

---

## 현재 단계에서의 권장 방향

현재 프로젝트의 우선순위는 다음과 같습니다.

1. Legacy / Revision benchmark 안정화
2. nGNN이 Level 1 개선에 실제로 기여하는지 검증
3. nGNN 기반 L1 embedding을 L2/Fusion에 연결
4. 이후 MC와의 결합 검토

이 순서에서는 rooted subgraph를 precompute하는 편이 더 적합합니다.
왜냐하면 현재 필요한 것은 입력 다양성보다 **비교 가능한 안정 실험**이기 때문입니다.

---

## precompute 파이프라인의 역할 범위

precompute 파이프라인은 다음 범위를 담당합니다.

- 원본 L1 입력 graph 로딩
- root 정책 결정
- k-hop rooted subgraph 추출
- max node budget 적용
- root / hop / node mapping metadata 생성
- 디스크 저장
- split 단위(train/val/test) 인덱싱
- 이후 학습용 dataset에서 로드 가능하도록 포맷 제공

반면 다음은 precompute 파이프라인의 직접 책임이 아닙니다.

- nGNN encoder 학습
- fraud head 학습
- Level 2 relation graph 학습
- Fusion scoring

즉 precompute는 **모델 훈련 이전 단계의 데이터 준비 계층**입니다.

---

## 기본 설계 원칙

### 원칙 1. 모델과 전처리를 분리한다

nGNN encoder 구현은 `src/gog_fraud/models/extensions/ngnn/` 아래에 두고,
precompute 스크립트 및 데이터 직렬화 로직은 데이터/파이프라인 계층에 둡니다.

### 원칙 2. deterministic extraction을 기본값으로 한다

같은 입력에 대해 결과가 달라지면 benchmark 비교가 어려워집니다.
초기 버전은 deterministic한 rooted subgraph 추출을 기본으로 합니다.

### 원칙 3. online 경로를 완전히 버리지는 않는다

향후 다음 확장을 위해 backend 선택지를 유지합니다.

- online transform
- data-level MC
- edge dropout after load
- adaptive local sampling

### 원칙 4. 8GB GPU 기준으로 설계한다

초기 설정은 보수적으로 가져갑니다.

- 작은 `num_hops`
- 작은 `max_nodes_per_subgraph`
- 작은 batch size
- mixed precision
- Level 1 embedding cache 병행

---

## precompute 이후 기대 효과

### 1. L1 실험의 안정화

nGNN 기반 Level 1 실험을 반복 가능하게 만듭니다.

### 2. L2 연결 용이성

L1-nGNN embedding 생성 이후 L2 학습 시 동일 입력 재사용이 가능해집니다.

### 3. 디버깅 비용 절감

추출 로직과 학습 로직을 분리하면,
성능 저하 원인이 extraction인지 model인지 더 쉽게 구분됩니다.

### 4. 향후 캐시 계층 확장 가능

다음과 같은 다층 캐시 전략으로 이어질 수 있습니다.

- rooted subgraph cache
- tensorized feature cache
- Level 1 embedding cache
- Level 2 relation artifact cache

---

## 추천 산출물

nGNN precompute 파이프라인은 최소한 다음 산출물을 만들어야 합니다.

- split별 rooted subgraph artifact
- extraction metadata
- dataset index manifest
- config snapshot
- extraction statistics report

---

## 요약

현재 단계에서 nGNN precompute 전략은 다음과 같이 정의할 수 있습니다.

- 목적: nGNN 입력 생성을 학습 이전에 안정적으로 고정
- 기본값: deterministic precompute + disk cache
- 이유: 8GB 환경, 재현성, 디버깅, benchmark 비교 가능성
- 역할: rooted subgraph 생성/저장/인덱싱
- 확장성: 향후 online transform 및 MC와 결합 가능한 구조 유지
