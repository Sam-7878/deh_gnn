# nGNN Precompute Dataflow Architecture

이 문서는 nGNN rooted subgraph precompute 파이프라인의 데이터 흐름과 구성 요소를 설명합니다.

---

## 상위 데이터 흐름

nGNN precompute 파이프라인은 다음 순서로 동작합니다.

1. 원본 Level 1 graph 로딩
2. root 선정
3. k-hop rooted subgraph 추출
4. 노드/엣지 remapping
5. root indicator 및 hop metadata 생성
6. max node budget 적용
7. artifact 직렬화
8. split index 및 manifest 생성
9. 학습용 dataset에서 재로딩

---

## 입력과 출력

### 입력

원본 입력은 Level 1 학습에 사용되는 graph 단위 샘플입니다.

예시:
- contract-centered transaction graph
- subgraph-level PyG `Data`
- node features / edge index / optional edge attributes
- sample id / contract id / label

### 출력

precompute 결과는 학습 시 즉시 로드 가능한 rooted subgraph artifact입니다.

권장 포함 항목:
- `x`
- `edge_index`
- `edge_attr` (optional)
- `root_node_idx`
- `root_indicator`
- `hop_ids`
- `original_node_ids`
- `sample_id`
- `contract_id`
- `label`
- `meta`

---

## 권장 모듈 구성

다음처럼 계층을 나누는 것이 바람직합니다.

### 1. Graph Source Layer

원본 L1 graph를 읽어오는 계층입니다.

역할:
- split별 원본 graph 로딩
- graph metadata 보존
- 전처리 입력 검증

예시 위치:
- `src/gog_fraud/data/datasets/`
- `src/gog_fraud/data/io/`

---

### 2. Rooted Subgraph Extraction Layer

root와 hop 조건에 따라 rooted subgraph를 생성합니다.

역할:
- root 선정
- `k_hop_subgraph` 또는 동등 로직 수행
- root remap
- 노드 수 제한
- disconnected fragment 방지 정책 적용

예시 위치:
- `src/gog_fraud/data/preprocessing/ngnn/subgraph_extraction.py`

---

### 3. Serialization Layer

추출 결과를 디스크에 저장 가능한 형태로 직렬화합니다.

역할:
- `.pt` artifact 저장
- split manifest 저장
- config fingerprint 기록
- 버전 정보 관리

예시 위치:
- `src/gog_fraud/data/preprocessing/ngnn/serialization.py`

---

### 4. Runtime Dataset Layer

학습 시점에 precomputed artifact를 로딩합니다.

역할:
- split manifest 기반 샘플 접근
- lazy loading 또는 memory-mapped loading
- batch collation 지원
- optional post-load augmentation 지원

예시 위치:
- `src/gog_fraud/data/datasets/ngnn_precomputed_dataset.py`

---

## rooted subgraph 내부 구조

각 rooted subgraph artifact는 단순히 작은 graph만 저장하는 것이 아니라,
nGNN encoder가 필요로 하는 구조 정보를 포함해야 합니다.

### 필수 필드

#### sample identity
- `sample_id`
- `contract_id`
- `label`
- `split`

#### graph structure
- `x`
- `edge_index`
- `edge_attr` (optional)

#### root metadata
- `root_node_idx`
- `root_indicator`

#### structural metadata
- `hop_ids`
- `num_nodes_original`
- `num_nodes_subgraph`
- `original_node_ids`

#### extraction metadata
- `num_hops`
- `max_nodes_per_subgraph`
- `root_policy`
- `extract_version`

---

## root 정책

precompute 파이프라인에서는 root 선정 정책을 명시적으로 관리해야 합니다.

가능한 정책:
- central contract node
- designated target address node
- transaction anchor node
- pre-labeled root node
- heuristic-selected root

중요:
- root 정책이 달라지면 rooted subgraph 의미가 달라지므로,
  metadata에 반드시 포함되어야 합니다.

---

## max node budget 적용 방식

8GB GPU 환경에서는 rooted subgraph가 무한정 커질 수 없습니다.
따라서 `max_nodes_per_subgraph` 정책이 중요합니다.

적용 방식 예시:
- root 우선 유지
- 가까운 hop 우선 유지
- 동일 hop 내 degree 기반 trimming
- temporal priority 기반 trimming
- deterministic node ordering 후 상위 N개 유지

권장:
- 초기 구현은 가장 단순하고 재현성 높은 방식 사용
- 예: root 및 낮은 hop 우선 + deterministic ordering

---

## 저장 단위 설계

저장 단위는 크게 두 가지가 가능합니다.

### A. sample-per-file

각 sample마다 rooted subgraph 하나를 `.pt` 파일로 저장

장점:
- 구현 단순
- 디버깅 쉬움
- 개별 샘플 재생성 용이

단점:
- 파일 수가 많아짐
- 파일 시스템 overhead 증가

---

### B. shard-per-file

여러 sample을 shard 단위로 묶어 저장

장점:
- I/O 효율 좋음
- 파일 수 감소
- 대규모 학습에 유리

단점:
- 구현 복잡
- 특정 샘플 디버깅 불편

권장:
- 초기 버전은 sample-per-file 또는 작은 shard
- 데이터 규모가 커지면 shard 기반으로 전환

---

## manifest 설계

precompute 결과를 로딩하기 위해 split별 manifest가 필요합니다.

권장 필드:
- `sample_id`
- `contract_id`
- `label`
- `artifact_path`
- `num_nodes`
- `num_edges`
- `root_node_idx`
- `split`
- `config_hash`

manifest의 역할:
- 빠른 인덱싱
- 데이터 무결성 확인
- split 재현성 유지
- 통계 리포트 생성

---

## online 확장과의 연결

precompute가 기본이더라도, 나중에 다음과 연결될 수 있도록 인터페이스를 맞추는 것이 좋습니다.

- online rooted subgraph extraction
- post-load edge dropout
- stochastic node masking
- adaptive neighborhood perturbation

즉 런타임 dataset layer는 다음 둘 다 받을 수 있도록 설계하는 것이 좋습니다.

- precomputed artifact
- online transformed artifact

---

## 요약

nGNN precompute dataflow architecture의 핵심은 다음과 같습니다.

- 원본 graph와 rooted subgraph 생성 로직을 분리한다
- extraction과 serialization을 독립 계층으로 둔다
- metadata를 풍부하게 저장한다
- split manifest를 중심으로 로딩 체계를 만든다
- 초기 버전은 단순하고 재현성 높은 방식으로 설계한다
