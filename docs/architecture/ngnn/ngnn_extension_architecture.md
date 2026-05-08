# nGNN Extension Architecture

이 문서는 기존 Fraud Detection 코어 구조(Level 1 / Level 2 / Fusion)를 유지한 채,
nGNN(Nested GNN)을 어떤 방식으로 확장 모듈로 설계할지 설명합니다.

---

## 설계 방향

nGNN은 현재 리팩토링 구조상 코어 모델 전체를 뒤엎는 방식이 아니라,
우선 **Level 1 encoder의 확장 구현체**로 들어가는 것이 적절합니다.

권장 위치:

- `src/gog_fraud/models/level1/`
- `src/gog_fraud/models/extensions/ngnn/`
- `src/gog_fraud/data/transforms/`
- `src/gog_fraud/pipelines/`

예시 구조:

- `src/gog_fraud/models/extensions/ngnn/interfaces.py`
- `src/gog_fraud/models/extensions/ngnn/subgraph_extractor.py`
- `src/gog_fraud/models/extensions/ngnn/nested_encoder.py`
- `src/gog_fraud/models/extensions/ngnn/readout.py`
- `src/gog_fraud/models/extensions/ngnn/cache.py`

---

## nGNN의 핵심 구성 요소

nGNN은 단순히 GNN layer 몇 개를 깊게 쌓는 것이 아니라,
**nested subgraph를 추출하고 그것을 다시 encoding하는 구조**로 보는 것이 적절합니다.

핵심 구성은 다음과 같습니다.

### 1. Rooted Subgraph Extractor

역할:
- 각 중심 노드 또는 대상 subgraph에 대해 root-centered neighborhood를 추출
- hop 기반 또는 policy 기반 nested subgraph 생성
- 학습/평가 시 재현 가능한 입력 단위 제공

주요 파라미터:
- `num_hops`
- `max_nodes_per_subgraph`
- `sampling_policy`
- `include_edge_attr`
- `root_policy`

출력:
- rooted subgraph list
- root node index mapping
- optional hop partition metadata

---

### 2. Nested Encoder

역할:
- 추출된 rooted subgraph를 encode
- root node와 context node를 구분해 representation 생성
- hop별 또는 subgraph별 구조 차이를 embedding에 반영

가능한 설계:
- root-aware message passing
- hop-wise encoding
- subgraph-level pooling
- shared encoder across nested subgraphs

입력:
- rooted subgraph
- node features
- edge index / edge attributes
- optional positional or structural encoding

출력:
- subgraph embedding
- root-aware embedding
- optional intermediate states

---

### 3. Nested Readout

역할:
- 여러 nested subgraph representation을 하나의 Level 1 embedding으로 집계
- root-centered local evidence를 fraud score에 유용한 표현으로 변환

대표 방식:
- mean pooling
- attention pooling
- hop-aware weighted pooling
- root-context fusion

권장:
- 초기 구현은 단순 pooling
- 이후 attention 기반 readout으로 확장

---

### 4. Level 1 Adapter

역할:
- 기존 L1 pipeline 인터페이스와 nGNN 출력 인터페이스를 맞춤
- 기존 Level 2 / Fusion이 수정 없이 입력 받을 수 있도록 정합성 유지

중요:
- 기존 `Level1Output`과 가능한 한 동일하거나 호환되는 결과 구조를 유지
- 실험 모드에서 `encoder_backend = gnn | ngnn` 선택 가능

---

## 권장 인터페이스

### RootedSubgraphExtractor

```python
class RootedSubgraphExtractor:
    def extract(self, graph, root_index=None):
        raise NotImplementedError
```
