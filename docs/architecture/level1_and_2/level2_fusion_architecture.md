# Level 2 & Fusion Architecture (Meta-graph Relation)

이 문서는 Level 1 인코더에 의해 생성된 개별 Subgraph 임베딩을 노드로 삼아, 여러 Subgraph 간의 연관 관계(시간순, 연관 계좌 등)를 처리하여 상위 계층에서의 의존성을 학습하는 과정으로 설계된 Level 2 아키텍처와 Fusion 로직을 설명합니다.

## `Level2Model` (Graph-of-Graph 구조)
- **위치:** `src/gog_fraud/models/level2/model.py`
- **입력:** Level 1 모델로부터 추출된 각 Subgraph 차원의 Embedding (`Level1Output` 추출 정보 참조) + 시간/동작의 흐름을 지닌 Temporal / Meta-edge 정보 리스트.
- **출력:** 공용 구조형 속성 데이터 `Level2Output` (Contextual Embedding 및 Score).
  
### 구조 요약 (Level2GATEncoder)
- Graph Attention Network 버전 2 (`GATv2Conv`) 엔진을 기반으로, 단순한 Topology를 넘어 어떤 이웃(특정 Subgraph)이 사기 패턴 형성에 가장 큰 영향을 주었는지 동적 연관도 분석(Attention Strength)을 진행합니다.
- 동일한 지갑 혹은 시간적으로 연결된 Subgraph를 연속적인 세탁 과정(Laundering Path / Campaign 흐름)으로 인식합니다.
- `Level2GraphReadout`를 통해 메시지 패싱이 한 번 더 요약되며, 이로써 하나의 노드 스코어가 아닌 복합적 맥락이 반영된 최종 `Level2Output` 평가값이 도출됩니다. 

## Fusion Network 구성
- 개별(Level 1) 평가값과 관계형(Level 2) 맥락 평가값을 결합하여 오탐률(False Positive)을 낮추기 위한 추론기입니다.
- **위치:** `src/gog_fraud/pipelines/fusion.py` 등.
- **주요 융합 모드**:
  1. **Score Fusion:** 양 모델에서 각각 도출해 낸 Anomaly Score / Logit 값을 가중 합(Weighted Sum)하거나 MLP에 통과시켜 결합합니다. (빠른 적용)
  2. **Embedding Fusion (권장):** Level 1 Embedding과 Level 2 Embedding을 Concatenate 하여 최종 Fully Connected 신경망 계층으로 연산합니다.
  3. **Joint Fine-tuning (심화):** 사전 분리된 두 모델 구조를 하나로 엮어 처음부터 끝까지 역전파를 통해 동시 학습합니다.
