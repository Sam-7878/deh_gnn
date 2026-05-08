# Level 1 Architecture (Sub-graph Representation)

Level 1은 개별 거래(서브그래프) 내부 구조의 이상 징후를 개별적으로 판별하는 아키텍처 코어입니다. 블록체인 상거래 집합 하나가 주어졌을 때 내부에서 발생한 자금의 흐름 특징을 잡아냅니다.

## 주요 역할과 위치
- **위치:** `src/gog_fraud/models/level1/model.py` 및 관련 패키지
- **입력:** 개별 Transaction Subgraph (단일 계약이나 지갑을 중심으로 한 내부 네트워크 구조)
- **출력:** 공용 구조형 속성 데이터 `Level1Output` (Score, Node Embedding, Graph Embedding 포함)
  
## 핵심 모델 (`Level1Model`)
Phase-1 `Level1Model`은 내부 노드들이 구성한 위상 정보(Topology)와 메타 정보를 처리합니다.

- **`Level1GNNEncoder` (GNN Base):** 
  여러 개의 `GINConv` 등 GNN 레이어들을 적층하여 노드 단위의 내재적 특성을 Embedding으로 변환. (Fan-in, Fan-out, 반복 패턴, 구조적 Hub 집중 현상을 포착)
- **`GraphReadout`:** 
  그래프 내부 노드들의 Embedding을 종합(`global_mean_pool`, `global_max_pool` 등)하여 하나의 Subgraph-level Vector(Representation)로 요약합니다.
- **`Level1FraudHead`:** 
  도출된 그래프 임베딩을 바탕으로 사기(`Fraud`) 여부에 대한 Score (Logit 및 확률형 예측값)를 계산합니다. 이 스코어가 초기 모델링의 기초 판단 및 Fusion Layer에서 평가 기준으로 적용됩니다.

## 인터페이스 통일점 (`Level1Output` Data Class)
기존 Legacy 구현체에서 직접적으로 `logits`만을 반환하던 코드를 개선하여, 이후 Level 2 및 파이프라인으로 전송될 메타정보를 함께 전달합니다.
- `graph_id`: 처리 중인 서브그래프의 고유 식별자.
- `embedding`: 생성된 전체 요약 데이터 특성(그래프 수준). Level 2의 직접적인 Feature로 전달됨.
- `logits`, `score`: 이상 거래 판단의 기초값.
- `label`: 지도 학습 처리에 필요한 Ground Truth.
- 모델들은 명확하게 위 Data Class 타입을 반환하며, 훈련 루프(`Level1Trainer`)는 이를 통해 안전한 파이프라인 학습과 오차 계산을 수행합니다.
