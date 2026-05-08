# Pipeline and Data Handling Architecture

이 문서는 기초 이상 거래 탐지 데이터 세트부터 모델들의 단계적 평가, 훈련 및 융합에 이르는 End-to-end 실험 파이프라인의 과정을 구조화합니다.

## `run_fraud_benchmark.py` (핵심 실행 파이프라인)
- **위치:** `src/gog_fraud/pipelines/run_fraud_benchmark.py`
- 모델 파라미터 최적화, 불확실성/사기 징후(MC-nGNN 등급) 비교 평가를 모두 조율합니다.
  
### 평가 및 실행 단계 (Evaluation Phases)
1. **Legacy**
   과거의 벤치마크 모형(`DOMINANT`, `DONE`, `GAE`)을 어댑터 패키지를 통해 레거시 검증 절차에 맞춰 동작시킵니다.
2. **Revision L1**
   레거시에 대응할 최신 `Level1Model` (인터페이스: `Level1Output`을 명확히 반환)를 이용해 Subgraph 단위만의 Fraud Score 및 개별 AUC/AP(매크로 정확도)를 도출하여, 관계(Relation)가 포함되지 않은 구조의 한계 및 성능 기준선을 보여줍니다.
3. **Revision L1+L2 (단독 모델)**
   `Level1Trainer`가 산출한 고도화 임베딩(representation)을 추출 및 고정(Freeze)하여, Level 2 구조망(`Level2Trainer`)의 노드 입력 Feature로 주입해 시간적 흐름/사기 연관 전파(Graph Attention Network 의존 관계)를 실험합니다. 
4. **Revision Full (Fusion)**
   모든 Level 1 점수(Score 및 Embedding)와 Level 2에서 확산/보정된 Context Embedding 값들을 최종 결합(Fusion Network)하거나 관절 학습시켜 도출된 최고 수준의 탐지 능력을 정량 평가합니다.

## `Level1GraphDataset` / `Level2GraphDataset`
- 이질적인 (개별 트랜잭션, 노드 메타 통계 / 묶음 Subgraph 연관도) 두 개의 그래프 유형을 혼동 없이 다루기 위한 분리.
- **Level 1 Edge:** 한 지갑의 송수신 내역 집합 (Transaction Network). 송신자, 상호작용 지표 등이 명시됨.
- **Level 2 Edge:** 서로 상관없는 (혹은 연관 가능성이 존재하는) 시간순 서브그래프들의 Time Edge, Temporal Continuity Meta-Edge입니다.

## 설정 관리 (YAML Configs)
- `configs/benchmark/` 디렉토리에 명시된 설정 파일들을 기반으로 모델 하이퍼파라미터(`lr`, `hid_dim`, `dropout` 등)가 조절되며 `_build_level1_model`, `_build_level1_trainer` 등의 팩토리 함수가 객체들을 적응할 수 있게 준비합니다. (이때의 인자와 도메인 언어가 통일되도록 인터페이스를 통합.)
