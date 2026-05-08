# MC-nGNN for GoG Project Skeleton

이 문서는 "MC-nGNN for GoG" 프로젝트의 최상위 패키지 트리와 핵심 아키텍처 원칙을 설명하는 기본 골격 문서입니다. 세부 모듈의 구조는 서브 문서(`docs/architecture/*.md`)를 참조하십시오.

## 원칙 및 특징 (Hierarchical Fraud Stack)
1. **Level 1과 Level 2의 물리적/논리적 분리**: 개별 Subgraph의 내부 특성(Level 1)과 Subgraph 간 관계 네트워크(Level 2)를 별도 모듈로 분리하여 각 계층의 책임을 강화.
2. **인터페이스 표준화**: 모든 모델 추론의 결과는 공통 데이터 클래스(`Level1Output`, `Level2Output`)로 통일하여 구성 요소 간 데이터 전송의 유연성과 디버깅을 개선.
3. **학습 루프 분리**: 각 계층(Target)의 훈련 파이프라인(Trainer)을 구조적으로 독립시키고 Fusion을 통해 통합 성능 관리.
4. **구성 기반 실험**: YAML 등 정형화된 구성을 통해서 동적인 모델 파이프라인 생성과 비교 실험 통제.

## 주요 패키지 트리 (`src/gog_fraud/`)

```text
src/gog_fraud/
 ├── common/             # 공용 데이터 구조 정의 계층 (types.py, Level1Output, Level2Output 등)
 ├── data/               # 데이터 파이프라인, 로딩
 │    ├── level1/        # Level 1 개별 그래프/입력 특성 데이터셋 생성 (내부 엣지 등 포함)
 │    └── level2/        # Level 2 간 릴레이션 메타-그래프 및 Temporal/Campaign 연결 제공
 ├── models/             # 핵심 모델 정의 아키텍처
 │    ├── level1/        # Subgraph 내부 구조 및 노드 특성 인코더 (Level1Model)
 │    ├── level2/        # Subgraph 임베딩 입력을 받아 관계 맵에서 상호작용 학습 (Level2Model)
 │    └── fusion/        # 각 스코어를 결합/응용하는 Fusion Network 관리기
 ├── training/           # 모델별 Trainer 파이프라인 및 루프
 │    └── loops/         # Level1Trainer, Level2Trainer 훈련의 로직 제어 핵심
 ├── pipelines/          # End-to-end 모델 구동 스크립트 모음 (run_fraud_benchmark.py 등)
 ├── evaluation/         # F1, ROC-AUC 및 불확실성/사기탐지 매트릭 지원
 └── adapters/           # 과거 벤치마크 혹은 Legacy 모형의 테스트 런 호환 계층
```

## 서브 문서 (Architecture Details)
아래의 서브 문서를 참조하여 개별 소스의 구조와 인터페이스 정의를 파악할 수 있습니다.
- [Level 1, 2 Plan for Decoupling and Architectural Refactoring (docs/architecture/level1_and_2/architectural_plan.md)](docs/architecture/level1_and_2/architectural_plan.md)
- [Level 1 Architecture (docs/architecture/level1_and_2/level1_architecture.md)](docs/architecture/level1_and_2/level1_architecture.md)
- [Level 2 & Fusion Architecture (docs/architecture/level1_and_2/level2_fusion_architecture.md)](docs/architecture/level1_and_2/level2_fusion_architecture.md)
- [Data & Pipeline Architecture (docs/architecture/level1_and_2/pipeline_and_data.md)](docs/architecture/level1_and_2/pipeline_and_data.md)
