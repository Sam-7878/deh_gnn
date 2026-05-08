# DeH-GNN (Decoupled Hierarchical Graph Neural Network)

**DeH-GNN**은 GoatBank 생태계의 핵심 보안 및 자금 세탁 방지(AML) 인프라로 작동하는 계층형 그래프 신경망 기반 사기 탐지 엔진입니다. 

## 🚀 Overview
기존 상용 AML 솔루션의 한계를 극복하고 멀티 체인(Ethereum, BSC, Polygon) 상에서 발생하는 복잡하고 은닉된 자금 세탁 패턴을 실시간으로 탐지합니다. 단일 그래프 추론의 한계를 뛰어넘어 Level 1(개별 하위 그래프 특성)과 Level 2(하위 그래프 간의 메타 릴레이션)를 분리하여 학습하는 **계층적 아키텍처**를 채택했습니다.

## ✨ Core Features
- **Decoupled Architecture**: Subgraph 내부 구조(Level 1)와 관계 네트워크(Level 2)를 물리적/논리적으로 분리.
- **Uncertainty-Aware Alerting**: Monte Carlo (MC) Dropout을 활용하여 예측의 불확실성을 정량화하고, 오탐(False Positive)을 최소화.
- **Streaming & Static Processing**: 스트리밍 리플레이 시뮬레이션을 통한 실시간 탐지 및 정적(Static) 대규모 그래프 배치 처리 동시 지원.
- **Ablation & Benchmarking**: 설정(Config) 기반의 실험 파이프라인을 통해 Legacy, Level 1 단독, Level 1+2 조합 간의 ROC-AUC, PR-AUC, F1 성능을 비교.

## 📂 Repository Structure
- `src/` : 모델, 학습 파이프라인, 평가 코드가 포함된 코어 소스코드
- `configs/` : YAML 기반의 실험 세팅 파일 (MC, nGNN 등)
- `data/` : 학습 및 벤치마크 평가용 그래프 데이터 (캐시 및 전처리)
- `docs/` : 아키텍처 및 시스템 세부 설계 문서 (`docs/architecture/` 참조)
- `scripts/` & `tests/` : 자동화된 벤치마크 런 및 단위 테스트

## ⚙️ Getting Started
자세한 아키텍처와 구동 방법은 `docs/architecture/` 내부의 마크다운 문서들을 참조해 주세요.
