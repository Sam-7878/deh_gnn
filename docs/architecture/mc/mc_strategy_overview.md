# MC Development Strategy Overview

이 문서는 Legacy vs Revision 4-way benchmark pipeline이 완료된 이후, 다음 단계인 Monte Carlo(MC) 개발을 어떤 목적과 순서로 진행할지에 대한 상위 전략을 정리합니다.

---

## 왜 지금 MC인가

현재 실험 파이프라인은 다음 4개 축으로 정리되었습니다.

- Legacy baseline: DOMINANT / DONE / GAE / AnomalyDAE / CoLA
- Revision L1: Level 1 only
- Revision L1+L2: Level 1 + Level 2
- Revision Full: Level 1 + Level 2 + Fusion

이제 코어 구조(Level 1, Level 2, Fusion)의 역할이 분명해졌기 때문에, MC는 더 이상 구조를 대신하는 핵심이 아니라 **완성된 Fraud Detection 파이프라인 위에 부착되는 확장 기능**으로 보는 것이 적절합니다.

핵심 판단은 다음과 같습니다.

- 실질적인 탐지 성능은 Level 1 품질에 더 크게 좌우됩니다.
- Level 2는 개별 서브그래프 점수만으로는 잡기 어려운 관계 맥락을 보강합니다.
- 따라서 MC는 Level 1/Level 2/Fusion이 먼저 안정화된 이후 적용해야 효과와 원인을 분리해 해석할 수 있습니다.

---

## MC의 역할 재정의

PDF 기준으로 MC는 크게 두 가지 역할로 정리할 수 있습니다.

### 1. Feature Approximation / Graph Statistic Estimation

정확 계산 비용이 큰 구조적 특성 계산을 근사합니다.

예시:
- Random Walk with Restart 근사
- Motif 계산 근사
- Centrality 성격의 구조 통계 근사

이 역할의 목적은 **실시간성 확보와 계산 병목 완화**입니다.

### 2. Uncertainty Estimation

모델 예측값의 불확실성을 추정합니다.

예시:
- prediction variance 측정
- low-confidence filtering
- selective prediction
- calibration 보조

이 역할의 목적은 **과신(Overconfidence) 완화와 오탐(False Positive) 감소**입니다.

---

## 개발 우선순위

MC는 아래 순서로 붙이는 것이 바람직합니다.

### Stage 0. 코어 파이프라인 고정

선행 조건:
- Level 1 단독 학습/평가 가능
- Level 2 단독 학습/평가 가능
- Level 1 frozen + Level 2 학습 가능
- Fusion 포함 최종 통합 평가 가능

### Stage 1. Model-level MC 적용

가장 먼저 도입할 MC는 **Model MC**입니다.

핵심 아이디어:
- 평가 시 dropout을 활성화한 상태로 다중 forward 수행
- 평균 score와 불확실성 통계 산출

장점:
- 기존 모델 구조를 크게 바꾸지 않아도 됨
- 현재 구축된 L1/L2/Fusion 파이프라인과의 결합이 수월함
- MC 효과를 가장 해석 가능하게 측정할 수 있음

### Stage 2. Data-level MC 적용

다음은 **Data MC**입니다.

핵심 아이디어:
- 학습 시 edge perturbation, edge dropout 등으로 입력 그래프에 확률적 변형을 가함
- 구조적 희소성과 위상 노이즈에 강한 표현을 유도

장점:
- 데이터 희소성 상황에서 Recall 향상 가능성
- 구조 변동성에 대한 강건성 확보

### Stage 3. Model MC + Data MC 결합

최종적으로 두 방식을 결합합니다.

목표:
- 학습 시 입력 변동성 반영
- 평가 시 예측 불확실성 추정
- 탐지 성능과 신뢰도 모두 확보

---

## MC 개발의 핵심 원칙

### 원칙 1. MC는 코어가 아니라 확장 플러그인이다

코어 파이프라인은 다음 구조로 유지합니다.

- Level 1
- Level 2
- Fusion

MC는 이 구조의 외부에서 결합되는 **Adapter / Strategy Pattern 기반 확장 모듈**이어야 합니다.

금지 사항:
- 코어 모델 내부에 `if use_mc:` 분기 난립
- Level 1 / Level 2 / Fusion 로직과 MC 로직의 강결합

### 원칙 2. 원인 분리가 가능해야 한다

nGNN 또는 Level 1/2 자체가 불안정한 상태에서 MC를 먼저 붙이면,
성능 저하 원인이 다음 중 무엇인지 분리하기 어려워집니다.

- Level 1 구조 문제
- Level 2 relation 처리 문제
- Fusion 로직 문제
- MC sampling / dropout / uncertainty 계산 문제

따라서 MC는 **안정화된 모델 위에서만 결합**해야 합니다.

### 원칙 3. 실시간 탐지 관점에서 계산량을 통제해야 한다

MC는 계산량이 커지기 쉽기 때문에 다음 전략이 중요합니다.

- 전체 그래프에 일괄 적용하지 않기
- Risk score가 높은 노드/서브그래프 주변만 선택적 샘플링
- 다중 forward 횟수 제한
- 캐시와 배치 단위 실험 자동화

---

## 논문/실험 관점에서 기대되는 MC 기여점

### 1. Recall 및 F1 개선

기존 GoG는 데이터 희소성 때문에 은닉된 사기 패턴을 놓칠 수 있습니다.
MC 기반 샘플링은 복잡한 위상 패턴을 증폭시켜 소수 클래스 탐지력을 개선할 가능성이 있습니다.

강조 포인트:
- 정상 거래 탐지력은 유지
- 사기 거래 재현율 상승
- Minor class F1-score 개선

### 2. ROC-AUC 및 확률 보정 개선

기존 모델은 경계 사례에서 과신하는 문제가 있을 수 있습니다.
MC 앙상블을 통해 더 보정된 예측 score를 산출하고, uncertainty를 함께 측정할 수 있습니다.

강조 포인트:
- calibrated probability
- overconfidence 완화
- false positive 감소

### 3. 계산 병목 완화

RWR, motif 등 비용이 큰 계산을 MC 근사로 대체함으로써,
성능 손실을 최소화하면서 실시간 특성 추출 가능성을 확보합니다.

---

## MC 이후 nGNN으로 넘어가는 이유

MC는 현재 파이프라인의 **신뢰도와 실시간성**을 강화합니다.
nGNN은 그 다음 단계에서 **표현력과 계층적 구조 처리 능력**을 강화합니다.

즉 순서는 다음이 자연스럽습니다.

1. L1 / L2 / Fusion 안정화
2. MC로 uncertainty + approximation 체계 구축
3. nGNN으로 local/global propagation 고도화

---

## 요약

현재 단계에서 MC는 다음과 같이 정의할 수 있습니다.

- 목적: 성능 향상 + 불확실성 추정 + 계산 병목 완화
- 위치: 코어 파이프라인 외부의 확장 모듈
- 우선순위: Model MC -> Data MC -> 결합형 MC
- 전제 조건: Level 1 / Level 2 / Fusion이 안정적으로 동작해야 함
- 기대 효과: F1, AUC, Calibration, Robustness 개선
