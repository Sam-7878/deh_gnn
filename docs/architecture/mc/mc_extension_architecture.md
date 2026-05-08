# MC Extension Architecture

이 문서는 기존 Fraud Detection 코어 구조(Level 1 / Level 2 / Fusion)를 유지한 채, Monte Carlo(MC) 기능을 어떻게 확장 모듈로 설계할지 설명합니다.

---

## 위치와 설계 방향

기존 리팩토링 방향에 맞추어 MC 관련 구현은 코어 모델 폴더와 분리합니다.

권장 위치:

- `src/gog_fraud/models/extensions/mc/`
- `src/gog_fraud/training/`
- `src/gog_fraud/pipelines/`
- `configs/`

초기 PDF 초안에서는 다음과 같은 전용 폴더가 제안되었습니다.

- `mc/mc_dropout.py`
- `mc/mc_metrics.py`

리팩토링 버전에서는 이를 아래처럼 옮기는 편이 일관적입니다.

- `src/gog_fraud/models/extensions/mc/mc_dropout.py`
- `src/gog_fraud/models/extensions/mc/mc_metrics.py`
- `src/gog_fraud/models/extensions/mc/interfaces.py`
- `src/gog_fraud/models/extensions/mc/adaptive_sampling.py`

---

## MC 확장의 두 축

### 1. Model-level MC

평가 시점에 dropout을 유지한 채 여러 번 forward를 수행하여 score 분포를 얻는 방식입니다.

핵심 개념:
- train-time dropout을 inference에서 재활용
- multiple stochastic forward
- mean score + uncertainty 산출

적용 대상:
- Level 1 단독 모델
- Level 2 단독 모델
- Fusion 최종 모델

장점:
- 기존 구조 변경 최소화
- uncertainty 도입이 가장 간단함
- 현재 benchmark 파이프라인에 붙이기 쉬움

---

### 2. Data-level MC

입력 그래프 자체에 확률적 변형을 가해 학습하는 방식입니다.

예시:
- edge dropout
- edge perturbation
- 부분 neighborhood sampling
- high-risk 지역 집중 샘플링

핵심 개념:
- 구조적 변동성에 대한 강건성 확보
- 데이터 희소성과 노이즈에 대응
- 관계 구조에 대한 일반화 성능 향상

---

## 권장 인터페이스

PDF 기획 방향상, MC는 코어 로직에 직접 박아 넣기보다 인터페이스 기반 확장으로 두는 것이 적절합니다.

### UncertaintyEstimator

```python
class UncertaintyEstimator:
    def estimate(self, model, batch) -> dict:
        raise NotImplementedError

```
