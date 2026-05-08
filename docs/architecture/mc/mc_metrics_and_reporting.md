# MC Metrics and Reporting

이 문서는 Monte Carlo(MC) 적용 이후 어떤 지표를 중심으로 성능을 측정하고, 어떤 형태로 결과를 해석해야 하는지 정리합니다.

---

## 핵심 평가 지표

PDF에서 Fraud Detection 관점의 핵심 지표로 강조된 것은 다음과 같습니다.

### 1. F1-score

의미:
- Precision과 Recall의 조화평균
- 소수 클래스인 fraud 탐지력을 단일 수치로 반영

MC 관점에서 기대 효과:
- 희소한 사기 패턴을 더 잘 포착
- Recall 향상
- 결과적으로 minor class F1 개선

특히 다음 메시지를 논문/리포트에서 강조할 수 있습니다.

- 정상 거래 탐지력은 유지
- 사기 거래 재현율은 향상
- class imbalance 환경에서 더 유의미한 개선

---

### 2. ROC-AUC

의미:
- threshold와 무관하게 정상/이상 구분 능력을 측정
- score ranking의 품질을 반영

MC 관점에서 기대 효과:
- 경계 사례에서의 과신 완화
- calibration 개선
- false positive 감소

즉 MC는 단순히 score를 높이는 것보다,
**score의 신뢰도와 분별력 자체를 개선**하는 방향으로 해석해야 합니다.

---

### 3. AP (Average Precision)

PDF 문맥상 legacy baseline 결과 해석에 함께 사용된 지표입니다.

의미:
- class imbalance 상황에서 precision-recall 곡선 기반 품질 측정
- fraud처럼 positive 비율이 낮은 환경에서 중요

권장:
- F1, ROC-AUC와 함께 항상 기록
- chain별 imbalance 차이를 감안하여 해석

---

## MC 전용 보고 지표

MC를 넣은 이후에는 일반 분류 지표 외에도 아래 항목을 함께 기록하는 것이 바람직합니다.

### 1. Mean Score

다중 forward의 평균 예측값

활용:
- 최종 fraud score로 사용
- deterministic baseline과 직접 비교

### 2. Prediction Variance

샘플 간 예측 score의 분산

활용:
- 불확실성의 가장 기본적인 척도
- low-confidence 사례 탐지
- 분석용 ranking

### 3. Confidence Bucketing

예측 샘플을 confidence 구간으로 나누어 성능을 기록

예시:
- high confidence
- medium confidence
- low confidence

활용:
- uncertainty가 실제 오류와 연결되는지 검증
- selective prediction 정책 설계

### 4. False Positive under Uncertainty

uncertainty가 높은 구간에서 false positive가 집중되는지 확인

활용:
- 운영단에서 경고 score 후처리 가능성 평가
- threshold 재설계 근거 확보

---

## 권장 리포트 구조

### 공통 리포트 표

| Experiment | Model | F1 | ROC-AUC | AP | Latency | Notes |
|---|---|---:|---:|---:|---:|---|
| Baseline | L1 | - | - | - | - | deterministic |
| MC-Model | L1 + MC | - | - | - | - | eval-time MC |
| MC-Data | L1 + Data MC | - | - | - | - | train-time perturbation |
| MC-Hybrid | L1 + Data + Model MC | - | - | - | - | combined |

---

### 불확실성 리포트 표

| Experiment | Mean Score Source | Uncertainty Metric | High-unc FP Rate | Low-unc FP Rate | Comment |
|---|---|---|---:|---:|---|
| L1 + MC | Level1Output score | variance | - | - | |
| L1+L2 + MC | Fusion score | variance | - | - | |
| Full + MC | final fused score | variance | - | - | |

---

## Uncertainty가 0으로 나오는 문제에 대한 진단 포인트

PDF에서는 이전 baseline 실험에서 `Unc=0.0000` 문제가 분석되었습니다.
이 경험은 새로운 MC 구현에서도 매우 중요합니다.

### 가능한 원인

- dropout이 존재하지만 실제로 비활성화됨
- inference path가 deterministic branch만 탐
- batch norm / decision function 구조상 randomness가 반영되지 않음
- 모델 내부에 stochastic layer 자체가 없음

### 개발 체크리스트

- dropout probability가 0이 아닌가
- evaluation 시 dropout을 강제로 활성화했는가
- 같은 입력에 대해 sample 간 score가 실제 달라지는가
- sample-wise score tensor를 직접 저장해 확인했는가

---

## 체인별 해석 포인트

PDF 기준으로 체인별 데이터 분포 차이가 큽니다.

예시:
- Polygon은 fraud 비율이 매우 낮은 극단적 불균형
- BSC는 더 큰 규모와 복잡한 로컬 그래프를 가짐

따라서 MC 결과는 단순 평균 성능보다 아래 관점으로 해석해야 합니다.

### 1. 불균형 데이터에서 Recall이 실제 늘었는가
### 2. false positive 증가 없이 F1이 개선되었는가
### 3. uncertainty 기반 filtering이 체인별로 다르게 작동하는가
### 4. chain-specific overfitting이 완화되는가

---

## 논문 기여 포인트로 연결되는 해석

MC 결과는 다음 문장으로 귀결될 수 있어야 합니다.

### 자원 효율적 스케일링

- 대규모 블록체인 그래프를 8GB 환경에서 다룰 수 있도록 최적화했다.
- exact한 고비용 계산 대신 MC 근사를 도입했다.

### 성능 손실 없는 근사화

- RWR, motif 등 계산 병목을 MC 샘플링으로 대체했다.
- 성능을 유지하거나 개선하면서 실시간성에 가까운 처리가 가능해졌다.

### uncertainty-aware fraud detection

- deterministic model의 과신 문제를 완화했다.
- calibrated score와 uncertainty를 통해 더 신뢰도 높은 탐지를 수행했다.

---

## 로그 및 저장 권장 항목

각 실험 run마다 아래를 저장하는 것이 좋습니다.

- seed
- config yaml snapshot
- model checkpoint hash 또는 path
- dataset split 정보
- chain 정보
- F1 / ROC-AUC / AP
- mean latency
- mc_samples
- dropout / edge_dropout 설정
- uncertainty summary statistics
- sample-level prediction dump 경로

---

## 요약

MC의 성능 평가는 단순 분류 성능 비교를 넘어야 합니다.

중점 포인트:
- F1, ROC-AUC, AP
- uncertainty의 실제 유효성
- false positive 감소 여부
- 8GB 환경에서의 계산 효율성
- chain 간 일반화 성능

즉, MC는 단순한 성능 개선 옵션이 아니라
**신뢰 가능한 fraud detection을 위한 확률적 확장 계층**으로 보고 평가해야 합니다.
