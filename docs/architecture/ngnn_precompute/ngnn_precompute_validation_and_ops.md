

# nGNN Precompute Validation and Operations

이 문서는 nGNN rooted subgraph precompute 결과를 어떻게 검증하고,
운영 시 어떤 체크리스트를 따라야 하는지 정리합니다.

---

## 왜 검증이 중요한가

precompute 파이프라인은 한 번 잘못 돌리면
그 위의 모든 nGNN 실험이 잘못된 입력에 기반하게 됩니다.

특히 다음 문제는 조기에 잡아야 합니다.

- root가 잘못 잡힘
- hop 추출 범위 오류
- 노드 remapping 오류
- manifest 누락
- label misalignment
- split leakage
- artifact 손상

따라서 precompute 결과는 단순 저장만이 아니라,
**구조적 무결성 검증**과 **운영 검증**을 반드시 포함해야 합니다.

---

## 검증 계층

검증은 다음 4단계로 나누는 것이 좋습니다.

### 1. sample-level structural validation

각 artifact 단위로 검사합니다.

체크 항목:
- `x` 존재 여부
- `edge_index` shape 유효성
- `root_node_idx` 범위 유효성
- `root_indicator`와 root index 일치 여부
- `hop_ids` 길이와 노드 수 일치 여부
- `original_node_ids` 존재 여부
- label / sample_id / contract_id 존재 여부

---

### 2. split-level consistency validation

split 전체에 대해 검사합니다.

체크 항목:
- manifest row 수와 실제 파일 수 일치 여부
- sample id 중복 여부
- contract id 중복 정책 확인
- train / val / test leakage 여부
- split별 label 분포 확인

---

### 3. config-level reproducibility validation

같은 config에서 재실행 시 결과가 일관적인지 확인합니다.

체크 항목:
- config hash 동일 여부
- artifact version 일치 여부
- deterministic extraction 재현 여부
- summary 통계 차이 여부

---

### 4. downstream compatibility validation

학습 파이프라인과 실제로 연결 가능한지 확인합니다.

체크 항목:
- DataLoader 로딩 성공 여부
- PyG batching 성공 여부
- `Level1nGNN.forward()` 입력 shape 적합 여부
- output embedding dim 정상 여부

---

## 권장 자동 검증 항목

### Smoke validation

작은 샘플 수에 대해 빠르게 확인합니다.

예시:
- 샘플 10개 precompute
- artifact 저장 후 즉시 로딩
- `Level1nGNN` forward 1회 실행
- embedding dim 확인

### Full validation

split 전체에 대해 수행합니다.

예시:
- manifest integrity 검사
- 통계 리포트 생성
- 실패 샘플 로그 생성
- 랜덤 샘플 시각 점검

---

## 권장 통계 리포트

precompute가 끝난 뒤 아래 지표를 요약하는 것이 좋습니다.

### 구조 통계
- 총 샘플 수
- split별 샘플 수
- 평균 노드 수
- 평균 edge 수
- hop별 평균 노드 수

### trimming 통계
- trim 발생 비율
- trim 전/후 평균 노드 수
- root 유지율
- trim 실패 수

### 예외 통계
- root 미탐지 수
- 빈 graph 수
- serialization 실패 수
- invalid metadata 수

---

## 운영 모드 권장안

### 1. dry-run mode

역할:
- 저장 없이 추출만 수행
- 통계와 예외만 확인

활용:
- 새 config 검증
- hop/trim 정책 점검
- 디스크 사용량 예측

### 2. normal mode

역할:
- artifact 저장
- manifest 생성
- summary 저장

### 3. strict mode

역할:
- 오류 하나라도 발생하면 즉시 중단
- benchmark용 artifact 생성 시 사용

### 4. resume mode

역할:
- 이미 생성된 artifact는 건너뛰고 누락분만 재생성

---

## 시각적 점검 권장안

자동 검증 외에도 일부 샘플은 사람이 직접 확인하는 것이 좋습니다.

권장 확인 항목:
- root node가 의도한 노드인지
- 1-hop / 2-hop 범위가 맞는지
- trim 정책이 이상하게 작동하지 않는지
- edge attr가 누락되지 않았는지

가능하다면 notebook 또는 debug script로 다음을 지원합니다.

- 원본 graph 시각화
- rooted subgraph 시각화
- hop color 표시
- root 강조 표시

---

## 자주 발생할 수 있는 문제

### 1. root가 잘못 매핑되는 경우

증상:
- root indicator가 여러 개이거나 0개
- root_node_idx가 edge_index 범위를 벗어남

원인:
- subgraph extraction 후 remapping 오류
- trim 후 root index 갱신 누락

### 2. trim 후 의미 손실이 큰 경우

증상:
- 대부분 샘플이 강하게 trim됨
- hop 2 정보가 거의 사라짐
- 성능이 baseline보다 악화

원인:
- `max_nodes_per_subgraph`가 너무 작음
- trim priority가 task와 맞지 않음

### 3. 파일 수 과다로 I/O 병목이 생기는 경우

증상:
- DataLoader가 느림
- 학습보다 파일 열기 비용이 큼

대응:
- shard 저장 검토
- runtime memory cache 사용
- SSD 기준 경로 사용

### 4. split leakage가 생기는 경우

증상:
- validation/test 성능이 비정상적으로 높음

원인:
- manifest 생성 시 sample id 중복
- 동일 contract의 파생 artifact가 split을 넘나듦

---

## downstream 검증 체크리스트

nGNN precompute 완료 후 최소한 아래를 확인해야 합니다.

- `Level1nGNN`가 precomputed artifact를 정상 로드하는가
- embedding dimension이 기존 L1 backend와 호환되는가
- L1 only 학습이 정상 수행되는가
- L1 embedding cache 생성이 가능한가
- L2가 nGNN embedding을 입력으로 받을 수 있는가

---

## 운영 관점 권장 산출물

각 precompute run은 다음 파일을 남기는 것이 좋습니다.

- `config.yaml`
- `summary.json`
- `train_manifest.jsonl`
- `val_manifest.jsonl`
- `test_manifest.jsonl`
- `failures.jsonl`
- optional `samples_debug/`

---

## 요약

nGNN precompute는 단순 전처리가 아니라,
전체 nGNN 실험의 입력 신뢰성을 보장하는 기반 계층입니다.

중점 포인트:
- sample-level structural validation
- split-level consistency
- config-level reproducibility
- downstream compatibility
- 운영 모드 분리
- 실패 및 예외 로그 관리

즉 precompute 파이프라인은
**빠르게 한 번 돌리는 스크립트**가 아니라
**반복 가능한 실험 인프라의 일부**로 설계되어야 합니다.
