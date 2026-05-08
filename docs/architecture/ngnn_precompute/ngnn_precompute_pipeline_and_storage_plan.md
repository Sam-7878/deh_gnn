# nGNN Precompute Pipeline and Storage Plan

이 문서는 nGNN rooted subgraph precompute를 실제로 어떻게 실행하고,
어떤 저장 구조와 config를 사용할지 설명합니다.

---

## 권장 디렉토리 구조

다음과 같이 역할을 분리하는 것을 권장합니다.

- `src/gog_fraud/models/extensions/ngnn/`
- `src/gog_fraud/data/preprocessing/ngnn/`
- `src/gog_fraud/data/datasets/`
- `src/gog_fraud/pipelines/`
- `configs/ngnn/`

예시:

- `src/gog_fraud/data/preprocessing/ngnn/precompute_rooted_subgraphs.py`
- `src/gog_fraud/data/preprocessing/ngnn/subgraph_extraction.py`
- `src/gog_fraud/data/preprocessing/ngnn/serialization.py`
- `src/gog_fraud/data/preprocessing/ngnn/manifest.py`
- `src/gog_fraud/data/datasets/ngnn_precomputed_dataset.py`
- `src/gog_fraud/pipelines/run_ngnn_precompute.py`

---

## 실행 파이프라인 개요

precompute는 아래 단계로 구성됩니다.

### Step 1. 입력 split 로딩

- train / val / test split 로딩
- sample id 및 contract id 확보
- 원본 graph 유효성 검증

### Step 2. extraction config 고정

예시:
- `num_hops`
- `max_nodes_per_subgraph`
- `root_policy`
- `include_edge_attr`
- `trim_policy`
- `artifact_format`

이 config는 이후 artifact와 함께 저장되어야 합니다.

### Step 3. rooted subgraph 추출

각 sample마다:
- root 결정
- k-hop rooted subgraph 추출
- node remap
- metadata 생성

### Step 4. artifact 저장

권장 저장:
- rooted subgraph `.pt`
- split별 manifest `.jsonl` 또는 `.csv`
- extraction summary `.json`
- config snapshot `.yaml`

### Step 5. 통계 리포트 생성

예시:
- 평균 노드 수
- 평균 edge 수
- hop별 분포
- trimming 발생 비율
- 실패 샘플 수

---

## 권장 artifact 저장 구조

예시 구조:

```text
artifacts/
  ngnn/
    ethereum/
      v1_h1_max128_root-contract/
        config.yaml
        summary.json
        train_manifest.jsonl
        val_manifest.jsonl
        test_manifest.jsonl
        train/
          sample_000001.pt
          sample_000002.pt
        val/
          sample_010001.pt
        test/
          sample_020001.pt
