import json
import random
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gog_fraud.data.level1.dataset import (
    Level1GraphDataset,
    get_graph_ids,
    infer_in_dim,
    infer_struct_dim,
    load_graph_list,
    save_graph_list,
)
from gog_fraud.data.level1.builder import (
    Level1BuildConfig,
    build_level1_split_bundle,
    build_and_save_level1_splits,
)
from gog_fraud.evaluation.fraud_metrics import (
    binary_classification_metrics,
    find_best_f1_threshold,
    topk_metrics,
)
from gog_fraud.evaluation.evaluator import Level1Evaluator
from gog_fraud.models.level1.model import Level1Model, Level1ModelConfig


def make_synthetic_graph(
    graph_id: int,
    label: int,
    timestamp: int,
    num_nodes: int = 5,
    in_dim: int = 16,
    struct_dim: int = 4,
    with_graph_id: bool = True,
):
    x = torch.randn(num_nodes, in_dim)

    src = torch.arange(0, num_nodes - 1, dtype=torch.long)
    dst = torch.arange(1, num_nodes, dtype=torch.long)
    edge_index = torch.stack(
        [torch.cat([src, dst]), torch.cat([dst, src])],
        dim=0,
    )

    y = torch.tensor([label], dtype=torch.float32)
    struct_feat = torch.tensor(
        [[
            float(graph_id),
            float(graph_id + 1),
            float(graph_id + 2),
            float(graph_id + 3),
        ]],
        dtype=torch.float32,
    )
    timestamp_tensor = torch.tensor([timestamp], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        struct_feat=struct_feat,
        timestamp=timestamp_tensor,
    )
    if with_graph_id:
        data.graph_id = torch.tensor([graph_id], dtype=torch.long)
    return data


def make_dataset(
    n_graphs: int = 12,
    in_dim: int = 16,
    struct_dim: int = 4,
    include_graph_id: bool = True,
):
    graphs = []
    for i in range(n_graphs):
        graphs.append(
            make_synthetic_graph(
                graph_id=i,
                label=i % 2,
                timestamp=i,
                num_nodes=5 + (i % 3),
                in_dim=in_dim,
                struct_dim=struct_dim,
                with_graph_id=include_graph_id,
            )
        )
    return graphs


def collect_struct_matrix(graphs):
    rows = []
    for g in graphs:
        feat = g.struct_feat
        if feat.dim() == 1:
            feat = feat.view(1, -1)
        rows.append(feat)
    return torch.cat(rows, dim=0)


def test_level1_dataset_load_and_infer_dims(tmp_path: Path):
    graphs = make_dataset(n_graphs=6, in_dim=16, struct_dim=4, include_graph_id=False)
    data_path = tmp_path / "graphs.pt"
    save_graph_list(str(data_path), graphs)

    loaded = load_graph_list(
        path=str(data_path),
        require_label=True,
        require_graph_id=True,
        validate=True,
    )
    dataset = Level1GraphDataset(loaded)

    assert len(dataset) == 6
    assert infer_in_dim(dataset.graphs) == 8
    assert infer_struct_dim(dataset.graphs) == 4

    graph_ids = get_graph_ids(dataset.graphs)
    assert graph_ids.shape[0] == 6
    assert graph_ids.tolist() == [0, 1, 2, 3, 4, 5]


def test_builder_random_split_is_reproducible():
    graphs = make_dataset(n_graphs=10, include_graph_id=True)

    cfg = Level1BuildConfig(
        split_mode="random",
        train_ratio=0.6,
        valid_ratio=0.2,
        test_ratio=0.2,
        seed=123,
        normalize_struct_features=False,
    )

    bundle1 = build_level1_split_bundle(graphs, cfg)
    bundle2 = build_level1_split_bundle(graphs, cfg)

    train_ids_1 = [int(g.graph_id.view(-1)[0].item()) for g in bundle1.train_graphs]
    train_ids_2 = [int(g.graph_id.view(-1)[0].item()) for g in bundle2.train_graphs]

    assert train_ids_1 == train_ids_2
    assert len(bundle1.train_graphs) == 6
    assert len(bundle1.valid_graphs) == 2
    assert len(bundle1.test_graphs) == 2


def test_builder_temporal_split_and_struct_normalization(tmp_path: Path):
    graphs = make_dataset(n_graphs=10, include_graph_id=True)
    random.Random(2024).shuffle(graphs)

    cfg = Level1BuildConfig(
        split_mode="temporal",
        train_ratio=0.5,
        valid_ratio=0.2,
        test_ratio=0.3,
        seed=42,
        timestamp_attr="timestamp",
        normalize_struct_features=True,
        output_dir=str(tmp_path / "processed"),
    )

    result = build_and_save_level1_splits(graphs, cfg)
    bundle = result["bundle"]
    saved = result["saved"]

    assert len(bundle.train_graphs) == 5
    assert len(bundle.valid_graphs) == 2
    assert len(bundle.test_graphs) == 3

    train_ids = [int(g.graph_id.view(-1)[0].item()) for g in bundle.train_graphs]
    valid_ids = [int(g.graph_id.view(-1)[0].item()) for g in bundle.valid_graphs]
    test_ids = [int(g.graph_id.view(-1)[0].item()) for g in bundle.test_graphs]

    assert train_ids == [0, 1, 2, 3, 4]
    assert valid_ids == [5, 6]
    assert test_ids == [7, 8, 9]

    train_struct = collect_struct_matrix(bundle.train_graphs)
    train_mean = train_struct.mean(dim=0)
    assert torch.allclose(train_mean, torch.zeros_like(train_mean), atol=1e-6)

    assert Path(saved["train_path"]).exists()
    assert Path(saved["valid_path"]).exists()
    assert Path(saved["test_path"]).exists()
    assert Path(saved["metadata_path"]).exists()

    with open(saved["metadata_path"], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["num_train_graphs"] == 5
    assert metadata["struct_normalization"]["enabled"] is True
    assert metadata["struct_normalization"]["applied"] is True
    assert metadata["struct_normalization"]["attr_name"] == "struct_feat"


def test_fraud_metrics_basic():
    y_true = torch.tensor([0, 1, 1, 0, 1], dtype=torch.float32)
    y_score = torch.tensor([0.10, 0.90, 0.80, 0.20, 0.70], dtype=torch.float32)

    metrics = binary_classification_metrics(y_true, y_score, threshold=0.5)
    best = find_best_f1_threshold(y_true, y_score)
    topk = topk_metrics(y_true, y_score, k=2)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0

    assert 0.0 <= best["best_threshold"] <= 1.0
    assert 0.0 <= best["best_f1"] <= 1.0

    assert topk["effective_k"] == 2
    assert topk["precision_at_k"] == 1.0
    assert abs(topk["recall_at_k"] - (2.0 / 3.0)) < 1e-6


def test_level1_evaluator_loader_and_bundle():
    graphs = make_dataset(n_graphs=8, in_dim=16, struct_dim=4, include_graph_id=True)
    loader = DataLoader(graphs, batch_size=4, shuffle=False)

    model_cfg = Level1ModelConfig(
        in_dim=16,
        hidden_dim=16,
        num_layers=2,
        dropout=0.1,
        readout="meanmax",
        struct_dim=4,
        struct_hidden_dim=8,
        out_dim=1,
    )
    model = Level1Model(model_cfg)

    evaluator = Level1Evaluator(device="cpu")
    result = evaluator.evaluate_loader(
        model=model,
        loader=loader,
        threshold=0.5,
        topk=[1, 3],
        search_best_threshold=True,
        return_bundle=True,
    )

    metrics = result["metrics"]
    bundle = result["bundle"]

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "bce_loss" in metrics
    assert "best_threshold" in metrics
    assert "best_f1" in metrics
    assert "top1_precision" in metrics
    assert "top3_precision" in metrics

    assert bundle["graph_id"].shape[0] == len(graphs)
    assert bundle["embedding"].shape[0] == len(graphs)
    assert bundle["logits"].shape == (len(graphs), 1)
    assert bundle["score"].shape == (len(graphs), 1)
    assert bundle["label"].shape == (len(graphs), 1)

    metrics_from_bundle = evaluator.evaluate_bundle(
        bundle=bundle,
        threshold=0.5,
        topk=2,
        search_best_threshold=True,
    )
    assert "accuracy" in metrics_from_bundle
    assert "top2_precision" in metrics_from_bundle
