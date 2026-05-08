from pathlib import Path
from typing import List

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gog_fraud.data.level2.relation_builder import (
    RelationBuilderConfig,
    build_level2_graph,
    build_knn_edges,
    build_temporal_window_edges,
    build_shared_entity_edges,
    derive_level2_label,
    save_level2_graph,
)
from gog_fraud.data.level2.dataset import (
    Level2GraphDataset,
    infer_level2_node_dim,
    infer_level2_edge_dim,
    load_level2_graph_list_from_pt,
    save_level2_graph_list,
    validate_level2_graph,
)
from gog_fraud.models.level2.model import Level2Model, Level2ModelConfig
from gog_fraud.training.loops.level2 import Level2Trainer, Level2TrainerConfig
from gog_fraud.pipelines.train_level2 import run_training


# ──────────────────────────────────────────────
# Synthetic fixtures
# ──────────────────────────────────────────────

LEVEL1_EMB_DIM = 16
N_LEVEL1_NODES = 6   # number of Level 1 graphs = nodes in Level 2


def make_synthetic_level1_bundle(
    n: int = N_LEVEL1_NODES,
    emb_dim: int = LEVEL1_EMB_DIM,
    with_label: bool = True,
) -> dict:
    """
    Fake a Level 1 embedding bundle with n entries.
    """
    bundle = {
        "graph_id": torch.arange(n, dtype=torch.long),
        "embedding": torch.randn(n, emb_dim),
        "logits": torch.randn(n, 1),
        "score": torch.rand(n, 1),
        "metadata": {
            "timestamp": torch.arange(n, dtype=torch.float32),
        },
    }
    if with_label:
        labels = torch.zeros(n, 1)
        labels[::2] = 1.0   # alternating
        bundle["label"] = labels
    else:
        bundle["label"] = None
    return bundle


def make_synthetic_level2_graph(
    n: int = N_LEVEL1_NODES,
    emb_dim: int = LEVEL1_EMB_DIM,
    knn_k: int = 2,
    with_label: bool = True,
) -> Data:
    bundle = make_synthetic_level1_bundle(n=n, emb_dim=emb_dim, with_label=with_label)
    cfg = RelationBuilderConfig(
        relation_modes=["embedding_knn"],
        knn_k=knn_k,
        knn_similarity="cosine",
        knn_self_loops=False,
        include_edge_weight=True,
        level2_label_strategy="any",
    )
    return build_level2_graph(bundle=bundle, cfg=cfg)


def make_level2_graph_list(
    n_graphs: int = 6,
    n_nodes: int = N_LEVEL1_NODES,
    emb_dim: int = LEVEL1_EMB_DIM,
) -> List[Data]:
    graphs = []
    for i in range(n_graphs):
        g = make_synthetic_level2_graph(
            n=n_nodes,
            emb_dim=emb_dim,
            knn_k=2,
            with_label=True,
        )
        # override y for variety
        g.y = torch.tensor([float(i % 2)])
        graphs.append(g)
    return graphs


# ──────────────────────────────────────────────
# Tests: relation_builder
# ──────────────────────────────────────────────

def test_knn_edge_builder_shape_and_symmetry():
    emb = torch.randn(8, 16)
    edge_index, edge_weight = build_knn_edges(emb, k=3, similarity="cosine")

    # edges must be [2, E]
    assert edge_index.dim() == 2
    assert edge_index.size(0) == 2

    # edge_weight matches number of edges
    assert edge_weight.shape[0] == edge_index.size(1)

    # all node indices must be valid
    assert int(edge_index.max().item()) <= 7
    assert int(edge_index.min().item()) >= 0

    # cosine similarity weights must be <= 1.0
    assert float(edge_weight.max().item()) <= 1.0 + 1e-5


def test_temporal_window_edge_builder():
    timestamps = torch.tensor([3.0, 1.0, 4.0, 0.0, 2.0])
    edge_index, edge_weight = build_temporal_window_edges(timestamps, window_size=2)

    assert edge_index.dim() == 2
    assert edge_index.size(0) == 2
    assert edge_weight.shape[0] == edge_index.size(1)

    # all indices in valid range
    assert int(edge_index.max().item()) <= 4
    assert int(edge_index.min().item()) >= 0


def test_shared_entity_edge_builder_adj_matrix():
    adj = torch.zeros(5, 5)
    adj[0, 1] = 1.0
    adj[1, 0] = 1.0
    adj[2, 3] = 0.5
    adj[3, 2] = 0.5

    edge_index, edge_weight = build_shared_entity_edges(adj, num_nodes=5)
    assert edge_index.size(0) == 2
    assert edge_weight.shape[0] == edge_index.size(1)
    assert edge_index.size(1) == 4  # 4 directed edges


def test_shared_entity_edge_builder_edge_list():
    edge_list = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    edge_index, edge_weight = build_shared_entity_edges(edge_list, num_nodes=4)
    assert edge_index.size(0) == 2
    assert edge_index.size(1) == 3
    assert edge_weight.shape[0] == 3


def test_derive_level2_label_strategies():
    labels = torch.tensor([0.0, 1.0, 0.0, 0.0])

    assert derive_level2_label(labels, strategy="any").item() == 1.0
    assert derive_level2_label(labels, strategy="majority").item() == 0.0
    assert abs(derive_level2_label(labels, strategy="mean").item() - 0.25) < 1e-6
    assert derive_level2_label(labels, strategy="max").item() == 1.0


def test_build_level2_graph_knn():
    bundle = make_synthetic_level1_bundle(n=8, emb_dim=16)
    cfg = RelationBuilderConfig(
        relation_modes=["embedding_knn"],
        knn_k=3,
        knn_similarity="cosine",
        include_edge_weight=True,
        level2_label_strategy="any",
    )
    g = build_level2_graph(bundle, cfg)

    # node features = embedding (16) + score (1)
    assert g.x.shape == (8, 17)
    assert g.edge_index.dim() == 2
    assert g.edge_attr is not None
    assert g.y is not None
    assert g.level1_embedding.shape == (8, 16)
    assert g.level1_score.shape == (8, 1)


def test_build_level2_graph_temporal():
    bundle = make_synthetic_level1_bundle(n=6, emb_dim=8)
    cfg = RelationBuilderConfig(
        relation_modes=["temporal_window"],
        temporal_window_size=2,
        timestamp_attr="timestamp",
        include_edge_weight=True,
        level2_label_strategy="majority",
    )
    g = build_level2_graph(bundle, cfg)

    assert g.x.shape == (6, 9)  # 8 emb + 1 score
    assert g.edge_index.size(0) == 2
    assert g.y is not None


def test_build_level2_graph_multi_relation():
    bundle = make_synthetic_level1_bundle(n=6, emb_dim=8)
    cfg = RelationBuilderConfig(
        relation_modes=["embedding_knn", "temporal_window"],
        knn_k=2,
        temporal_window_size=2,
        include_edge_weight=True,
        level2_label_strategy="any",
    )
    g = build_level2_graph(bundle, cfg)

    assert g.x.shape[0] == 6
    # multi-relation should still produce valid deduplicated edges
    assert g.edge_index.size(0) == 2
    assert g.edge_attr.shape[0] == g.edge_index.size(1)


# ──────────────────────────────────────────────
# Tests: dataset
# ──────────────────────────────────────────────

def test_level2_dataset_validate_and_dims():
    graphs = make_level2_graph_list(n_graphs=4)

    dataset = Level2GraphDataset(graphs, require_label=True, validate=True)
    assert len(dataset) == 4

    # node dim = emb (16) + score (1) = 17
    assert infer_level2_node_dim(dataset.graphs) == 17
    # edge dim = 1 (weight)
    assert infer_level2_edge_dim(dataset.graphs) == 1


def test_level2_dataset_save_and_load(tmp_path: Path):
    graphs = make_level2_graph_list(n_graphs=5)
    dataset = Level2GraphDataset(graphs)

    saved_path = tmp_path / "level2_graphs.pt"
    dataset.to_pt(str(saved_path))

    loaded = load_level2_graph_list_from_pt(
        str(saved_path), require_label=True, validate=True
    )
    assert len(loaded) == 5
    assert loaded[0].x.shape == graphs[0].x.shape


# ──────────────────────────────────────────────
# Tests: model forward
# ──────────────────────────────────────────────

def test_level2_model_forward():
    graphs = make_level2_graph_list(n_graphs=4)
    loader = DataLoader(graphs, batch_size=2, shuffle=False)
    batch  = next(iter(loader))

    cfg = Level2ModelConfig(
        in_dim=16,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        edge_dim=1,
        readout="meanmax",
        out_dim=1,
    )
    model = Level2Model(cfg)
    out   = model(batch)

    assert out.logits.shape == (2, 1)
    assert out.score.shape  == (2, 1)
    assert out.embedding.shape == (2, cfg.hidden_dim * 2)   # meanmax
    assert out.label.shape  == (2, 1)

    # graph_id는 graph-level: batch 내 2개의 Level 2 graph → [0, 1]
    assert out.graph_id.shape[0] == 2
    assert out.graph_id.tolist() == [0, 1]




def test_level2_model_forward_no_edge_attr():
    """Model must work even if edge_attr is not provided."""
    graphs = make_level2_graph_list(n_graphs=4)
    # strip edge_attr
    for g in graphs:
        g.edge_attr = None

    loader = DataLoader(graphs, batch_size=2, shuffle=False)
    batch  = next(iter(loader))

    cfg = Level2ModelConfig(
        in_dim=16,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        edge_dim=0,     # no edge feature
        readout="mean",
        out_dim=1,
    )
    model = Level2Model(cfg)
    out   = model(batch)

    assert out.logits.shape == (2, 1)


# ──────────────────────────────────────────────
# Tests: trainer
# ──────────────────────────────────────────────

def test_level2_trainer_train_and_eval():
    graphs = make_level2_graph_list(n_graphs=8)
    loader = DataLoader(graphs, batch_size=4, shuffle=True)

    cfg = Level2ModelConfig(
        in_dim=16, hidden_dim=16, num_layers=1,
        num_heads=4, dropout=0.0, edge_dim=1,
        readout="meanmax", out_dim=1,
    )
    model = Level2Model(cfg)

    trainer_cfg = Level2TrainerConfig(
        lr=1e-3, weight_decay=1e-4, epochs=1,
        batch_size=4, grad_accum_steps=1, use_amp=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer_cfg.lr)
    trainer   = Level2Trainer(
        model=model, optimizer=optimizer, cfg=trainer_cfg, device="cpu"
    )

    train_metrics = trainer.train_one_epoch(loader)
    eval_metrics  = trainer.evaluate(loader)

    for key in ["loss", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert key in train_metrics, f"Missing key in train_metrics: {key}"
        assert key in eval_metrics,  f"Missing key in eval_metrics: {key}"

    assert train_metrics["loss"] >= 0.0


# ──────────────────────────────────────────────
# Tests: pipeline
# ──────────────────────────────────────────────

def test_train_level2_pipeline_saves_checkpoints(tmp_path: Path):
    train_graphs = make_level2_graph_list(n_graphs=8)
    valid_graphs = make_level2_graph_list(n_graphs=4)
    output_dir   = tmp_path / "artifacts" / "level2"

    result = run_training(
        train_graphs=train_graphs,
        valid_graphs=valid_graphs,
        output_dir=str(output_dir),
        model_cfg=Level2ModelConfig(
            in_dim=16, hidden_dim=16, num_layers=1,
            num_heads=4, dropout=0.0, edge_dim=1,
            readout="meanmax", out_dim=1,
        ),
        trainer_cfg=Level2TrainerConfig(
            lr=1e-3, weight_decay=1e-4, epochs=2,
            batch_size=4, grad_accum_steps=1, use_amp=False,
        ),
        device="cpu",
    )

    assert Path(result["best_checkpoint"]).exists()
    assert Path(result["last_checkpoint"]).exists()
    assert len(result["history"]) == 2

    # checkpoint must be loadable
    ckpt = torch.load(
        result["best_checkpoint"], map_location="cpu", weights_only=False
    )
    assert "model_state_dict" in ckpt
    assert "model_config"     in ckpt
    assert "trainer_config"   in ckpt
    assert "epoch"            in ckpt
    assert "valid_metrics"    in ckpt


def test_level1_to_level2_full_pipeline(tmp_path: Path):
    """
    End-to-end: Level 1 bundle → Level 2 graph → train → checkpoint.
    Validates the Level 1 → Level 2 boundary.
    """
    # Step 1: Simulate Level 1 bundle export (as produced by export_level1_embeddings)
    bundle = make_synthetic_level1_bundle(n=8, emb_dim=16, with_label=True)

    # Step 2: Build Level 2 graph
    cfg_builder = RelationBuilderConfig(
        relation_modes=["embedding_knn", "temporal_window"],
        knn_k=3,
        temporal_window_size=2,
        include_edge_weight=True,
        level2_label_strategy="any",
    )
    level2_graph = build_level2_graph(bundle, cfg_builder)

    # Step 3: Replicate into a small dataset
    import copy
    train_graphs = [copy.deepcopy(level2_graph) for _ in range(6)]
    valid_graphs = [copy.deepcopy(level2_graph) for _ in range(3)]
    for i, g in enumerate(train_graphs + valid_graphs):
        g.y = torch.tensor([float(i % 2)])

    in_dim   = train_graphs[0].x.size(-1)
    edge_dim = 1 if (train_graphs[0].edge_attr is not None) else 0

    # Step 4: Train
    output_dir = tmp_path / "l1_to_l2"
    result = run_training(
        train_graphs=train_graphs,
        valid_graphs=valid_graphs,
        output_dir=str(output_dir),
        model_cfg=Level2ModelConfig(
            in_dim=in_dim,
            hidden_dim=16, num_layers=1, num_heads=4,
            dropout=0.0, edge_dim=edge_dim,
            readout="meanmax", out_dim=1,
        ),
        trainer_cfg=Level2TrainerConfig(
            lr=1e-3, epochs=1, batch_size=3,
            grad_accum_steps=1, use_amp=False,
        ),
        device="cpu",
    )

    assert Path(result["best_checkpoint"]).exists()
    history = result["history"]
    assert len(history) == 1

    train_metrics = history[0]["train"]
    valid_metrics = history[0]["valid"]

    assert "loss"    in train_metrics
    assert "pr_auc"  in valid_metrics
    assert "f1"      in valid_metrics
