from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# import sys
# from pathlib import Path

# # 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
# ROOT = Path(__file__).resolve().parent.parent / 'src'
# # ROOT = Path(__file__).resolve().parent
# sys.path.append(str(ROOT))

from gog_fraud.common.types import Level1EmbeddingBundle, Level1Output
from gog_fraud.models.level1.model import Level1Model, Level1ModelConfig
from gog_fraud.training.loops.level1 import Level1Trainer, Level1TrainerConfig
from gog_fraud.pipelines.train_level1 import run_training
from gog_fraud.pipelines.export_level1_embeddings import export_level1_embeddings


def make_synthetic_graph(
    graph_id: int,
    label: int,
    num_nodes: int = 5,
    in_dim: int = 16,
    struct_dim: int = 4,
):
    x = torch.randn(num_nodes, in_dim)

    # 단순 chain graph
    src = torch.arange(0, num_nodes - 1, dtype=torch.long)
    dst = torch.arange(1, num_nodes, dtype=torch.long)
    edge_index = torch.stack(
        [torch.cat([src, dst]), torch.cat([dst, src])],
        dim=0,
    )

    y = torch.tensor([label], dtype=torch.float32)
    graph_id_tensor = torch.tensor([graph_id], dtype=torch.long)
    struct_feat = torch.randn(1, struct_dim)

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        graph_id=graph_id_tensor,
        struct_feat=struct_feat,
    )


def make_dataset(n_graphs: int = 12, in_dim: int = 16, struct_dim: int = 4):
    graphs = []
    for i in range(n_graphs):
        label = i % 2
        graphs.append(
            make_synthetic_graph(
                graph_id=i,
                label=label,
                num_nodes=5 + (i % 3),
                in_dim=in_dim,
                struct_dim=struct_dim,
            )
        )
    return graphs


def test_types_contract():
    out = Level1Output(
        graph_id=torch.tensor([0, 1]),
        embedding=torch.randn(2, 16),
        logits=torch.randn(2, 1),
        score=torch.rand(2, 1),
        label=torch.tensor([[0.0], [1.0]]),
        aux={"phase": "unit-test"},
    )
    bundle = Level1EmbeddingBundle(
        graph_id=out.graph_id,
        embedding=out.embedding,
        logits=out.logits,
        score=out.score,
        label=out.label,
        metadata={"ok": True},
    )

    assert out.embedding.shape == (2, 16)
    assert bundle.score.shape == (2, 1)
    assert bundle.metadata["ok"] is True


def test_level1_model_forward():
    graphs = make_dataset(n_graphs=4, in_dim=16, struct_dim=4)
    loader = DataLoader(graphs, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    cfg = Level1ModelConfig(
        in_dim=16,
        hidden_dim=16,
        num_layers=2,
        dropout=0.1,
        readout="meanmax",
        struct_dim=4,
        struct_hidden_dim=8,
    )
    model = Level1Model(cfg)
    out = model(batch)

    assert out.graph_id.shape[0] == 2
    assert out.logits.shape == (2, 1)
    assert out.score.shape == (2, 1)
    assert out.embedding.shape[0] == 2
    assert out.embedding.shape[1] == model.out_dim
    assert out.label.shape == (2, 1)


def test_level1_trainer_train_and_eval():
    graphs = make_dataset(n_graphs=10, in_dim=16, struct_dim=4)
    loader = DataLoader(graphs, batch_size=4, shuffle=True)

    model_cfg = Level1ModelConfig(
        in_dim=16,
        hidden_dim=16,
        num_layers=2,
        dropout=0.1,
        readout="meanmax",
        struct_dim=4,
        struct_hidden_dim=8,
    )
    model = Level1Model(model_cfg)

    trainer_cfg = Level1TrainerConfig(
        lr=1e-3,
        weight_decay=1e-4,
        epochs=1,
        batch_size=4,
        grad_accum_steps=1,
        use_amp=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer_cfg.lr)
    trainer = Level1Trainer(
        model=model,
        optimizer=optimizer,
        cfg=trainer_cfg,
        device="cpu",
    )

    train_metrics = trainer.train_one_epoch(loader)
    eval_metrics = trainer.evaluate(loader)

    assert "loss" in train_metrics
    assert train_metrics["loss"] >= 0.0

    for key in ["loss", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert key in eval_metrics


def test_train_level1_pipeline_saves_checkpoints(tmp_path: Path):
    train_graphs = make_dataset(n_graphs=12, in_dim=16, struct_dim=4)
    valid_graphs = make_dataset(n_graphs=6, in_dim=16, struct_dim=4)

    output_dir = tmp_path / "artifacts" / "level1"
    result = run_training(
        train_graphs=train_graphs,
        valid_graphs=valid_graphs,
        output_dir=str(output_dir),
        model_cfg=Level1ModelConfig(
            in_dim=16,
            hidden_dim=16,
            num_layers=2,
            dropout=0.1,
            readout="meanmax",
            struct_dim=4,
            struct_hidden_dim=8,
        ),
        trainer_cfg=Level1TrainerConfig(
            lr=1e-3,
            weight_decay=1e-4,
            epochs=2,
            batch_size=4,
            grad_accum_steps=1,
            use_amp=False,
        ),
        device="cpu",
    )

    best_ckpt = Path(result["best_checkpoint"])
    last_ckpt = Path(result["last_checkpoint"])

    assert best_ckpt.exists()
    assert last_ckpt.exists()
    assert len(result["history"]) == 2


def test_export_level1_embeddings_creates_bundle(tmp_path: Path):
    train_graphs = make_dataset(n_graphs=8, in_dim=16, struct_dim=4)
    valid_graphs = make_dataset(n_graphs=4, in_dim=16, struct_dim=4)

    train_data_path = tmp_path / "train_graphs.pt"
    valid_data_path = tmp_path / "valid_graphs.pt"
    torch.save(train_graphs, train_data_path)
    torch.save(valid_graphs, valid_data_path)

    output_dir = tmp_path / "artifacts" / "level1"
    result = run_training(
        train_graphs=train_graphs,
        valid_graphs=valid_graphs,
        output_dir=str(output_dir),
        model_cfg=Level1ModelConfig(
            in_dim=16,
            hidden_dim=16,
            num_layers=2,
            dropout=0.1,
            readout="meanmax",
            struct_dim=4,
            struct_hidden_dim=8,
        ),
        trainer_cfg=Level1TrainerConfig(
            lr=1e-3,
            weight_decay=1e-4,
            epochs=1,
            batch_size=4,
            grad_accum_steps=1,
            use_amp=False,
        ),
        device="cpu",
    )

    export_path = tmp_path / "artifacts" / "embeddings" / "level1_bundle.pt"
    bundle = export_level1_embeddings(
        data_path=str(valid_data_path),
        checkpoint_path=result["best_checkpoint"],
        output_path=str(export_path),
        batch_size=2,
        device="cpu",
    )

    assert export_path.exists()
    assert bundle["graph_id"].shape[0] == len(valid_graphs)
    assert bundle["embedding"].shape[0] == len(valid_graphs)
    assert bundle["score"].shape == (len(valid_graphs), 1)
    assert bundle["logits"].shape == (len(valid_graphs), 1)
    assert bundle["label"].shape == (len(valid_graphs), 1)
    assert "metadata" in bundle
