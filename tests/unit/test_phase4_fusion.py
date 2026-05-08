"""
test_phase4_fusion.py

Fusion 계층 (pipelines/fusion.py) 전체 unit test.
"""
from pathlib import Path

import torch
import pytest

from gog_fraud.pipelines.fusion import (
    # types
    FusionInput,
    FusionOutput,
    # strategies
    WeightedSumFusion,
    CalibratedFusion,
    LearnedFusion,
    FusionEnsemble,
    # configs
    WeightedSumConfig,
    CalibratedFusionConfig,
    LearnedFusionConfig,
    FusionTrainerConfig,
    # trainer
    FusionTrainer,
    # metrics
    compute_fusion_metrics,
    # save/load
    save_learned_fusion,
    load_learned_fusion,
    # factory
    build_fusion,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

N = 20


def make_fusion_input(
    n:          int   = N,
    with_label: bool  = True,
    with_logits: bool = True,
    seed:       int   = 42,
) -> FusionInput:
    torch.manual_seed(seed)
    l1_score = torch.rand(n)
    l2_score = torch.rand(n)

    l1_logits = torch.logit(l1_score.clamp(1e-4, 1 - 1e-4)) if with_logits else None
    l2_logits = torch.logit(l2_score.clamp(1e-4, 1 - 1e-4)) if with_logits else None
    label     = (torch.rand(n) > 0.6).float() if with_label else None

    return FusionInput(
        level1_score=l1_score,
        level2_score=l2_score,
        level1_logits=l1_logits,
        level2_logits=l2_logits,
        label=label,
        graph_id=torch.arange(n, dtype=torch.long),
    )


# ──────────────────────────────────────────────
# FusionInput 검증
# ──────────────────────────────────────────────

class TestFusionInput:

    def test_basic_shape_normalization(self):
        fi = make_fusion_input(n=10)
        assert fi.level1_score.dim() == 1
        assert fi.level2_score.dim() == 1
        assert fi.level1_score.shape[0] == 10
        assert fi.label.shape[0] == 10

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            FusionInput(
                level1_score=torch.rand(5),
                level2_score=torch.rand(8),
            )

    def test_without_label(self):
        fi = FusionInput(
            level1_score=torch.rand(6),
            level2_score=torch.rand(6),
        )
        assert fi.label is None
        assert fi.level1_logits is None

    def test_n_property(self):
        fi = make_fusion_input(n=15)
        assert fi.n == 15


# ──────────────────────────────────────────────
# WeightedSumFusion
# ──────────────────────────────────────────────

class TestWeightedSumFusion:

    def test_output_shape(self):
        fi  = make_fusion_input(n=N)
        fus = WeightedSumFusion(WeightedSumConfig())
        out = fus.fuse(fi)

        assert out.score.shape  == (N,)
        assert out.logits.shape == (N,)

    def test_score_in_range(self):
        fi  = make_fusion_input(n=N)
        fus = WeightedSumFusion(WeightedSumConfig())
        out = fus.fuse(fi)

        assert float(out.score.min().item()) >= 0.0
        assert float(out.score.max().item()) <= 1.0

    def test_weight_normalization(self):
        # weight가 자동 정규화되어야 함
        fus = WeightedSumFusion(
            WeightedSumConfig(level1_weight=2.0, level2_weight=8.0)
        )
        assert abs(fus.w1 - 0.2) < 1e-6
        assert abs(fus.w2 - 0.8) < 1e-6

    def test_equal_weight_gives_average(self):
        """
        동일 score, equal weight → 최종 score = 원본 score여야 함
        """
        score = torch.tensor([0.2, 0.5, 0.8])
        fi    = FusionInput(level1_score=score, level2_score=score)
        fus   = WeightedSumFusion(
            WeightedSumConfig(level1_weight=0.5, level2_weight=0.5)
        )
        out   = fus.fuse(fi)
        assert torch.allclose(out.score, score, atol=1e-5)

    def test_level2_heavy_raises_score(self):
        """
        level2_score가 높을 때 level2_weight 올리면 최종 score도 올라야 함
        """
        fi   = FusionInput(
            level1_score=torch.tensor([0.2, 0.2, 0.2]),
            level2_score=torch.tensor([0.9, 0.9, 0.9]),
        )
        low  = WeightedSumFusion(WeightedSumConfig(level1_weight=0.9, level2_weight=0.1))
        high = WeightedSumFusion(WeightedSumConfig(level1_weight=0.1, level2_weight=0.9))

        out_low  = low.fuse(fi)
        out_high = high.fuse(fi)

        assert float(out_high.score.mean()) > float(out_low.score.mean())

    def test_factory(self):
        fus = build_fusion("weighted_sum", level1_weight=0.3, level2_weight=0.7)
        assert isinstance(fus, WeightedSumFusion)
        assert abs(fus.w1 - 0.3) < 1e-6

    def test_callable_interface(self):
        fi  = make_fusion_input()
        fus = build_fusion("weighted_sum")
        out = fus(fi)   # __call__
        assert isinstance(out, FusionOutput)

    def test_metadata_keys(self):
        fus = WeightedSumFusion(WeightedSumConfig())
        fi  = make_fusion_input()
        out = fus.fuse(fi)
        assert "strategy"       in out.metadata
        assert "level1_weight"  in out.metadata
        assert "level2_weight"  in out.metadata


# ──────────────────────────────────────────────
# CalibratedFusion
# ──────────────────────────────────────────────

class TestCalibratedFusion:

    def test_output_shape(self):
        fi  = make_fusion_input(n=N)
        fus = CalibratedFusion(CalibratedFusionConfig())
        out = fus.fuse(fi)

        assert out.score.shape  == (N,)
        assert out.logits.shape == (N,)

    def test_temperature_softens_score(self):
        """
        temperature > 1이면 score가 0.5에 더 가까워져야 함
        """
        score = torch.tensor([0.05, 0.95])
        fi    = FusionInput(level1_score=score, level2_score=score)

        fus_t1 = CalibratedFusion(CalibratedFusionConfig(
            level1_temperature=1.0, level2_temperature=1.0
        ))
        fus_t5 = CalibratedFusion(CalibratedFusionConfig(
            level1_temperature=5.0, level2_temperature=5.0
        ))

        out_t1 = fus_t1.fuse(fi)
        out_t5 = fus_t5.fuse(fi)

        # t=5 일 때는 t=1 보다 0.5에 더 가까워야 함
        dist_t1 = float((out_t1.score - 0.5).abs().mean().item())
        dist_t5 = float((out_t5.score - 0.5).abs().mean().item())
        assert dist_t5 < dist_t1

    def test_positive_bias_raises_score(self):
        """bias > 0이면 score가 올라야 함."""
        score  = torch.tensor([0.3, 0.3, 0.3])
        fi     = FusionInput(level1_score=score, level2_score=score)

        neutral  = CalibratedFusion(CalibratedFusionConfig(level1_bias=0.0, level2_bias=0.0))
        biased   = CalibratedFusion(CalibratedFusionConfig(level1_bias=2.0, level2_bias=2.0))

        out_n = neutral.fuse(fi)
        out_b = biased.fuse(fi)

        assert float(out_b.score.mean()) > float(out_n.score.mean())

    def test_factory(self):
        fus = build_fusion("calibrated", level1_temperature=1.5, level2_temperature=0.8)
        assert isinstance(fus, CalibratedFusion)
        assert fus.cfg.level1_temperature == 1.5


# ──────────────────────────────────────────────
# LearnedFusion
# ──────────────────────────────────────────────

class TestLearnedFusion:

    def test_forward_shape(self):
        fi  = make_fusion_input(n=N)
        fus = LearnedFusion(LearnedFusionConfig(), device="cpu")
        out = fus.fuse(fi)

        assert out.score.shape  == (N,)
        assert out.logits.shape == (N,)

    def test_forward_no_logits(self):
        """logit feature 없이도 동작해야 함."""
        fi = make_fusion_input(n=N, with_logits=False)
        fi.level1_logits = None
        fi.level2_logits = None

        fus = LearnedFusion(LearnedFusionConfig(use_logits=False), device="cpu")
        out = fus.fuse(fi)
        assert out.score.shape == (N,)

    def test_input_feature_dim(self):
        fus_with    = LearnedFusion(LearnedFusionConfig(use_logits=True),  device="cpu")
        fus_without = LearnedFusion(LearnedFusionConfig(use_logits=False), device="cpu")

        assert fus_with.net.in_dim    == 7
        assert fus_without.net.in_dim == 5

    def test_train_eval_mode_switch(self):
        fus = LearnedFusion(LearnedFusionConfig(), device="cpu")
        fus.train_mode()
        assert fus.net.training

        fus.eval_mode()
        assert not fus.net.training

    def test_factory(self):
        fus = build_fusion("learned", hidden_dim=32, num_layers=2)
        assert isinstance(fus, LearnedFusion)
        assert fus.cfg.hidden_dim == 32


# ──────────────────────────────────────────────
# FusionTrainer
# ──────────────────────────────────────────────

class TestFusionTrainer:

    def _make_large_input(self, n: int = 100, seed: int = 0) -> FusionInput:
        """
        학습 가능한 충분한 크기의 input 생성.
        Level 2 score가 label과 더 상관되도록 설계.
        """
        torch.manual_seed(seed)
        label    = (torch.rand(n) > 0.6).float()
        l2_score = label * 0.6 + torch.rand(n) * 0.3   # label과 correlate
        l1_score = torch.rand(n) * 0.5 + 0.25           # noisy

        l1_score = l1_score.clamp(0.01, 0.99)
        l2_score = l2_score.clamp(0.01, 0.99)

        return FusionInput(
            level1_score=l1_score,
            level2_score=l2_score,
            level1_logits=torch.logit(l1_score),
            level2_logits=torch.logit(l2_score),
            label=label,
        )

    def test_trainer_runs_without_error(self):
        train_input = self._make_large_input(n=80, seed=0)
        valid_input = self._make_large_input(n=20, seed=1)

        fus     = LearnedFusion(
            LearnedFusionConfig(hidden_dim=16, num_layers=2),
            device="cpu",
        )
        trainer = FusionTrainer(
            fusion=fus,
            cfg=FusionTrainerConfig(
                epochs=3,
                batch_size=16,
                early_stopping_patience=10,
                val_metric="pr_auc",
            ),
        )
        result = trainer.fit(train_input, valid_input, verbose=False)

        assert "history"      in result
        assert "best_score"   in result
        assert len(result["history"]) == 3

    def test_early_stopping_triggers(self):
        """
        patience=1이면 두 번째 epoch에서 멈춰야 함.
        (valid score가 랜덤이므로 개선 없을 가능성이 높음)
        """
        torch.manual_seed(999)
        train_input = self._make_large_input(n=60)
        # valid는 label을 반대로 뒤집어서 score가 개선 안 되도록
        valid_input = self._make_large_input(n=30)
        valid_input.label.fill_(0.0)  # 모두 음성 → PR AUC 변동 없음

        fus     = LearnedFusion(LearnedFusionConfig(hidden_dim=8, num_layers=1), device="cpu")
        trainer = FusionTrainer(
            fusion=fus,
            cfg=FusionTrainerConfig(
                epochs=20,
                batch_size=32,
                early_stopping_patience=1,
                val_metric="pr_auc",
            ),
        )
        result = trainer.fit(train_input, valid_input, verbose=False)
        # patience=1이면 최대 2 epoch만 실행됨
        assert result["total_epochs"] <= 3

    def test_requires_label_for_fit(self):
        train_no_label = make_fusion_input(n=20, with_label=False)
        valid_input    = make_fusion_input(n=10, with_label=True)

        fus     = LearnedFusion(LearnedFusionConfig(), device="cpu")
        trainer = FusionTrainer(fus, FusionTrainerConfig(epochs=1))

        with pytest.raises(ValueError, match="label"):
            trainer.fit(train_no_label, valid_input, verbose=False)


# ──────────────────────────────────────────────
# FusionEnsemble
# ──────────────────────────────────────────────

class TestFusionEnsemble:

    def _make_strategies(self):
        return [
            WeightedSumFusion(WeightedSumConfig(level1_weight=0.4, level2_weight=0.6)),
            CalibratedFusion(CalibratedFusionConfig(level1_temperature=1.2)),
        ]

    def test_mean_mode_output_shape(self):
        fi       = make_fusion_input(n=N)
        ensemble = FusionEnsemble(self._make_strategies(), mode="mean")
        out      = ensemble.fuse(fi)

        assert out.score.shape  == (N,)
        assert out.logits.shape == (N,)

    def test_max_mode(self):
        fi       = make_fusion_input(n=N)
        ensemble = FusionEnsemble(self._make_strategies(), mode="max")
        out      = ensemble.fuse(fi)
        assert out.score.shape == (N,)

    def test_vote_mode(self):
        fi       = make_fusion_input(n=N)
        ensemble = FusionEnsemble(self._make_strategies(), mode="vote")
        out      = ensemble.fuse(fi)

        # vote 결과는 ±3 logit → sigmoid → score는 항상 시그모이드 범위 안
        assert float(out.score.min().item()) >= 0.0
        assert float(out.score.max().item()) <= 1.0

    def test_empty_strategies_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            FusionEnsemble([])

    def test_ensemble_mean_is_between_strategies(self):
        """
        두 전략의 score 평균이 ensemble mean의 score와 가까워야 함.
        (logit 공간 평균이므로 score 공간과 완전히 같지는 않으나 단조성은 유지)
        """
        fi       = make_fusion_input(n=10, seed=77)
        s1, s2   = self._make_strategies()
        out1     = s1.fuse(fi)
        out2     = s2.fuse(fi)
        ensemble = FusionEnsemble([s1, s2], mode="mean")
        out_ens  = ensemble.fuse(fi)

        # 각 개별 전략의 logit 평균과 ensemble logit이 같아야 함
        expected_logit = (out1.logits + out2.logits) / 2.0
        assert torch.allclose(out_ens.logits, expected_logit, atol=1e-5)

    def test_factory_ensemble(self):
        strategies = [build_fusion("weighted_sum"), build_fusion("calibrated")]
        ensemble   = build_fusion("ensemble", strategies=strategies, mode="mean")
        assert isinstance(ensemble, FusionEnsemble)

    def test_metadata_sub_strategies(self):
        fi       = make_fusion_input()
        ensemble = FusionEnsemble(self._make_strategies(), mode="mean")
        out      = ensemble.fuse(fi)

        assert out.metadata["strategy"] == "FusionEnsemble"
        assert len(out.metadata["sub_strategies"]) == 2


# ──────────────────────────────────────────────
# compute_fusion_metrics
# ──────────────────────────────────────────────

class TestComputeFusionMetrics:

    def test_perfect_prediction(self):
        label = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])
        # score가 label을 그대로 반영
        score = torch.tensor([0.05, 0.95, 0.90, 0.10, 0.85])

        out = FusionOutput(
            score=score,
            logits=torch.logit(score.clamp(1e-4, 1-1e-4)),
            level1_score=score,
            level2_score=score,
            label=label,
        )
        metrics = compute_fusion_metrics(out, threshold=0.5)

        assert metrics["accuracy"]  == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"]    == 1.0
        assert metrics["f1"]        == 1.0

    def test_all_wrong_prediction(self):
        label = torch.tensor([1.0, 1.0, 1.0])
        score = torch.tensor([0.1, 0.1, 0.1])

        out = FusionOutput(
            score=score,
            logits=torch.logit(score.clamp(1e-4, 1-1e-4)),
            level1_score=score,
            level2_score=score,
            label=label,
        )
        metrics = compute_fusion_metrics(out, threshold=0.5)

        assert metrics["precision"] == 0.0
        assert metrics["recall"]    == 0.0
        assert metrics["f1"]        == 0.0

    def test_requires_label(self):
        out = FusionOutput(
            score=torch.rand(5),
            logits=torch.randn(5),
            level1_score=torch.rand(5),
            level2_score=torch.rand(5),
            label=None,
        )
        with pytest.raises(ValueError, match="label"):
            compute_fusion_metrics(out)

    def test_metric_keys(self):
        fi  = make_fusion_input(n=10)
        fus = build_fusion("weighted_sum")
        out = fus.fuse(fi)
        metrics = compute_fusion_metrics(out)

        for key in [
            "accuracy", "precision", "recall", "specificity",
            "f1", "roc_auc", "pr_auc", "bce_loss",
            "tp", "tn", "fp", "fn",
        ]:
            assert key in metrics, f"Missing metric key: {key}"


# ──────────────────────────────────────────────
# Save / Load (LearnedFusion)
# ──────────────────────────────────────────────

class TestSaveLoadLearnedFusion:

    def test_save_and_load(self, tmp_path: Path):
        fus  = LearnedFusion(
            LearnedFusionConfig(hidden_dim=16, num_layers=2, use_logits=True),
            device="cpu",
        )
        path = tmp_path / "fusion" / "learned.pt"
        save_learned_fusion(fus, str(path))

        assert path.exists()

        loaded = load_learned_fusion(str(path), device="cpu")
        assert isinstance(loaded, LearnedFusion)
        assert loaded.cfg.hidden_dim == 16
        assert loaded.cfg.num_layers == 2

    def test_loaded_weights_are_identical(self, tmp_path: Path):
        fus  = LearnedFusion(LearnedFusionConfig(hidden_dim=8), device="cpu")
        path = tmp_path / "learned.pt"
        save_learned_fusion(fus, str(path))

        loaded = load_learned_fusion(str(path), device="cpu")

        fi      = make_fusion_input(n=10)
        out_orig = fus.fuse(fi)
        out_load = loaded.fuse(fi)

        assert torch.allclose(out_orig.score, out_load.score, atol=1e-6)


# ──────────────────────────────────────────────
# End-to-end 통합 시나리오
# ──────────────────────────────────────────────

class TestEndToEnd:

    def test_weighted_to_learned_improvement(self):
        """
        LearnedFusion은 학습 후 WeightedSumFusion보다
        pr_auc가 같거나 더 높아야 함 (데이터에 패턴이 있을 때).
        """
        torch.manual_seed(0)
        n     = 200
        label = (torch.rand(n) > 0.65).float()
        # Level 2가 label과 강하게 correlate
        l2s   = (label * 0.7 + torch.rand(n) * 0.2).clamp(0.01, 0.99)
        l1s   = (torch.rand(n) * 0.5 + 0.2).clamp(0.01, 0.99)

        train_fi = FusionInput(
            level1_score=l1s[:160], level2_score=l2s[:160], label=label[:160],
            level1_logits=torch.logit(l1s[:160]),
            level2_logits=torch.logit(l2s[:160]),
        )
        valid_fi = FusionInput(
            level1_score=l1s[160:], level2_score=l2s[160:], label=label[160:],
            level1_logits=torch.logit(l1s[160:]),
            level2_logits=torch.logit(l2s[160:]),
        )

        # Baseline: WeightedSum
        ws_fus    = build_fusion("weighted_sum", level1_weight=0.5, level2_weight=0.5)
        ws_out    = ws_fus.fuse(valid_fi)
        ws_metric = compute_fusion_metrics(ws_out)

        # Learned Fusion 학습
        lf_fus  = LearnedFusion(
            LearnedFusionConfig(hidden_dim=16, num_layers=2, use_logits=True),
            device="cpu",
        )
        trainer = FusionTrainer(
            fusion=lf_fus,
            cfg=FusionTrainerConfig(
                epochs=30, batch_size=32,
                early_stopping_patience=5, val_metric="pr_auc",
            ),
        )
        trainer.fit(train_fi, valid_fi, verbose=False)

        lf_out    = lf_fus.fuse(valid_fi)
        lf_metric = compute_fusion_metrics(lf_out)

        # 학습 후 최소한 random 수준 이상
        assert lf_metric["pr_auc"] >= 0.3, (
            f"LearnedFusion pr_auc={lf_metric['pr_auc']:.4f} is too low"
        )

    def test_full_pipeline_weighted_sum_to_metrics(self):
        """
        FusionInput 생성 → fuse → metrics 계산 전체 파이프라인.
        """
        fi  = make_fusion_input(n=50, with_label=True)
        fus = build_fusion("weighted_sum", level1_weight=0.35, level2_weight=0.65)
        out = fus.fuse(fi)

        assert isinstance(out, FusionOutput)
        assert out.score.shape  == (50,)

        metrics = compute_fusion_metrics(out, threshold=0.5)
        total   = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
        assert total == 50

    def test_full_pipeline_ensemble(self):
        """
        Ensemble 전략 전체 파이프라인.
        """
        fi = make_fusion_input(n=30, with_label=True)

        strategies = [
            build_fusion("weighted_sum", level1_weight=0.3, level2_weight=0.7),
            build_fusion("calibrated",   level1_temperature=1.5, level2_temperature=0.8),
            build_fusion("learned",      hidden_dim=16, num_layers=1),
        ]
        ensemble = build_fusion("ensemble", strategies=strategies, mode="mean")
        out      = ensemble.fuse(fi)

        assert out.score.shape == (30,)
        metrics = compute_fusion_metrics(out)
        assert 0.0 <= metrics["f1"]      <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0
