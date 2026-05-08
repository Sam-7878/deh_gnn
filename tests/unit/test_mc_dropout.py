import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass

from gog_fraud.models.extensions.mc.config import MCDropoutConfig
from gog_fraud.models.extensions.mc.interfaces import MCOutput
from gog_fraud.models.extensions.mc.mc_dropout import MCDropoutEstimator
from gog_fraud.models.extensions.mc.utils import patch_dropout

@dataclass
class DummyOutput:
    score: torch.Tensor
    aux: dict

class DummyGNNBase(nn.Module):
    def __init__(self, with_dropout=True):
        super().__init__()
        self.with_dropout = with_dropout
        self.fc = nn.Linear(10, 10)
        self.dropout = nn.Dropout(p=0.0) if with_dropout else nn.Identity()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        h = self.fc(x)
        if self.with_dropout:
            h = self.dropout(h)
        h = self.fc2(h)
        return DummyOutput(score=torch.sigmoid(h), aux={})

def test_mc_dropout_restores_state():
    model = DummyGNNBase(with_dropout=True)
    model.eval()

    dropout_layer = model.dropout
    assert dropout_layer.p == 0.0
    assert not dropout_layer.training

    with patch_dropout(model, target_p=0.5):
        assert dropout_layer.p == 0.5
        assert dropout_layer.training

    # Should be back to pure eval
    assert dropout_layer.p == 0.0
    assert not dropout_layer.training

def test_mc_dropout_estimator_basic():
    config = MCDropoutConfig(mc_samples=5, dropout_p=0.2, execution_mode="sequential")
    estimator = MCDropoutEstimator(config)
    model = DummyGNNBase()
    
    # We need randomness to trigger variance, meaning the linear layer must be untrained / have weights
    x = torch.randn(4, 10)
    
    mc_out = estimator.estimate(model, x)
    
    assert isinstance(mc_out, MCOutput)
    assert mc_out.mean_score.shape == (4, 1)
    assert mc_out.uncertainty.shape == (4, 1)
    
    # If dropout was active and ran multiple times over untrained weights, variance > 0
    # Floating point precision might make it exactly zero occasionally, but std() across 5 iterations of 0.2 dropout over 10 elements is extremely unlikely to perfectly flat out.
    assert (mc_out.uncertainty > 0).any()

def test_mc_dropout_inject_aux():
    config = MCDropoutConfig(mc_samples=3, dropout_p=0.1, inject_into_aux=True)
    estimator = MCDropoutEstimator(config)
    
    model = DummyGNNBase()
    x = torch.randn(2, 10)
    
    mc_out = estimator.estimate(model, x)
    
    assert "mc_mean_score" in mc_out.base_output.aux
    assert "mc_uncertainty" in mc_out.base_output.aux
    assert torch.allclose(mc_out.mean_score, mc_out.base_output.aux["mc_mean_score"])
    
if __name__ == "__main__":
    pytest.main([__file__])
