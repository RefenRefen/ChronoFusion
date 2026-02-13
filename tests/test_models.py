import sys
from pathlib import Path

import pytest
import torch

# Ensure local package import works even if not installed into the venv.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chronofusion.models.transformer import (
    MultiScaleTransformer,
    PositionalEncoding,
    SingleScaleEncoder,
)


def _rand(batch: int, seq: int, dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randn(batch, seq, dim, generator=g)


def test_positional_encoding_shape_and_no_nan():
    pe = PositionalEncoding(d_model=16, max_len=64)
    x = _rand(batch=2, seq=10, dim=16, seed=1)

    y = pe(x)

    assert y.shape == (2, 10, 16)
    assert torch.isfinite(y).all()


def test_single_scale_encoder_forward_shape():
    model = SingleScaleEncoder(
        input_dim=7,
        d_model=16,
        nhead=4,
        num_layers=2,
        dropout=0.0,
    ).eval()

    x = _rand(batch=3, seq=11, dim=7, seed=2)
    y = model(x)

    assert y.shape == (3, 11, 16)
    assert torch.isfinite(y).all()


def test_single_scale_encoder_is_causal_no_future_leakage():
    """
    The encoder builds an explicit causal attention mask. If we change only a
    future token (position j), earlier outputs (positions < j) should not change.
    """
    torch.manual_seed(0)
    model = SingleScaleEncoder(
        input_dim=5,
        d_model=20,
        nhead=4,
        num_layers=2,
        dropout=0.0,
    ).eval()

    batch, seq, dim = 2, 12, 5
    x = _rand(batch=batch, seq=seq, dim=dim, seed=3)
    j = 9  # "future" position we will perturb

    y1 = model(x)

    x2 = x.clone()
    x2[:, j, :] = x2[:, j, :] + 100.0  # large perturbation at a single future timestep
    y2 = model(x2)

    # Earlier timesteps must match (within numerical tolerance).
    assert torch.allclose(y1[:, :j, :], y2[:, :j, :], atol=1e-5, rtol=1e-5)
    # The perturbed timestep (and potentially later ones) should change.
    assert not torch.allclose(y1[:, j:, :], y2[:, j:, :], atol=1e-5, rtol=1e-5)


def test_multiscale_transformer_forward_shape_and_gradients():
    torch.manual_seed(0)
    model = MultiScaleTransformer(
        input_dim_low=6,
        input_dim_high=4,
        d_model=24,
        nhead=4,
        num_layers=2,
        output_dim=13,
    ).train()

    x_low = _rand(batch=2, seq=8, dim=6, seed=4)
    x_high = _rand(batch=2, seq=8, dim=4, seed=5)

    out = model(x_low, x_high)
    assert out.shape == (2, 8, 13)
    assert torch.isfinite(out).all()

    loss = out.mean()
    loss.backward()

    # Ensure we have gradients flowing to key parameters
    assert model.fusion_proj.weight.grad is not None
    assert torch.isfinite(model.fusion_proj.weight.grad).all()


def test_invalid_head_dim_raises():
    # torch requires d_model divisible by nhead for multihead attention
    with pytest.raises(Exception):
        _ = SingleScaleEncoder(input_dim=3, d_model=10, nhead=4, num_layers=1)
