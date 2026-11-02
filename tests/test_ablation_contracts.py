from dataclasses import replace

import pytest
import torch

from src.config import cfg
from src.model import build_model


def config(mode):
    return replace(cfg, MODALITY_MODE=mode, FUSION_DIM=8, CNN_CHANNELS=(8,),
                   TEMP_EMBED_DIM=8, ATTENTION_HEADS=2, RNN_HIDDEN_DIM=8,
                   RNN_LAYERS=1, FC_HIDDEN_DIM=8)


@pytest.mark.parametrize("mode", ["wca_gru", "concat", "vib_only", "temp_only"])
def test_variants_have_finite_forward_backward_and_declared_attention(mode):
    model = build_model(config(mode))
    vib, temp = torch.randn(2, 4, cfg.VIB_FEATURE_DIM), torch.randn(2, 4, cfg.TEMP_FEATURE_DIM)
    output, attention = model(vib, temp, return_attention=True)
    assert output.shape == (2,) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert (attention is not None) == (mode == "wca_gru")


@pytest.mark.parametrize("mode,absent", [("vib_only", "temp"), ("temp_only", "vib")])
def test_single_modality_does_not_consume_other_input(mode, absent):
    model = build_model(config(mode)).eval()
    vib, temp = torch.randn(2, 4, cfg.VIB_FEATURE_DIM), torch.randn(2, 4, cfg.TEMP_FEATURE_DIM)
    with torch.no_grad():
        before = model(vib, temp)
        after = model(vib, temp + 100) if absent == "temp" else model(vib + 100, temp)
    torch.testing.assert_close(before, after)
    assert model.fusion is None
    assert getattr(model, absent + "_encoder") is None


def test_concat_preserves_both_feature_blocks():
    model = build_model(config("concat")).eval()
    vib, temp = torch.randn(2, 4, cfg.VIB_FEATURE_DIM), torch.randn(2, 4, cfg.TEMP_FEATURE_DIM)
    inputs = []
    hook = model.temporal.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
    with torch.no_grad():
        model(vib, temp)
        torch.testing.assert_close(inputs[0][..., :8], model.vib_encoder(vib))
        torch.testing.assert_close(inputs[0][..., 8:], model.temp_encoder(temp))
    hook.remove()
    assert model.temporal.rnn.input_size == 16


def test_rejects_unknown_ablation():
    with pytest.raises(ValueError, match="Unsupported"):
        build_model(config("unknown"))
