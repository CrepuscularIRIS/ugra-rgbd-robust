"""DFormerv1/v2 host adapter: builds all seven models, runs a DAT step, and checks the paper's Sec. 3.1 claim that a
constant depth map removes the depth term of the DFormerv2 geometry prior.

Needs the upstream DFormer code ($DFORMER_ROOT, default repos/DFormer) and its pure-python deps; skipped otherwise.
Weights/data tests additionally need $DFORMERV2_WEIGHTS / $RGBD_DATA_ROOT and are skipped when absent.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dat import paths  # noqa: E402

if not os.path.exists(os.path.join(paths.DFORMER_ROOT, "models", "builder.py")):
    pytest.skip(f"upstream DFormer not found at {paths.DFORMER_ROOT} (run scripts/setup_dformer.sh)",
                allow_module_level=True)
paths.activate(paths.DFORMER_ROOT)
try:
    from dat.hosts import dformer
except Exception as e:  # pragma: no cover
    pytest.skip(f"cannot import the DFormer host: {e}", allow_module_level=True)
from dat.availability import AvailabilitySampler  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = sorted(dformer._MODELS)


def _inputs(b=1, h=480, w=640):
    g = torch.Generator().manual_seed(0)
    return torch.randn(b, 3, h, w, generator=g).to(DEV), torch.randn(b, 3, h, w, generator=g).to(DEV)


@pytest.mark.parametrize("name", MODELS)
def test_builds_and_outputs_one_logit_map_per_class(name):
    host = dformer.make_host(name)
    model = host.build(device=DEV).eval()
    rgb, depth = _inputs()
    with torch.no_grad():
        out = host.forward(model, rgb, depth)
    assert out.shape == (1, 40, 480, 640) and torch.isfinite(out).all()
    assert host.empty_value == 0.0 and host.supports_rgb_only


def test_sun_rgbd_has_37_classes():
    assert dformer.make_host("DFormerv2-S", "SUNRGBD").num_classes == 37


@pytest.mark.parametrize("name", ["DFormerv1-T", "DFormerv2-S"])
def test_one_dat_step_is_finite_and_reaches_all_parameters(name):
    host = dformer.make_host(name)
    model = host.build(device=DEV).train()
    rgb, depth = _inputs(2)
    label = torch.randint(0, 40, (2, 480, 640)).to(DEV)
    depth_in, conds = AvailabilitySampler(c=host.empty_value, seed=3)(depth)
    loss = host.loss(model, rgb, depth_in, label)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0
    got = sum(p.grad is not None for p in model.parameters())
    assert got > 0.9 * sum(1 for _ in model.parameters())


def test_dformerv2_constant_depth_gives_the_same_features_whatever_the_constant():
    """Sec. 3.1: a constant depth map makes the depth term of the geometry prior vanish, so the value is irrelevant."""
    host = dformer.make_host("DFormerv2-S")
    model = host.build(device=DEV).eval()
    rgb, _ = _inputs()
    with torch.no_grad():
        a = model.backbone(rgb, torch.zeros(1, 3, 480, 640, device=DEV))
        b = model.backbone(rgb, torch.full((1, 3, 480, 640), 0.7, device=DEV))
        real = model.backbone(rgb, torch.randn(1, 3, 480, 640, device=DEV))
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-5)
    assert not torch.allclose(a[0], real[0], atol=1e-3), "real depth must change the features"


def test_upstream_schedule_is_read_from_the_config():
    s = dformer.make_host("DFormerv2-S").schedule()
    assert s["batch_size"] == 16 and s["lr"] == 6e-5 and s["epochs"] == 500 and s["num_train"] == 795
    assert dformer.make_host("DFormerv1-T").schedule()["batch_size"] == 8


CKPT = os.path.join(os.environ.get("DFORMERV2_WEIGHTS", ""), "NYU", "DFormerv2_Small_NYU.pth")


@pytest.mark.skipif(not os.path.exists(CKPT), reason="released DFormerv2-S NYU checkpoint not available")
def test_released_checkpoint_loads_with_no_missing_or_unexpected_keys():
    host = dformer.make_host("DFormerv2-S")
    model = host.build(device=DEV)
    host.load_weights(model, CKPT)  # raises on any missing/unexpected key


CKPT_V1 = os.path.join(os.environ.get("DFORMERV1_WEIGHTS", ""), "NYUDepthv2", "NYUv2_DFormer_Tiny.pth")


@pytest.mark.skipif(not os.path.exists(CKPT_V1), reason="released DFormerv1-T NYU checkpoint not available")
def test_dformerv1_checkpoint_loads_and_only_the_four_unused_norms_are_tolerated():
    host = dformer.make_host("DFormerv1-T")
    model = host.build(device=DEV)
    host.load_weights(model, CKPT_V1)  # 0 missing; backbone.norm0-3 are stored but unused
    state = torch.load(CKPT_V1, map_location="cpu", weights_only=False)["model"]
    state.pop(next(k for k in state if k.startswith("backbone.layers") or k.startswith("backbone.downsample")), None)
    torch.save({"model": state}, "/tmp/_v1_missing_key.pth")
    try:
        with pytest.raises(RuntimeError):
            host.load_weights(model, "/tmp/_v1_missing_key.pth")  # a genuinely missing key must still fail
    finally:
        os.remove("/tmp/_v1_missing_key.pth")
