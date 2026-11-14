"""CMNeXt host tests. Skipped when repos/DELIVER (scripts/setup_cmnext.sh) or its pure-python deps are missing.

    CUDA_VISIBLE_DEVICES=1 python -m pytest tests/test_host_cmnext.py -q
    CMNEXT_NYU_CKPT=/path/cmnext_b4_nyu_rgbd.pth  enables the released-weights test.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dat import paths  # noqa: E402
from dat.availability import AvailabilitySampler  # noqa: E402
from dat.hosts.base import get_host  # noqa: E402

if not os.path.isdir(os.path.join(paths.CMNEXT_ROOT, "semseg")):
    pytest.skip("repos/DELIVER missing; run scripts/setup_cmnext.sh", allow_module_level=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = os.environ.get("CMNEXT_NYU_CKPT", os.path.join(paths.ROOT, "checkpoints", "cmnext_b4_nyu_rgbd.pth"))


def _build(name, device, **kw):
    host = get_host(name)
    try:
        return host, host.build(num_classes=40, device=device, **kw)
    except ImportError as e:  # fvcore / tabulate / timm not installed
        pytest.skip(f"CMNeXt dependencies missing: {e}")


@pytest.fixture(scope="module")
def b4():
    return _build("CMNeXt-B4", DEVICE)


def test_registry_and_conventions():
    host = get_host("CMNeXt-B4")
    assert host.name == "CMNeXt-B4" and host.family == "cmnext"
    assert host.empty_value == 0.0 and host.supports_rgb_only
    with pytest.raises(KeyError):
        get_host("CMNeXt-B0")  # upstream defines B2/B4/B5 only


def test_forward_shape_at_input_resolution(b4):
    host, model = b4
    rgb, depth = torch.randn(2, 3, 480, 640, device=DEVICE), torch.rand(2, 3, 480, 640, device=DEVICE)
    model.eval()
    with torch.no_grad():
        assert host.forward(model, rgb, depth).shape == (2, 40, 480, 640)


def test_one_dat_step(b4):
    host, model = b4
    model.train()
    rgb, depth = torch.randn(2, 3, 480, 640, device=DEVICE), torch.rand(2, 3, 480, 640, device=DEVICE)
    label = torch.randint(0, 40, (2, 480, 640), device=DEVICE)
    depth_tilde, conds = AvailabilitySampler(c=host.empty_value, seed=0)(depth)
    assert depth_tilde.shape == depth.shape and len(conds) == 2
    loss = host.loss(model, rgb, depth_tilde, label)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_rgb_only_model_builds_and_runs():
    host, model = _build("CMNeXt-B2", "cpu", rgb_only=True)  # modals=['img'] -> num_modals == 0
    assert model.backbone.num_modals == 0
    rgb = torch.randn(1, 3, 128, 160)
    model.eval()
    with torch.no_grad():  # depth is ignored by the single-modality network
        assert torch.equal(host.forward(model, rgb, torch.zeros(1, 3, 128, 160)),
                           host.forward(model, rgb, torch.ones(1, 3, 128, 160)))
        assert host.forward(model, rgb, None).shape == (1, 40, 128, 160)


def test_load_weights_is_strict(tmp_path):
    host, model = _build("CMNeXt-B2", "cpu")
    state = model.state_dict()
    good = tmp_path / "good.pth"
    torch.save({"model_state_dict": {"module." + k: v for k, v in state.items()}}, good)  # train_mm + DDP format
    host.load_weights(model, str(good))
    k = next(iter(state))
    bad_missing, bad_extra = tmp_path / "missing.pth", tmp_path / "extra.pth"
    torch.save({n: v for n, v in state.items() if n != k}, bad_missing)
    torch.save({**state, "bogus.weight": torch.zeros(1)}, bad_extra)
    for bad in (bad_missing, bad_extra):
        with pytest.raises(RuntimeError):
            host.load_weights(model, str(bad))


@pytest.mark.skipif(not os.path.isfile(CKPT), reason="no released NYU checkpoint (set CMNEXT_NYU_CKPT)")
def test_load_released_nyu_checkpoint(b4):
    host, model = b4
    host.load_weights(model, CKPT)  # raises on any missing / unexpected key
    model.eval()
    rgb, depth = torch.randn(1, 3, 480, 640, device=DEVICE), torch.rand(1, 3, 480, 640, device=DEVICE)
    with torch.no_grad():
        assert torch.isfinite(host.forward(model, rgb, depth)).all()
