"""Tests of the GeminiFusion host adapter (dat/hosts/geminifusion.py).

    CUDA_VISIBLE_DEVICES=1 python -m pytest tests/test_host_geminifusion.py -q

Model tests skip cleanly when repos/GeminiFusion (scripts/setup_geminifusion.sh), its dependencies (timm,
mmcv>=2 for ConvModule) or a GPU are missing. The checkpoint / data tests additionally need the released NYU
weights (GEMINIFUSION_CKPT_DIR) and the DFormer-layout NYU dataset (RGBD_DATA_ROOT or /data/datasets/rgbd).
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dat import paths  # noqa: E402
from dat.availability import AvailabilitySampler, Mixture  # noqa: E402
from dat.hosts import geminifusion as gf  # noqa: E402
from dat.hosts.base import get_host  # noqa: E402

CKPT_DIR = os.environ.get(
    "GEMINIFUSION_CKPT_DIR",
    "/media/lingxufeng/de78661b-7aa5-47d2-a396-313197e9aa60/home/lingxufeng/paper/rgbd-availability/"
    "weights/GeminiFusion/NYUDepthv2")
NYU_ROOT = next((p for p in (os.path.join(paths.DATA_ROOT, "NYUDepthv2"), "/data/datasets/rgbd/NYUDepthv2")
                 if os.path.isfile(os.path.join(p, "test.txt"))), None)
DEVICE = os.environ.get("DAT_TEST_DEVICE", "cuda")  # run under CUDA_VISIBLE_DEVICES=1 to use GPU 1


def _need_model_stack():
    if not os.path.isfile(os.path.join(paths.GEMINIFUSION_ROOT, "models", "segformer.py")):
        pytest.skip("repos/GeminiFusion missing: run scripts/setup_geminifusion.sh")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    try:
        import timm  # noqa: F401
        from mmcv.cnn import ConvModule  # noqa: F401
    except ImportError as e:
        pytest.skip(f"missing dependency: {e}")


@pytest.fixture(scope="module")
def b3():
    _need_model_stack()
    host = get_host("GeminiFusion-B3")
    model = host.build(40, DEVICE)
    yield host, model
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- no repo / GPU needed
def test_host_metadata_and_registry():
    host = get_host("GeminiFusion-B3")
    assert host.family == "geminifusion" and host.name == "GeminiFusion-B3"
    assert host.supports_rgb_only is True and host.single_stream_mode is False  # constant-depth RGB-only; no 1-stream net
    assert host.empty_value == 0.0  # depth is /255 only and padded with 0 upstream
    assert get_host("GeminiFusion-B5").backbone == "mit_b5"
    with pytest.raises(KeyError):
        gf.make_host("GeminiFusion-B1")


def test_load_weights_is_strict_about_keys(tmp_path):
    host = gf.make_host("GeminiFusion-B3")
    net = nn.Sequential(nn.Linear(2, 2))
    torch.save({"segmenter": {"module." + k: v for k, v in net.state_dict().items()}}, tmp_path / "ok.pth")
    host.load_weights(net, str(tmp_path / "ok.pth"))  # DDP prefix stripped
    assert host.last_load_report["missing"] == 0

    sd = {k: v for k, v in net.state_dict().items() if k != "0.bias"}
    torch.save({"segmenter": sd}, tmp_path / "missing.pth")
    with pytest.raises(RuntimeError, match="missing"):
        host.load_weights(net, str(tmp_path / "missing.pth"))

    torch.save({"segmenter": {**net.state_dict(), "extra.weight": torch.zeros(1)}}, tmp_path / "extra.pth")
    with pytest.raises(RuntimeError, match="unexpected"):
        host.load_weights(net, str(tmp_path / "extra.pth"))

    # the one tolerated leftover family is the TokenFusion score_predictor, and only that
    torch.save({"segmenter": {**net.state_dict(), "encoder.score_predictor.0.w": torch.zeros(1)}},
               tmp_path / "tf.pth")
    host.load_weights(net, str(tmp_path / "tf.pth"))
    assert host.last_load_report["ignored_known_unused"] == 1


# ---------------------------------------------------------------- random-weight B3 on GPU
def test_forward_shape_and_dat_training_step(b3):
    host, model = b3
    rgb = torch.randn(2, 3, 480, 640, device=DEVICE)
    depth = torch.rand(2, 3, 480, 640, device=DEVICE)  # upstream: grey depth / 255, tiled to 3 channels

    model.eval()
    with torch.no_grad():
        assert host.forward(model, rgb, depth).shape == (2, 40, 480, 640)
        assert host.forward(model, rgb, depth[:, :1]).shape == (2, 40, 480, 640)  # 1-channel depth accepted

    model.train()
    sampler = AvailabilitySampler(c=host.empty_value, mixture=Mixture(0.0, 0.5, 0.5), seed=0)
    d_tilde, conds = sampler(depth)
    assert set(conds) <= {"missing", "partial"}
    assert torch.all(d_tilde[[i for i, c in enumerate(conds) if c == "missing"]] == host.empty_value)
    label = torch.randint(0, 40, (2, 480, 640), device=DEVICE)
    label[:, :20] = 255  # ignored pixels
    loss = host.loss(model, rgb, d_tilde, label)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.alpha.grad is not None  # the detached-ensemble term trains alpha
    model.zero_grad(set_to_none=True)


# ---------------------------------------------------------------- released NYU checkpoints
@pytest.mark.parametrize("name,fname,ignored", [("GeminiFusion-B3", "mit-b3.pth.tar", 80),
                                                ("GeminiFusion-B5", "mit_b5.pth.tar", 0)])
def test_released_nyu_checkpoint_loads_with_strict_accounting(name, fname, ignored):
    _need_model_stack()
    path = os.path.join(CKPT_DIR, fname)
    if not os.path.isfile(path):
        pytest.skip(f"no checkpoint at {path}")
    host = get_host(name)
    model = host.build(40, "cpu")
    host.load_weights(model, path)
    rep = host.last_load_report
    assert rep["missing"] == 0 and rep["unexpected"] == 0
    assert rep["loaded"] == len(model.state_dict())  # every model tensor came from the file
    assert rep["ignored_known_unused"] == ignored  # B3 file: 80 dead TokenFusion score_predictor tensors
    assert torch.allclose(model.alpha.detach(), torch.ones(2))  # released alpha


def _load_nyu(line):
    from PIL import Image
    rgb_p, lab_p = line
    rgb = np.array(Image.open(os.path.join(NYU_ROOT, rgb_p)).convert("RGB"))
    d = np.array(Image.open(os.path.join(NYU_ROOT, rgb_p.replace("RGB/", "Depth/").rsplit(".", 1)[0] + ".png")))
    d = d[..., 0] if d.ndim == 3 else d  # grey depth replicated across channels (+ alpha)
    lab = np.array(Image.open(os.path.join(NYU_ROOT, lab_p))).astype(np.int64) - 1  # 0 = unlabeled -> -1
    return rgb, d, lab


def test_released_b3_is_sensible_with_real_depth_and_worse_with_constant_depth():
    """Evidence that the adapter's preprocessing matches upstream's: with the released NYU B3 weights, 50 test
    images give mIoU far above chance (2.5%), and replacing depth by the constant c lowers it."""
    _need_model_stack()
    path = os.path.join(CKPT_DIR, "mit-b3.pth.tar")
    if NYU_ROOT is None or not os.path.isfile(path):
        pytest.skip("NYU dataset or released B3 checkpoint not available")
    host = get_host("GeminiFusion-B3")
    model = host.build(40, DEVICE)
    host.load_weights(model, path)
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    lines = [l.split() for l in open(os.path.join(NYU_ROOT, "test.txt"))]
    conf = {k: np.zeros((40, 40), np.int64) for k in ("real", "const")}
    with torch.no_grad():
        for i in np.linspace(0, len(lines) - 1, 50).astype(int):
            rgb, d, lab = _load_nyu(lines[i])
            x = (torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(DEVICE) / 255 - mean) / std
            d3 = (torch.from_numpy(d).float()[None, None].to(DEVICE) / 255).expand(-1, 3, -1, -1)
            ok = lab >= 0
            for k, dep in (("real", d3), ("const", torch.full_like(d3, host.empty_value))):
                pred = host.forward(model, x, dep).argmax(1)[0].cpu().numpy()
                conf[k] += np.bincount(40 * lab[ok] + pred[ok], minlength=1600).reshape(40, 40)

    def miou(c):
        iou = np.diag(c) / (c.sum(0) + c.sum(1) - np.diag(c) + 1e-9)
        return float(iou[c.sum(1) > 0].mean())

    real, const = miou(conf["real"]), miou(conf["const"])
    print(f"\nGeminiFusion-B3 released NYU weights, 50 test images: mIoU real depth {100 * real:.2f}, "
          f"constant depth {100 * const:.2f}")
    assert real > 0.35, "released weights should be far above chance (2.5%) if preprocessing is right"
    assert real > const
