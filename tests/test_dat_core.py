"""Tests of DAT's core definitions: availability sampling (Eq. 2-4, Algorithm 1) and the T/B/G metrics (Eq. 1, 5).

    python -m pytest tests/test_dat_core.py -q        (or: python tests/test_dat_core.py)
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dat import availability as av  # noqa: E402
from dat import metrics, presets  # noqa: E402


def test_mixture_validation():
    with pytest.raises(ValueError):
        av.Mixture(0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        av.Mixture(1.2, -0.2, 0.0)
    assert av.Mixture().as_tuple() == (0.50, 0.25, 0.25)


def test_condition_frequencies_match_the_mixture():
    gen = torch.Generator().manual_seed(0)
    conds = av.sample_conditions(40000, av.Mixture(0.50, 0.25, 0.25), gen)
    freq = {c: conds.count(c) / len(conds) for c in av.CONDITIONS}
    assert abs(freq[av.COMPLETE] - 0.50) < 0.01
    assert abs(freq[av.MISSING] - 0.25) < 0.01
    assert abs(freq[av.PARTIAL] - 0.25) < 0.01


def test_degenerate_mixtures_give_conventional_and_rgb_only_training():
    gen = torch.Generator().manual_seed(1)
    assert set(av.sample_conditions(500, presets.get("C"), gen)) == {av.COMPLETE}
    assert set(av.sample_conditions(500, presets.get("RGB_ONLY"), gen)) == {av.MISSING}


def test_eq2_keeps_depth_where_mask_is_one_and_the_constant_elsewhere():
    depth = torch.randn(3, 1, 12, 16)
    mask = (torch.rand(3, 1, 12, 16) > 0.5).float()
    out = av.apply_availability(depth, mask, c=0.0)
    assert torch.equal(out[mask == 1], depth[mask == 1])
    assert torch.all(out[mask == 0] == 0.0)
    out_c = av.apply_availability(depth, mask, c=-1.5)  # a non-zero empty value, e.g. another architecture
    assert torch.all(out_c[mask == 0] == -1.5)


def test_three_channel_depth_is_masked_identically_across_channels():
    depth = torch.randn(2, 3, 20, 20)
    mask = torch.zeros(2, 1, 20, 20)
    mask[:, :, :10] = 1
    out = av.apply_availability(depth, mask, 0.0)
    assert torch.equal(out[:, :, :10], depth[:, :, :10]) and torch.all(out[:, :, 10:] == 0)


def test_complete_and_missing_masks_are_all_ones_and_all_zeros():
    gen = torch.Generator().manual_seed(2)
    m = av.condition_masks([av.COMPLETE, av.MISSING], 30, 40, av.PartialMaskSpec(), gen)
    assert m.shape == (2, 1, 30, 40) and m[0].min() == 1 and m[1].max() == 0


def test_partial_masks_remove_a_bounded_share_and_keep_the_rest():
    gen = torch.Generator().manual_seed(3)
    spec = av.PartialMaskSpec(area=(0.10, 0.50))
    removed = []
    for _ in range(300):
        keep = av.sample_partial_mask(48, 64, spec, gen)
        assert keep.shape == (48, 64) and set(keep.unique().tolist()) <= {0.0, 1.0}
        removed.append(1.0 - keep.mean().item())
    assert 0.03 < min(removed) and max(removed) < 0.65  # rounding/overlap give some slack around (0.10, 0.50)
    assert 0.15 < float(np.mean(removed)) < 0.45


def test_rect_partial_mask_is_one_filled_rectangle():
    gen = torch.Generator().manual_seed(4)
    spec = av.PartialMaskSpec(kinds=("rect",))
    for _ in range(50):
        hole = av.sample_partial_mask(40, 50, spec, gen) == 0
        ys, xs = hole.any(1).nonzero().flatten(), hole.any(0).nonzero().flatten()
        box = hole[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        assert box.all(), "a single rectangle must be completely filled"


def test_sampler_is_reproducible_and_leaves_complete_samples_untouched():
    depth = torch.randn(64, 1, 24, 32)
    a, ca = av.AvailabilitySampler(c=0.0, seed=7)(depth)
    b, cb = av.AvailabilitySampler(c=0.0, seed=7)(depth)
    assert torch.equal(a, b) and ca == cb
    for i, cond in enumerate(ca):
        if cond == av.COMPLETE:
            assert torch.equal(a[i], depth[i])
        elif cond == av.MISSING:
            assert torch.all(a[i] == 0)


def test_test_time_conditions():
    depth = torch.randn(2, 1, 40, 50)
    assert torch.all(av.missing_depth(depth, 0.0) == 0)
    p = av.partial_depth(depth, 0.0, fraction=0.25, kind="rect", seed=5)
    removed = (p == 0).float().mean().item()
    assert 0.2 < removed < 0.3 and torch.equal(p, av.partial_depth(depth, 0.0, 0.25, "rect", 5))


def test_tbg_identity_on_example_endpoints():
    """Arithmetic fixtures only: the numbers are the ones printed in the manuscript's Table 1, used here to check
    the T/B/G formulas. They are NOT measurements of this repository (see scripts/retest_official.py for those)."""
    e = metrics.Endpoints(s_c=56.0, s_m=49.1, s_rgb=53.1)
    e.check_identity()
    assert round(e.t, 1) == -4.0 and round(e.g, 1) == 6.9 and round(e.b, 1) == 2.9
    d = metrics.Endpoints(s_c=56.6, s_m=56.1, s_rgb=53.1)
    assert round(d.t, 1) == 3.0 and round(d.s_m - e.s_m, 1) == 7.0
    assert metrics.Endpoints(56.5, 43.6).t is None  # T is undefined until an RGB-only score is supplied


def test_iou_from_confusion_matrix_and_classwise_delta():
    hist = np.array([[8, 2], [1, 9]], dtype=float)  # class 0: tp 8, fp 1, fn 2 -> 8/11; class 1: 9/12
    iou = metrics.iou_from_hist(hist)
    assert np.allclose(iou, [100 * 8 / 11, 100 * 9 / 12])
    assert metrics.miou_from_hist(hist) == pytest.approx(iou.mean())
    delta, mean = metrics.classwise_delta(hist, hist)
    assert np.all(delta == 0) and mean == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_msf_protocol_matches_single_scale_for_a_pointwise_model():
    """predict() must reduce to the plain forward pass for a model that commutes with resizing and flipping."""
    import torch.nn as nn

    from dat.evaluate import OFFICIAL_SCALES, predict

    class Pointwise(nn.Module):
        def forward(self, rgb, depth):
            return torch.cat([rgb[:, :1], depth[:, :1], rgb[:, 1:2] - depth[:, :1]], dim=1)

    class Host:
        @staticmethod
        def forward(model, rgb, depth):
            return model(rgb, depth)

    m, g = Pointwise(), torch.Generator().manual_seed(0)
    rgb, depth = torch.randn(1, 3, 64, 96, generator=g), torch.randn(1, 3, 64, 96, generator=g)
    single = predict(Host, m, rgb, depth).argmax(1)
    assert torch.equal(predict(Host, m, rgb, depth, (1.0,), True).argmax(1), single)  # flip-equivariant model
    ms = predict(Host, m, rgb, depth, OFFICIAL_SCALES, True)
    assert ms.shape == (1, 3, 64, 96) and torch.isfinite(ms).all()
    assert (ms.argmax(1) == single).float().mean() > 0.9  # resizing blurs a little; the decision is the same
