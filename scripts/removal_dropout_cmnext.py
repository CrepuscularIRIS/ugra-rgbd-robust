"""Named-Removal Dropout for CMNeXt — the operator, plus the canary that proves it is EXACT.

R IS NOT INVENTED HERE. It is CMNeXt's own depth-free configuration `MODALS: ['img']`, already in
`semseg/models/backbones/cmnext.py` as the `else` branch of every `if self.num_modals > 0`:

    if self.num_modals > 0:                     else:
        x{i}_cam, x{i}_f = self.FRMs[k](...)        outs.append(x{i}_cam)
        x_fused = self.FFMs[k](x{i}_cam, x{i}_f)
        outs.append(x_fused)

Two things happen in that branch, and BOTH must be undone per dropped sample or R is only
approximate:

  1. `FRMs` MUTATES THE TRUNK. Its forward is
         out_x1 = x1 + lambda_c * cw[1] * x2 + lambda_s * sw[1] * x2
     and the returned `out_x1` feeds `patch_embed{i+1}`. So depth reaches the RGB trunk even
     though the trunk never consumes `x_fused`. Undoing only step 2 leaves this in place.
  2. `outs` receives `x_fused` instead of `x{i}_cam`.

So per dropped sample we select the PRE-FRM `x{i}_cam` at both sites. That is exactly what the
depth-free network computes, which is why the canary below can demand BITWISE equality rather
than a tolerance.

WHY THE CANARY IS THE WHOLE POINT. "Removal Dropout" is only a method if the thing it trains on is
genuinely the named removal. If the canary fails, the arm trains on some third thing that is
neither the sentinel nor R, and the experiment answers no question. NO GPU IS SPENT BEFORE IT PASSES.

    python3 scripts/removal_dropout_cmnext.py --canary --gpu 1
"""
import argparse
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT / "repos" / "DELIVER"
BACKBONE_VARIANT = "B4"   # the audited checkpoint is CMNeXt-B4; the canary must run on it

# Set by the training loop each step: a bool tensor (B,) — True means "this sample sees R".
_DROP = None


def set_drop(mask) -> None:
    global _DROP
    _DROP = mask


def get_drop():
    return _DROP


def _sel(drop, a, b):
    """Per-sample select: a where dropped, b otherwise. Shapes are (B,C,H,W)."""
    if drop is None:
        return b
    m = drop.reshape(-1, *([1] * (b.dim() - 1))).to(b.device)
    return torch.where(m, a, b)


def install(cmnext_module):
    """Monkeypatch CMNeXt.forward to honour the per-sample removal mask.

    Mirrors the upstream forward exactly; the ONLY additions are the two `_sel` calls per stage.
    Kept as a patch rather than an edit so the audited repo stays byte-identical on disk.
    """
    import functools
    import torch as _t

    def forward(self, x: list) -> list:
        drop = get_drop()
        x_cam = x[0]
        if self.num_modals > 0:
            x_ext = x[1:]
        B = x_cam.shape[0]
        outs = []

        stages = [
            (self.patch_embed1, self.block1, self.norm1, 0),
            (self.patch_embed2, self.block2, self.norm2, 1),
            (self.patch_embed3, self.block3, self.norm3, 2),
            (self.patch_embed4, self.block4, self.norm4, 3),
        ]
        extra = [
            (getattr(self, "extra_block1", None), getattr(self, "extra_norm1", None)),
            (getattr(self, "extra_block2", None), getattr(self, "extra_norm2", None)),
            (getattr(self, "extra_block3", None), getattr(self, "extra_norm3", None)),
            (getattr(self, "extra_block4", None), getattr(self, "extra_norm4", None)),
        ]

        cur = x_cam
        for k, (pe, blocks, norm, idx) in enumerate(stages):
            cur, H, W = pe(cur)
            for blk in blocks:
                cur = blk(cur, H, W)
            x_cam_pre = norm(cur).reshape(B, H, W, -1).permute(0, 3, 1, 2)

            if self.num_modals > 0:
                x_ext, _, _ = self.extra_downsample_layers[idx](x_ext)
                x_f = (self.tokenselect(x_ext, self.extra_score_predictor[idx])
                       if self.num_modals > 1 else x_ext[0])
                for blk in extra[idx][0]:
                    x_f = blk(x_f)
                x_f_n = extra[idx][1](x_f)

                x_cam_frm, x_f_frm = self.FRMs[idx](x_cam_pre, x_f_n)
                x_fused = self.FFMs[idx](x_cam_frm, x_f_frm)

                # (1) undo the trunk mutation, (2) undo the fused output — per dropped sample
                cur_out = _sel(drop, x_cam_pre, x_cam_frm)
                outs.append(_sel(drop, x_cam_pre, x_fused))

                if idx < 3:
                    x_ext = ([x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x_f_frm
                              for x_ in x_ext] if self.num_modals > 1 else [x_f_frm])
                cur = cur_out
            else:
                outs.append(x_cam_pre)
                cur = x_cam_pre

        return outs

    cmnext_module.CMNeXt.forward = forward
    return forward


def canary(gpu: str) -> int:
    """With the mask set for EVERY sample, the patched RGB-D model must reproduce the
    depth-free `MODALS:['img']` model BITWISE. Anything else means R is not R."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))

    from semseg.models.backbones import cmnext as CM

    B, C, H, W = 2, 3, 512, 512
    torch.manual_seed(0)
    rgb = torch.randn(B, C, H, W)
    dep = torch.randn(B, C, H, W)

    # reference: the architecture's OWN depth-free network
    torch.manual_seed(1)
    ref = CM.CMNeXt(BACKBONE_VARIANT, ["img"]).eval()
    # subject: the RGB-D network, same trunk weights, with the removal mask fully on
    torch.manual_seed(1)
    sub = CM.CMNeXt(BACKBONE_VARIANT, ["img", "depth"]).eval()
    res = sub.load_state_dict(ref.state_dict(), strict=False)
    # `missing_keys` here are the RGB-D model's EXTRA params (FRM/FFM/extra_*), which the
    # depth-free model does not have — expected, and harmless because the mask discards their
    # output. What must be zero is `unexpected_keys`: a reference weight that found no home
    # would mean the two trunks are not the same trunk.
    n_ref = len(ref.state_dict())
    print(f"reference trunk keys: {n_ref}   unconsumed: {len(res.unexpected_keys)}   "
          f"RGB-D-only extras left at init: {len(res.missing_keys)}")
    if res.unexpected_keys:
        print(f"  REFUSING: {len(res.unexpected_keys)} reference weights did not transfer "
              f"(e.g. {res.unexpected_keys[:3]}); the comparison would be void")
        return 2

    install(CM)

    with torch.no_grad():
        set_drop(None)
        out_ref = ref([rgb])
        set_drop(torch.ones(B, dtype=torch.bool))
        out_sub = sub([rgb, dep])

    worst = max((a - b).abs().max().item() for a, b in zip(out_ref, out_sub))
    print(f"\nmax |depth-free  −  removal-masked RGB-D| over {len(out_ref)} stages: {worst:.3e}")

    # the control: with the mask OFF, the two must DIFFER, or the mask is doing nothing
    with torch.no_grad():
        set_drop(torch.zeros(B, dtype=torch.bool))
        out_on = sub([rgb, dep])
    ctrl = max((a - b).abs().max().item() for a, b in zip(out_ref, out_on))
    print(f"max |depth-free  −  fused RGB-D| (must be >> 0):            {ctrl:.3e}")

    ok = worst < 1e-6 and ctrl > 1e-3
    print("\nVERDICT")
    if worst >= 1e-6:
        print(f"  FAIL — the masked path is NOT the depth-free path ({worst:.3e}). R is not R;")
        print("  Removal Dropout would train on a third thing. Do not run the arm.")
    elif ctrl <= 1e-3:
        print("  FAIL — the mask changes nothing; the control did not move.")
    else:
        print("  PASS — with the mask on, the RGB-D network reproduces its own depth-free")
        print("  forward EXACTLY, and with it off the two differ. R is implemented correctly")
        print("  and Named-Removal Dropout trains on the named removal.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--gpu", default="1")
    a = ap.parse_args()
    sys.exit(canary(a.gpu) if a.canary else 0)
