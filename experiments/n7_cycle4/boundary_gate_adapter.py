"""n7 cycle-4 (M3) core modules — Detached Semantic-Boundary Gate + Zero-Init Parallel Edge Adapter.

Anti-conflation design (B04): the depth signal is used ONLY as second-ORDER evidence feeding a
boundary head that is supervised by the SEMANTIC LABEL boundary (not depth edges) and stop-gradiented
into the gate. A receding planar surface has a smooth first-order depth gradient but ~zero second-order
response AND is not a semantic boundary -> gate ~0 there -> the zero-init parallel adapter contributes
nothing on planar interiors (the exact classes n7-cycle3 corrupted).

Shared preprocessing (`second_order_depth_response`) MUST be imported by BOTH the training script and
the frozen eval branch so the B-input distribution matches between train and eval (prereg risk R6).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# DFormer TrainPre/ValPre depth normalize: (x/255 - mean) / std with these constants
# (see repos/DFormer/utils/dataloader/dataloader.py). Invert before any log/Laplacian.
DEPTH_NORM_MEAN = 0.48
DEPTH_NORM_STD = 0.28
# Below this per-sample peak |Laplacian|, treat the map as "no depth structure" and return zeros
# so max-normalization cannot amplify pure-ramp discretization noise into a false unit peak.
_RESP_PEAK_FLOOR = 1e-4


# ----------------------------------------------------------------------------------------------------
# Shared preprocessing (train + eval MUST use this exact function)
# ----------------------------------------------------------------------------------------------------
def denormalize_depth(
    depth_1ch: torch.Tensor,
    mean: float = DEPTH_NORM_MEAN,
    std: float = DEPTH_NORM_STD,
) -> torch.Tensor:
    """Invert dataloader depth normalize -> relative depth in ~[0, 1].

    modal_x from RGBXDataset is (pixel/255 - mean) / std, often negative. log/Laplacian on that
    tensor silently kills near-field structure (everything < mean clamps to eps). Always denorm
    before second_order_depth_response / depth_edge_target.
    """
    return (depth_1ch * std + mean).clamp(0.0, 1.0)


def _box_blur(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Fixed box prefilter (no learnable params). x: [B,1,H,W].

    Prefer box/Gaussian over median here: a median of a diagonal planar ramp is NOT the
    center pixel, so median injects curvature and re-creates the planar false-fire M3 forbids.
    Box blur preserves linear ramps (Laplacian of a linear field stays ~0).
    """
    pad = k // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(xp, kernel_size=k, stride=1)


def second_order_depth_response(
    depth_1ch: torch.Tensor,
    k_med: int = 5,
    eps: float = 1e-3,
    already_denorm: bool = False,
) -> torch.Tensor:
    """Second-order (curvature-like) depth response, robust to high-freq depth noise.

    A smooth planar ramp (constant first derivative) -> ~0 response at ANY slope; a true depth step
    -> a spike. This is the derivative-order antidote to the gradient/discontinuity conflation.

    depth_1ch: [B,1,H,W] dataloader-normalized depth by default (denorm'd inside). Pass
    already_denorm=True only if values are already relative depth in ~[0,1].
    Returns [B,1,H,W] in ~[0,1] (per-sample max-normalized) after a fixed box prefilter.
    Train and eval MUST call this same function (no train/eval branch).
    """
    _ = eps  # API compat (was log-depth clamp); linear Laplacian needs no log-eps
    d0 = depth_1ch if already_denorm else denormalize_depth(depth_1ch)
    d = _box_blur(d0, k_med)                               # damp noise BEFORE differentiation
    # Linear Laplacian (NOT log-depth): second difference is exactly ~0 on constant-gradient
    # ramps at any slope; log-depth Laplacian is -a^2/(ax+b)^2 and false-fires on receding planes.
    # That false fire is the same class of conflation M3 was designed to kill.
    lap_k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                         dtype=d.dtype, device=d.device).view(1, 1, 3, 3)
    lap = F.conv2d(F.pad(d, (1, 1, 1, 1), mode="reflect"), lap_k)
    resp = lap.abs()
    # Peak from interior only. Reflect-pad creates ~k//2 px edge junk on pure ramps;
    # if that junk sets the max, max-norm fabricates a unit-peak false "boundary" map.
    br = k_med // 2 + 1  # box-pad + Laplacian pad: edge junk spans k//2+1 px
    if resp.shape[-1] > 2 * br and resp.shape[-2] > 2 * br:
        peak_src = resp[:, :, br:-br, br:-br]
    else:
        peak_src = resp
    m = peak_src.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
    weak = m < _RESP_PEAK_FLOOR
    out = (resp / m.clamp_min(1e-6)).clamp(0.0, 1.0)
    # Zero pad-artifact border so it cannot act as a free edge signal into B.
    if resp.shape[-1] > 2 * br and resp.shape[-2] > 2 * br:
        out = out.clone()
        out[:, :, :br, :] = 0
        out[:, :, -br:, :] = 0
        out[:, :, :, :br] = 0
        out[:, :, :, -br:] = 0
    return torch.where(weak, torch.zeros_like(out), out)


def label_boundary_target(label: torch.Tensor, ignore_index: int, width: int = 2) -> torch.Tensor:
    """Morphological-gradient boundary of the SEMANTIC label map (NOT depth edges).

    label: [B,H,W] long. Returns [B,1,H,W] float in {0,1}: 1 on a `width`-px band straddling any
    class transition, 0 elsewhere. ignore_index pixels never generate a positive boundary.
    """
    lab = label.clone()
    valid = (lab != ignore_index).float().unsqueeze(1)     # [B,1,H,W]
    labf = lab.float().unsqueeze(1)
    labf = labf * valid                                    # zero-out ignore so it doesn't fabricate edges
    k = 2 * width + 1
    dil = F.max_pool2d(labf, k, 1, width)
    ero = -F.max_pool2d(-labf, k, 1, width)
    boundary = (dil != ero).float()
    # suppress boundaries touching ignore regions (erode the valid mask by the same kernel)
    valid_er = -F.max_pool2d(-valid, k, 1, width)
    boundary = boundary * valid_er
    return boundary


def depth_edge_target(
    depth_1ch: torch.Tensor,
    thresh_pct: float = 0.90,
    already_denorm: bool = False,
) -> torch.Tensor:
    """FIRST-order depth-edge target — the WRONG supervision (ablation arm A2/A3 `supervision=depth`).

    This deliberately conflates within-object gradients with edges: a per-sample high-percentile
    threshold on the raw first-order |grad(depth)|. If the model trained on THIS regresses the planar
    cluster (vs label-boundary supervision), the decoupling claim is confirmed.
    depth_1ch: [B,1,H,W] dataloader-normalized by default (denorm'd inside).
    """
    d0 = depth_1ch if already_denorm else denormalize_depth(depth_1ch)
    d = _box_blur(d0, 5)
    gx = F.pad(d, (1, 0, 0, 0), mode="reflect")[:, :, :, 1:] - d
    gy = F.pad(d, (0, 0, 1, 0), mode="reflect")[:, :, 1:, :] - d
    g = (gx.abs() + gy.abs())
    # per-sample percentile threshold -> a binary "edge" map that fires on smooth ramps too
    flat = g.flatten(1)
    thr = torch.quantile(flat, thresh_pct, dim=1).view(-1, 1, 1, 1)
    return (g >= thr).to(depth_1ch.dtype)


# ----------------------------------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------------------------------
class _DWSepConv(nn.Module):
    """Depthwise-separable conv (3x3 dw + 1x1 pw)."""
    def __init__(self, cin, cout, act=True):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, 1, 0, bias=True)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class BoundaryHead(nn.Module):
    """Predict a semantic-boundary map B in [0,1] at the stage-2 resolution.

    Inputs (all resized to stage-2 HxW): second-order depth response (1ch), a projection of RGB
    stem features (rgb_proj_dim), and a depth-availability mask (1ch). Trained ONLY by pos-weighted
    BCE vs label_boundary_target; B is stop-gradiented before it gates the adapter.
    """
    def __init__(self, rgb_in_dim, rgb_proj_dim=32, hidden=32):
        super().__init__()
        self.rgb_proj = nn.Conv2d(rgb_in_dim, rgb_proj_dim, 1, 1, 0)
        cin = rgb_proj_dim + 1 + 1                          # rgb_proj + depth_resp + avail_mask
        self.body = nn.Sequential(
            _DWSepConv(cin, hidden),
            _DWSepConv(hidden, hidden),
        )
        self.out = nn.Conv2d(hidden, 1, 1, 1, 0)            # logits (BCEWithLogits)

    def forward(self, depth_resp, rgb_feat, avail_mask):
        r = self.rgb_proj(rgb_feat)
        x = torch.cat([r, depth_resp, avail_mask], dim=1)
        return self.out(self.body(x))                      # [B,1,H,W] logits


class ZeroInitParallelAdapter(nn.Module):
    """Parallel edge adapter dim->bottleneck->dim (NEW params, added in parallel; final conv zero-init
    so epoch-0 output == baseline)."""
    def __init__(self, dim=256, bottleneck=64):
        super().__init__()
        self.reduce = _DWSepConv(dim, bottleneck)
        self.expand_dw = nn.Conv2d(bottleneck, bottleneck, 3, 1, 1, groups=bottleneck, bias=False)
        self.expand_pw = nn.Conv2d(bottleneck, dim, 1, 1, 0, bias=True)
        nn.init.zeros_(self.expand_pw.weight)
        nn.init.zeros_(self.expand_pw.bias)                # zero-init => adapter output starts at 0

    def forward(self, feat):
        return self.expand_pw(self.expand_dw(self.reduce(feat)))
