"""A2S (announced-to-silent curriculum) training — implementation per Benchmark-v1 §Method
with the Opus design fixes (2026-11-16): paired augmentation via random-state replay,
gates on H_k = S_m - S_k with scene-clustered paired bootstrap, level-then-family slot
split, recheck of skipped levels, and mixture driven by the enabled set only.

    python scripts/train_a2s.py --steps 120 --batch-size 8 --out runs/a2s_smoke
    python scripts/train_a2s.py --preset-only DAT --steps 60 --batch-size 8 --out runs/dat_base

Curriculum (paper Table + Algorithm 1):
  A = {}          mix 50:25:25:0            (DBT)
  A enabled       mix 40:20:20:20, corruption slot: uniform over enabled levels,
                  then uniform over families within the level
  LEVELS = {1: [f8cross], 2: [f5s50, f5s90]}
  gates at T/3 and 2T/3: probe 200 fixed train images (ValPre, single scale); enable
  level j (and recheck skipped ones) when LB95(H_j) > 0, H_j = S_m - S_j.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path.home() / "workspace/ugra-rgbd-robust"))
os.environ.setdefault("DFORMER_ROOT", "/data/projects/DFormer")

from eval_variants import NYU_BASE, shim_root  # noqa: E402

NYU_C = REPO / "data/NYU_corrupt"
LEVELS: dict[int, list[str]] = {1: ["f8cross"], 2: ["f5s50", "f5s90"]}
ROOTS = {"f8cross": NYU_C / "F8/cross/Depth",
         "f5s50": NYU_C / "F5/s50/Depth",
         "f5s90": NYU_C / "F5/s90/Depth"}
PROBE_N = 200
N_EXPECTED = 1449
GT_MIN_PIX = 1000  # probe class set: classes with at least this many GT pixels on the probe


class PairDataset(Dataset):
    """Clean dataset + variant datasets, same frame index, IDENTICAL augmentation.

    Upstream TrainPre draws all randomness from Python `random` (mirror / scale /
    crop), so replaying `random.getstate()` before each variant call reproduces the
    exact same transform; the per-sample `torch.equal(label)` assertion is the guard.
    augment=False uses ValPre (probe / no-augmentation runs)."""

    def __init__(self, host, fams: list[str], augment: bool = True):
        from dat.data import _setting
        from utils.dataloader.dataloader import TrainPre, ValPre
        from utils.dataloader.RGBXDataset import RGBXDataset
        from dat.hosts.dformer import make_host
        for f in fams:  # fail loudly on incomplete variants (same rule as make_variants)
            n = len(list((ROOTS[f]).glob("*.png")))
            if n != N_EXPECTED:
                raise FileNotFoundError(f"{ROOTS[f]}: {n}/{N_EXPECTED} depth files")
        cfg = host.cfg
        self.fams = fams
        pre = (lambda c: TrainPre(c.norm_mean, c.norm_std, c.x_is_single_channel, c)) if augment \
            else (lambda c: ValPre(c.norm_mean, c.norm_std, c.x_is_single_channel, c))
        self.clean = RGBXDataset(_setting(cfg), "train", pre(cfg))
        self.variants = []
        for f in fams:
            shim = shim_root("NYUDepthv2", NYU_BASE, ROOTS[f], None)
            h2 = make_host(host.name, "NYUDepthv2", data_root=str(shim))
            self.variants.append(RGBXDataset(_setting(h2.cfg), "train", pre(h2.cfg)))
        self.aligned = 0

    def __len__(self):
        return len(self.clean)

    def id_of(self, i: int) -> str:
        return Path(self.clean._file_names[i].split()[0]).stem

    def __getitem__(self, i):
        st = random.getstate()
        item = dict(self.clean[i])
        for name, ds in zip(self.fams, self.variants):
            random.setstate(st)  # replay the same mirror/scale/crop
            v = ds[i]
            assert os.path.basename(v["fn"]) == os.path.basename(item["fn"]), (v["fn"], item["fn"])
            assert torch.equal(v["label"], item["label"]), f"aug misaligned on {item['fn']}/{name}"
            item[f"modal_x_{name}"] = v["modal_x"]
        self.aligned += 1
        return item


class A2SSampler:
    """Mixture driven by the enabled set A only; slot: level first, then family."""

    def __init__(self, c: float, seed: int = 0):
        from dat.availability import PartialMaskSpec, apply_availability, sample_partial_mask
        self.c, self.spec = c, PartialMaskSpec()
        self.enabled: set[int] = set()
        self.gen = torch.Generator().manual_seed(seed + 1)
        self._av = apply_availability
        self._spm = sample_partial_mask

    def __call__(self, batch: dict) -> tuple[torch.Tensor, list[str]]:
        depth = batch["modal_x"]
        b, _, h, w = depth.shape
        mix = (0.50, 0.25, 0.25) if not self.enabled else (0.40, 0.20, 0.20)
        p0, p1, p2 = mix
        u = torch.rand(b, generator=self.gen)
        conds, out = [], depth.clone()
        lvls = sorted(self.enabled)
        for i, v in enumerate(u.tolist()):
            if v < p0:
                cond = "complete"
            elif v < p0 + p1:
                cond, out[i] = "missing", torch.full_like(depth[i], self.c)
            elif v < p0 + p1 + p2:
                cond = "partial"
                keep = self._spm(h, w, self.spec, self.gen)
                out[i] = self._av(depth[i], keep.unsqueeze(0).unsqueeze(0), self.c)[0]
            elif not lvls:
                cond, out[i] = "missing", torch.full_like(depth[i], self.c)  # A empty: no corrupt arm
            else:
                lvl = lvls[int(torch.randint(len(lvls), (1,), generator=self.gen))]
                fam = LEVELS[lvl][int(torch.randint(len(LEVELS[lvl]), (1,), generator=self.gen))]
                cond = f"corrupt:{fam}"
                out[i] = batch[f"modal_x_{fam}"][i]
            conds.append(cond)
        return out, conds


@torch.no_grad()
def probe_hists(host, model, loader, fams: list[str], c: float, device: str) -> dict[str, np.ndarray]:
    """Per-image confusion matrices for arms = missing + families (H needs S_m, not S_c)."""
    from dat.evaluate import predict
    k = host.num_classes
    arms = ["missing"] + fams
    out = {a: [] for a in arms}
    model.eval()
    for b in loader:
        rgb, lab = b["data"].to(device), b["label"].to(device)
        keep = lab != 255
        for a in arms:
            d = torch.full_like(b["modal_x"], c) if a == "missing" else b[f"modal_x_{a}"]
            pred = predict(host, model, rgb, d.to(device)).argmax(1)
            out[a].append(torch.bincount(lab[keep] * k + pred[keep], minlength=k * k).view(k, k).cpu().numpy())
    model.train()
    return {a: np.stack(v) for a, v in out.items()}


def level_harm_lb95(hists, fams, scene, classes, B: int = 1000, seed: int = 0) -> tuple[float, float]:
    """H_level = mean_f mIoU(missing) - mIoU(f); scene-cluster paired bootstrap. (point, LB95)."""
    S = int(scene.max()) + 1
    agg = {a: np.stack([h[scene == s].sum(0) for s in range(S)]) for a, h in hists.items()}
    rng = np.random.default_rng(seed)
    W = rng.multinomial(S, np.full(S, 1 / S), size=B)
    W = np.vstack([np.ones((1, S), np.int64), W])  # row 0 = point estimate

    def miou(A):
        h = np.einsum("bs,sij->bij", W, A)
        tp = np.einsum("bii->bi", h)[:, classes]
        u = (h.sum(1) + h.sum(2))[:, classes] - tp
        return np.nanmean(np.where(u > 0, tp / np.maximum(u, 1), np.nan), axis=1)

    m = miou(agg["missing"])
    harm = np.mean([m - miou(agg[f]) for f in fams], axis=0)
    return float(harm[0]), float(np.percentile(harm[1:], 2.5))


def _check_bootstrap() -> None:
    rng, k, N = np.random.default_rng(0), 4, 60
    scene = np.repeat(np.arange(20), 3)
    good = np.stack([np.diag(rng.integers(50, 100, k)) + rng.integers(0, 5, (k, k)) for _ in range(N)])
    bad = good.copy()
    bad[:, 0, 0] //= 4
    bad[:, 0, 1] += good[:, 0, 0] - bad[:, 0, 0]
    cls = np.arange(k)
    assert level_harm_lb95({"missing": good, "f": good}, ["f"], scene, cls)[1] <= 0
    assert level_harm_lb95({"missing": good, "f": bad}, ["f"], scene, cls)[1] > 0


def build_probe(train_ds: PairDataset, cache: Path, seed: int = 7):
    """Fixed 200-image probe (seeded sample over TRAIN-split ids, not file order) + scene ids."""
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["ids"], np.array(d["scene_idx"]), d["scene_names"]
    import h5py
    train_ids = [train_ds.id_of(i) for i in range(len(train_ds))]
    with h5py.File(REPO / "data/nyu_depth_v2_labeled.mat", "r") as f:
        from nyu_corrupt_core import read_frame
        scene_names = {i: read_frame(f, int(i))["scene"] for i in train_ids}
    rng = np.random.default_rng(seed)
    ids = [str(i) for i in rng.choice(sorted(train_ids), PROBE_N, replace=False)]
    uniq = sorted({scene_names[i] for i in ids})
    sidx = {s: k for k, s in enumerate(uniq)}
    d = {"ids": ids, "scene_idx": [sidx[scene_names[i]] for i in ids], "scene_names": uniq}
    cache.write_text(json.dumps(d))
    return d["ids"], np.array(d["scene_idx"]), d["scene_names"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="DFormerv2-S")
    ap.add_argument("--pretrained", default="/data/projects/DFormer/DFormerv2_Small_pretrained.pth")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-probe", action="store_true", help="probes unavailable: enable all levels on schedule")
    ap.add_argument("--preset-only", help="skip A2S: run plain DBT for the it/s baseline")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _check_bootstrap()

    from dat.hosts.dformer import make_host

    fams = sorted({f for lv in LEVELS.values() for f in lv})
    host = make_host(a.host, "NYUDepthv2", pretrained=a.pretrained, data_root=str(NYU_BASE.parent))
    model = host.build(device=a.device)
    sched = host.schedule()
    from utils.init_func import group_weight
    from utils.lr_policy import WarmUpPolyLR
    groups = group_weight([], model, torch.nn.BatchNorm2d, a.lr)
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=sched["weight_decay"])
    policy = WarmUpPolyLR(a.lr, 0.9, a.steps, a.steps // 10)

    ds = PairDataset(host, fams, augment=a.preset_only is None)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                        drop_last=True, pin_memory=True)
    probe_ids, probe_scene, _ = build_probe(ds, REPO / "data/a2s_probe.json")
    idx_of = {ds.id_of(i): i for i in range(len(ds))}
    probe_ds = PairDataset(host, fams, augment=False)
    probe_loader = DataLoader(Subset(probe_ds, [idx_of[i] for i in probe_ids]),
                              batch_size=1, shuffle=False, num_workers=2)

    sampler = A2SSampler(c=host.empty_value, seed=a.seed)
    enabled: set[int] = set(LEVELS) if a.no_probe else set()
    plain_dbt = None
    if a.preset_only:
        from dat.availability import AvailabilitySampler
        plain_dbt = AvailabilitySampler(c=host.empty_value, seed=a.seed)

    gates = {a.steps // 3: 1, 2 * a.steps // 3: 2}
    model.train()
    step, t0, run_loss, conds = 0, time.time(), 0.0, Counter()
    gate_log = []
    while step < a.steps:
        for batch in loader:
            if step >= a.steps:
                break
            if not a.preset_only and step in gates:
                tg = time.time()
                hists = probe_hists(host, model, probe_loader,
                                    sorted({f for lv in LEVELS.values() for f in lv}),
                                    host.empty_value, a.device)
                gt = hists["missing"].sum(0).sum(1)  # class set fixed from GT once
                classes = np.flatnonzero(gt >= GT_MIN_PIX)
                for j in range(1, gates[step] + 1):  # recheck every level not yet enabled
                    if j in enabled:
                        continue
                    pt, lb = level_harm_lb95(hists, LEVELS[j], probe_scene, classes, seed=step + j)
                    gate_log.append({"step": step, "level": j, "H": round(pt, 2),
                                     "lb95": round(lb, 2), "seconds": round(time.time() - tg, 1),
                                     "enabled": lb > 0})
                    if lb > 0:
                        enabled.add(j)
                sampler.enabled = enabled
                print(f"  gate@{step} {round(time.time()-tg,1)}s log={gate_log[-2:]} enabled={sorted(enabled)}",
                      flush=True)
                model.train()
            if plain_dbt is not None:
                depth_in, cs = plain_dbt(batch["modal_x"].to(a.device))
            else:
                depth_in, cs = sampler(batch)
            conds.update(cs)
            rgb = batch["data"].to(a.device, non_blocking=True)
            label = batch["label"].to(a.device, non_blocking=True)
            loss = host.loss(model, rgb, depth_in.to(a.device), label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            cur = policy.get_lr(step)
            for g in opt.param_groups:
                g["lr"] = cur
            opt.step()
            step += 1
            run_loss += float(loss.detach())
            if step % 20 == 0 or step == a.steps:
                print(f"step {step:5d}/{a.steps}  loss {run_loss / 20:.4f}  "
                      f"{step / (time.time() - t0):.2f} it/s  conditions {dict(conds)}", flush=True)
                run_loss = 0.0

    train_s = time.time() - t0
    summary = {"steps": a.steps, "batch_size": a.batch_size, "seconds": round(train_s, 1),
               "it_per_s_incl_gates": round(a.steps / train_s, 3),
               "gate_log": gate_log,
               "alignment_guards": "per-sample label-equality asserts run inside workers; no assertion fired",
               "projected_25k_steps_hours_at_bs8": round(25000 / (a.steps / train_s) / 3600, 2)}
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(os.path.join(a.out, "smoke_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
