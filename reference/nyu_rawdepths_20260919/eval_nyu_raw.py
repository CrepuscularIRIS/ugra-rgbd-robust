"""NYU test654 under its OWN real sensor invalidity masks (rawDepths == 0).

Three conditions:
  selfcheck_missing  all-zero availability -> must reproduce the frozen harness 'missing'
  raw_aligned        image i gets image i's real mask  (holes sit on that scene's own
                     glass / specular / far surfaces -- the physically correct test)
  raw_shuffled       image i gets a permuted image's real mask  (identical mask
                     distribution, scene-content correlation destroyed)

aligned vs shuffled isolates whether real failure hurts MORE because it lands on the
surfaces the model most depends on depth for.  Per-image confusions are kept so the
result can be stratified by the image's own hole fraction without breaking alignment.
"""
import os, sys, json, time
from pathlib import Path
CODE = Path('/home/lingxufeng/workspace/rgbd-ghost-depth/v2-floor-20260915/code')
sys.path.insert(0, str(CODE))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
import numpy as np, torch, h5py
from data import NYU
from model import Segmenter

HERE = Path('/tmp/claude-1000/-home-lingxufeng-workspace/d7138930-4084-445d-b683-50dadf463c22/scratchpad/tum')
MAT  = Path('/home/lingxufeng/workspace/data/NYU_raw/nyu_depth_v2_labeled.mat')
ROOT = Path('/home/lingxufeng/workspace/rgbd-ghost-depth')
ARMS = {
    'L0': (ROOT/'v2-floor-20260915/runs/VR0/checkpoint_latest.pth', 51.71083587183431),
    'C':  (ROOT/'architecture-loop-20260913/rounds/round2/progressive192/checkpoint_latest.pth', 48.22010026651055),
    'R':  (ROOT/'semantic-loss-20260914/runs/soft_iou/checkpoint_latest.pth', 52.01002433333796),
}

ds = NYU('test.txt', 480, 640, limit=654)
ids = [int(r[0].split('/')[-1].split('.')[0]) for r in ds.rows]
print(f'NYU test images: {len(ds)}', flush=True)

print('loading real masks from rawDepths ...', flush=True)
with h5py.File(MAT, 'r') as f:
    masks = np.stack([(np.array(f['rawDepths'][i]).T != 0).astype(np.float32) for i in ids])
hole = 1 - masks.reshape(len(ids), -1).mean(1)
print(f'real masks: {masks.shape}, hole fraction median {np.median(hole):.3f}', flush=True)
perm = np.random.default_rng(20260919).permutation(len(ids))
assert (perm != np.arange(len(ids))).mean() > 0.99

loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2, shuffle=False)

@torch.no_grad()
def run(model, cond):
    cms, realized = [], []
    for b, (x, d, y, m, bids) in enumerate(loader):
        s = b * 2; k = len(x)
        if cond == 'selfcheck_missing':
            added = torch.zeros_like(d)
        else:
            idx = np.arange(s, s + k) if cond == 'raw_aligned' else perm[s:s + k]
            added = torch.from_numpy(masks[idx][:, None])
        valid = m * added * torch.isfinite(d)
        dc = torch.where(valid.bool(), d, torch.zeros_like(d))
        realized.append(float(1 - valid.mean()))
        x, dc, vc, y = [v.cuda(non_blocking=True) for v in [x, dc, valid, y]]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x, dc, vc)
        pred = logits.argmax(1)
        for gt, pr in zip(y, pred):
            v = gt != 255
            cms.append(torch.bincount((gt[v] * 40 + pr[v]).long(), minlength=1600).reshape(40, 40).cpu().numpy())
    return np.stack(cms), float(np.mean(realized))

def miou(cm):
    u = cm.sum(0) + cm.sum(1) - cm.diagonal()
    return float(np.mean(cm.diagonal()[u > 0] / u[u > 0]) * 100)

summary, percm = {}, {}
for tag, (ckpt, ref) in ARMS.items():
    t0 = time.time()
    torch.manual_seed(20260913); torch.cuda.manual_seed_all(20260913)
    torch.backends.cudnn.benchmark = False
    model = Segmenter('progressive', 192, 'original'); model.load_encoder()
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model'], strict=True)
    model.cuda().eval()
    cms, _ = run(model, 'selfcheck_missing'); got = miou(cms.sum(0))
    ok = abs(got - ref) <= 1e-6
    print(f'[SELF-CHECK] {tag} {got:.9f} vs {ref:.9f} -> {"PASS" if ok else "FAIL"}', flush=True)
    summary[tag] = {'self_check': 'PASS' if ok else 'FAIL', 'selfcheck_missing': got, 'miou': {}, 'missing_fraction': {}}
    if ok:
        for cond in ('raw_aligned', 'raw_shuffled'):
            cms, mf = run(model, cond)
            percm[f'{tag}_{cond}'] = cms
            summary[tag]['miou'][cond] = miou(cms.sum(0))
            summary[tag]['missing_fraction'][cond] = mf
            print(f'  {tag} {cond:13s} mIoU {summary[tag]["miou"][cond]:.4f}  realized missing {mf:.4f}', flush=True)
    summary[tag]['minutes'] = round((time.time() - t0) / 60, 1)
    del model; torch.cuda.empty_cache()

# stratify the ALIGNED result by each image's own hole fraction
edges = np.percentile(hole, [0, 33.3, 66.7, 100]); edges[-1] += 1e-9
strata = {}
for i in range(3):
    sel = (hole >= edges[i]) & (hole < edges[i + 1])
    strata[f'tercile{i+1}'] = {'n': int(sel.sum()),
                               'hole_mean': float(hole[sel].mean()),
                               'availability': float(1 - hole[sel].mean()),
                               'miou': {t: miou(percm[f'{t}_raw_aligned'][sel].sum(0)) for t in ARMS}}
    print(f"  tercile{i+1}: n={sel.sum():3d}  avail {100*(1-hole[sel].mean()):.1f}%  "
          + '  '.join(f'{t} {strata[f"tercile{i+1}"]["miou"][t]:.2f}' for t in ARMS), flush=True)

summary['_strata'] = strata
summary['_hole'] = {'median': float(np.median(hole)), 'mean': float(hole.mean()),
                    'min': float(hole.min()), 'max': float(hole.max())}
(HERE/'NYU_RAW_RESULTS.json').write_text(json.dumps(summary, indent=2))
print('\nDONE', flush=True)
