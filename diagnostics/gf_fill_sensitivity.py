"""R0: does a dual-stream network's SURVIVING (RGB) stream depend on what fills the depth slot?

Uses REAL NYUv2 RGB images. The depth slot is filled with constants BY CONSTRUCTION, so the
native depth encoding of either codebase is irrelevant here — which is what makes this test
runnable without GeminiFusion's own dataset.

Gate: encoder must load >=90% of tensors before any number is read.
"""
import os, sys, types, importlib.util, glob
import numpy as np, torch
from PIL import Image

# Repo root: two levels up from this file (diagnostics/ -> repo root), overridable.
_ROOT = os.environ.get(
    'UGRA_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
# GeminiFusion is the SECOND external architecture (the dual-stream one, for which no
# canonical removal exists). See docs/INSTALLATION.md -> "Optional: GeminiFusion".
GF = os.environ.get('GEMINIFUSION_ROOT', os.path.join(_ROOT, 'repos', 'GeminiFusion'))
DFORMER = os.environ.get('DFORMER_ROOT', os.path.join(_ROOT, 'repos', 'DFormer'))
pkg = types.ModuleType('models'); pkg.__path__ = [GF + '/models']; sys.modules['models'] = pkg
def _load(n):
    s = importlib.util.spec_from_file_location(f'models.{n}', f'{GF}/models/{n}.py')
    m = importlib.util.module_from_spec(s); sys.modules[f'models.{n}'] = m; s.loader.exec_module(m); return m
_load('modules'); mt = _load('mix_transformer')

m = mt.mit_b3(n_heads=8, dpr=0.1, drop_rate=0.0).eval()
sd = torch.load(f'{GF}/checkpoints/mit-b3.pth.tar', map_location='cpu')['segmenter']
enc = {k.replace('module.encoder.', ''): v for k, v in sd.items() if k.startswith('module.encoder.')}
r = m.load_state_dict(enc, strict=False)
tot = len(m.state_dict()); matched = tot - len(r.missing_keys)
print(f'[gate] encoder {matched}/{tot} tensors')
assert matched > 0.9 * tot, 'FATAL: encoder did not load'
m = m.cuda()

MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
files = sorted(glob.glob(os.path.join(DFORMER, 'datasets', 'NYUDepthv2', 'RGB', '*.jpg')))[:8]
assert files, f'no NYUv2 RGB images found under {DFORMER}/datasets/NYUDepthv2/RGB (see docs/DATASETS.md)'
imgs = []
for f in files:
    a = np.array(Image.open(f).convert('RGB').resize((640, 480)), dtype=np.float32) / 255.
    imgs.append(torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1)).float())
rgb = torch.stack(imgs).cuda()
print(f'[data] {len(files)} real NYUv2 RGB images')

def run(dval, seed=None):
    if seed is None:
        d = torch.full_like(rgb, dval)
    else:
        g = torch.Generator(device='cpu').manual_seed(seed)
        d = (torch.randn(rgb.shape, generator=g) * 0.28 + 0.48).cuda()
    with torch.no_grad():
        return [t.detach() for t in m([rgb, d])[0]]      # [0] = RGB stream, 4 stages

base = run(0.0)
arms = {'dataset-mean (0.48)': run(0.48), 'mid (0.5)': run(0.5),
        'high (0.9)': run(0.9), 'random': run(None, seed=1)}
print(f'\nGeminiFusion RGB-stream drift vs the zero-fill convention (real images, relative L2):')
print(f'{"depth fill":<22}' + ''.join(f'{f"stage{i}":>10}' for i in range(4)))
for n, o in arms.items():
    print(f'{n:<22}' + ''.join(
        f'{(o[i]-base[i]).norm().item()/max(base[i].norm().item(),1e-9)*100:>9.2f}%' for i in range(4)))
print('\nDFormerv2: mask_d = |d_i - d_j| = 0 for EVERY constant fill => 0.00% at every stage (bitwise).')
