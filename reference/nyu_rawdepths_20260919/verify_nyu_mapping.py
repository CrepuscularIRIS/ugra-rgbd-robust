"""Verify the nyu_depth_v2_labeled.mat index <-> DFormer-package file id mapping.

The DFormer NYUDepthv2 package ships 8-bit Depth PNGs that are a per-image min-max
encoding of the INPAINTED depth.  If mat index i corresponds to file id f, then
encoding mat['depths'][i] the same way must reproduce Depth/f.png.  Anything less
than a near-exact match means the mapping is wrong and rawDepths must not be used.
"""
import sys
from pathlib import Path
import numpy as np, h5py, cv2

MAT = Path('/home/lingxufeng/workspace/data/NYU_raw/nyu_depth_v2_labeled.mat')
DATA = Path('/home/lingxufeng/workspace/ugra-rgbd-robust/repos/DFormer/datasets/NYUDepthv2')

def encode(d):
    lo, hi = float(d.min()), float(d.max())
    return np.clip(np.round((d - lo) / max(hi - lo, 1e-12) * 255.0), 0, 255).astype(np.uint8)

with h5py.File(MAT, 'r') as f:
    print('keys:', list(f.keys()))
    for k in ('depths', 'rawDepths', 'images', 'labels'):
        if k in f: print(f'  {k:10s} {f[k].shape} {f[k].dtype}')
    n = f['depths'].shape[0]
    probes = [int(x) for x in sys.argv[1:]] or [0, 1, 655, 1090, 1145, 1448]
    for fid in probes:
        png = cv2.imread(str(DATA/'Depth'/f'{fid}.png'), 0)
        if png is None:
            print(f'  id {fid}: no local PNG'); continue
        best = None
        for cand in {fid, fid-1, fid+1}:
            if not (0 <= cand < n): continue
            d = np.array(f['depths'][cand]).T          # HDF5 stores (W,H) -> transpose
            e = encode(d)
            agree = float((e == png).mean())
            mad = float(np.abs(e.astype(int) - png.astype(int)).mean())
            if best is None or agree > best[1]: best = (cand, agree, mad)
        print(f'  file id {fid:5d} -> mat index {best[0]:5d}   exact-pixel agreement {100*best[1]:6.2f}%   MAD {best[2]:.3f}')
