"""Where third-party code and data live. Everything is overridable from the environment, mirroring scripts/env.sh.

    UGRA_ROOT          this repository                       (auto-detected)
    DFORMER_ROOT       upstream VCIP-RGBD/DFormer checkout   ($UGRA_ROOT/repos/DFormer)
    CMNEXT_ROOT        upstream jamycheung/DELIVER (CMNeXt)  ($UGRA_ROOT/repos/DELIVER)
    GEMINIFUSION_ROOT  upstream JiaDingCN/GeminiFusion       ($UGRA_ROOT/repos/GeminiFusion)
    RGBD_DATA_ROOT     NYUDepthv2/ and SUNRGBD/ live here    ($UGRA_ROOT/data)
    DAT_EXTRA_PATH     os.pathsep-separated extra sys.path entries (e.g. a folder of pure-python deps)
"""
from __future__ import annotations

import os
import sys

ROOT = os.environ.get("UGRA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DFORMER_ROOT = os.environ.get("DFORMER_ROOT", os.path.join(ROOT, "repos", "DFormer"))
CMNEXT_ROOT = os.environ.get("CMNEXT_ROOT", os.path.join(ROOT, "repos", "DELIVER"))
GEMINIFUSION_ROOT = os.environ.get("GEMINIFUSION_ROOT", os.path.join(ROOT, "repos", "GeminiFusion"))
DATA_ROOT = os.environ.get("RGBD_DATA_ROOT", os.path.join(ROOT, "data"))


def activate(*roots: str) -> None:
    """Put third-party roots (and DAT_EXTRA_PATH) on sys.path, first match wins."""
    extra = [p for p in os.environ.get("DAT_EXTRA_PATH", "").split(os.pathsep) if p]
    for p in [*extra, *roots]:
        if p and p not in sys.path:
            sys.path.insert(0, p)
