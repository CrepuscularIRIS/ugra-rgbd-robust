# shellcheck shell=bash
#
# Shared bootstrap for the scripts in this directory. Sourced, never executed.
# Everything is overridable from the caller's environment:
#
#   UGRA_ROOT          this repository                       (set by the calling script)
#   DFORMER_ROOT       upstream VCIP-RGBD/DFormer checkout   ($UGRA_ROOT/repos/DFormer)
#   CMNEXT_ROOT        upstream jamycheung/DELIVER (CMNeXt)  ($UGRA_ROOT/repos/DELIVER)
#   GEMINIFUSION_ROOT  upstream JiaDingCN/GeminiFusion       ($UGRA_ROOT/repos/GeminiFusion)
#   RGBD_DATA_ROOT     NYUDepthv2/ and SUNRGBD/ live here    ($UGRA_ROOT/data)

if [ -z "${UGRA_ROOT:-}" ]; then
    echo "env.sh: UGRA_ROOT is unset; source this from a script that sets it." >&2
    return 1 2>/dev/null || exit 1
fi
export UGRA_ROOT
export DFORMER_ROOT="${DFORMER_ROOT:-$UGRA_ROOT/repos/DFormer}"
export CMNEXT_ROOT="${CMNEXT_ROOT:-$UGRA_ROOT/repos/DELIVER}"
export GEMINIFUSION_ROOT="${GEMINIFUSION_ROOT:-$UGRA_ROOT/repos/GeminiFusion}"
export RGBD_DATA_ROOT="${RGBD_DATA_ROOT:-$UGRA_ROOT/data}"
