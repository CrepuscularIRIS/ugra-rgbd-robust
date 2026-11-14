#!/usr/bin/env bash
#
# Fetch the upstream CMNeXt codebase (jamycheung/DELIVER) and apply the DAT patch.
#
# patches/cmnext-dat.patch adds depth-availability training to upstream's tools/train_mm.py behind
# --dat-preset (default off: conventional training is unchanged). It is generated against, and verified
# to apply cleanly to, upstream commit UPSTREAM_COMMIT below.
#
# Usage:
#   bash scripts/setup_cmnext.sh            # clone + pin + patch  (idempotent)
#   bash scripts/setup_cmnext.sh --check    # report PATCHED / STOCK / UNKNOWN for an existing checkout
#
# Upstream imports fvcore and tabulate (pure python, `pip install fvcore tabulate`) on import.
#
set -euo pipefail

# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"
CMNEXT_ROOT="${CMNEXT_ROOT:-$UGRA_ROOT/repos/DELIVER}"

UPSTREAM_URL="https://github.com/jamycheung/DELIVER.git"
UPSTREAM_COMMIT="b3a76c33a7b22098b7285d845f22c910abc5cb16"
PATCH="$UGRA_ROOT/patches/cmnext-dat.patch"

# Files the patch touches. Used to verify the result.
PATCHED_FILES=(
    tools/train_mm.py
)

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[ -f "$PATCH" ] || die "patch not found: $PATCH"

# --------------------------------------------------------------------------- check
if [ "${1:-}" = "--check" ]; then
    [ -d "$CMNEXT_ROOT" ] || die "no checkout at $CMNEXT_ROOT"
    cd "$CMNEXT_ROOT"
    if git apply --reverse --check "$PATCH" 2>/dev/null; then
        log "PATCHED — $CMNEXT_ROOT carries the DAT patch."
        exit 0
    elif git apply --check "$PATCH" 2>/dev/null; then
        die "STOCK — $CMNEXT_ROOT is unpatched. Run without --check to apply."
    else
        die "UNKNOWN — the checkout matches neither stock nor patched state.
     It may be locally modified, or at a different upstream commit than
     $UPSTREAM_COMMIT. Inspect with: cd $CMNEXT_ROOT && git status"
    fi
fi

# --------------------------------------------------------------------------- clone
if git -C "$CMNEXT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log "existing checkout at $CMNEXT_ROOT — reusing it"
else
    log "cloning $UPSTREAM_URL -> $CMNEXT_ROOT"
    mkdir -p "$(dirname "$CMNEXT_ROOT")"
    git clone "$UPSTREAM_URL" "$CMNEXT_ROOT"
fi

cd "$CMNEXT_ROOT"

# --------------------------------------------------------------------------- patch
# An already-patched checkout is a dirty tree, so test for that before the dirty-tree guard.
if git apply --reverse --check "$PATCH" 2>/dev/null; then
    log "already patched — nothing to do"
else
    log "pinning upstream to $UPSTREAM_COMMIT"
    git cat-file -e "${UPSTREAM_COMMIT}^{commit}" 2>/dev/null || git fetch --all --tags --quiet || true
    git cat-file -e "${UPSTREAM_COMMIT}^{commit}" 2>/dev/null \
        || die "upstream commit $UPSTREAM_COMMIT not found in $CMNEXT_ROOT."
    if ! git diff --quiet || ! git diff --cached --quiet; then
        die "the checkout at $CMNEXT_ROOT has uncommitted changes that are not this
     project's patch; refusing to touch it. Stash or remove them, then re-run."
    fi
    git checkout --quiet --detach "$UPSTREAM_COMMIT"

    log "applying patches/cmnext-dat.patch"
    git apply --check "$PATCH" || die "patch does not apply cleanly at $UPSTREAM_COMMIT"
    git apply "$PATCH"
    log "patch applied"
fi

# --------------------------------------------------------------------------- verify
for f in "${PATCHED_FILES[@]}"; do
    [ -f "$f" ] || die "expected patched file missing: $f"
done
git apply --reverse --check "$PATCH" 2>/dev/null || die "post-apply verification failed"
[ "$(git rev-parse HEAD)" = "$UPSTREAM_COMMIT" ] || log "note: HEAD is not $UPSTREAM_COMMIT (patch still applies)"

cat <<EOT

$(log "CMNeXt (DELIVER) is ready at $CMNEXT_ROOT")

Train (from $CMNEXT_ROOT, DDP as upstream):
  UGRA_ROOT=$UGRA_ROOT python tools/train_mm.py --cfg configs/nyu_rgbd.yaml --dat-preset DAT
  (without --dat-preset the run is conventional upstream training)
Released NYU weights: https://huggingface.co/InSAI-Lab/CMNeXt  (NYU_Depth_V2/cmnext_b4_nyu_rgbd.pth)

EOT
