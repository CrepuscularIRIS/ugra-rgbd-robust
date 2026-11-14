#!/usr/bin/env bash
#
# Fetch the upstream DFormer codebase (DFormerv1 and DFormerv2) at a pinned commit.
#
# DAT needs no patch for DFormer: it only changes the depth tensor passed to the model, which dat/train.py does
# itself. This script therefore only clones and pins; the checkout stays stock upstream.
#
# Usage:
#   bash scripts/setup_dformer.sh            # clone + pin  (idempotent)
#   bash scripts/setup_dformer.sh --check    # verify an existing checkout is at the pinned commit
#
# DFORMER_URL overrides the clone source (e.g. a local mirror).
#
set -euo pipefail

UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

UPSTREAM_URL="${DFORMER_URL:-https://github.com/VCIP-RGBD/DFormer.git}"
UPSTREAM_COMMIT="23cd8f833df12c74af45bc671e75891e48f8ff1b"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

verify() {
    [ -f "$DFORMER_ROOT/models/builder.py" ] || die "no DFormer checkout at $DFORMER_ROOT"
    local head
    head="$(git -C "$DFORMER_ROOT" rev-parse HEAD 2>/dev/null || true)"
    if [ "$head" = "$UPSTREAM_COMMIT" ]; then
        log "OK — $DFORMER_ROOT is at the pinned upstream commit"
    else
        log "note: $DFORMER_ROOT is at ${head:-an unknown state}, not the pinned $UPSTREAM_COMMIT (DAT usually still works)"
    fi
}

if [ "${1:-}" = "--check" ]; then
    verify
    exit 0
fi

if git -C "$DFORMER_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log "existing checkout at $DFORMER_ROOT — reusing it"
else
    log "cloning $UPSTREAM_URL -> $DFORMER_ROOT"
    mkdir -p "$(dirname "$DFORMER_ROOT")"
    git clone "$UPSTREAM_URL" "$DFORMER_ROOT"
fi

cd "$DFORMER_ROOT"
if [ "$(git rev-parse HEAD)" != "$UPSTREAM_COMMIT" ]; then
    if ! git diff --quiet || ! git diff --cached --quiet; then
        die "the checkout at $DFORMER_ROOT has uncommitted changes; refusing to move it to $UPSTREAM_COMMIT."
    fi
    git cat-file -e "${UPSTREAM_COMMIT}^{commit}" 2>/dev/null || git fetch --all --tags --quiet || true
    git cat-file -e "${UPSTREAM_COMMIT}^{commit}" 2>/dev/null || die "upstream commit $UPSTREAM_COMMIT not found."
    log "pinning upstream to $UPSTREAM_COMMIT"
    git checkout --quiet --detach "$UPSTREAM_COMMIT"
fi
verify

cat <<EOT

DFormer is ready at $DFORMER_ROOT. Next:
  export RGBD_DATA_ROOT=/path/to/data      # NYUDepthv2/ and SUNRGBD/ in the DFormer layout
  python -m dat.train --host DFormerv2-S --preset DAT --pretrained <DFormerv2_Small_pretrained.pth> --out runs/dat
Datasets and released weights: https://huggingface.co/bbynku/DFormerv2  (datasets/*.zip, DFormerv2/*)
EOT
