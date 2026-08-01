#!/usr/bin/env bash
#
# Fetch the upstream DFormer codebase and apply this project's method patch.
#
# The patch in patches/dformer-ugra.patch is what turns stock DFormerv2 into the
# instrument this paper uses: the depth-authority dial, the per-stage depth gate,
# the positional-alpha control and the training hooks. It is generated against, and
# verified to apply cleanly to, upstream commit UPSTREAM_COMMIT below.
#
# Usage:
#   bash scripts/setup_dformer.sh            # clone + patch
#   bash scripts/setup_dformer.sh --check    # verify an existing checkout is patched
#
set -euo pipefail

# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

UPSTREAM_URL="https://github.com/VCIP-RGBD/DFormer.git"
UPSTREAM_COMMIT="23cd8f833df12c74af45bc671e75891e48f8ff1b"
PATCH="$UGRA_ROOT/patches/dformer-ugra.patch"

# Files the patch touches. Used by --check to tell "patched" from "stock".
PATCHED_FILES=(
    models/builder.py
    models/encoders/DFormerv2.py
    utils/train.py
    utils/dataloader/dataloader.py
    utils/engine/engine.py
)

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[ -f "$PATCH" ] || die "patch not found: $PATCH"

# --------------------------------------------------------------------------- check
if [ "${1:-}" = "--check" ]; then
    [ -d "$DFORMER_ROOT" ] || die "no checkout at $DFORMER_ROOT"
    cd "$DFORMER_ROOT"
    if git apply --reverse --check "$PATCH" 2>/dev/null; then
        log "PATCHED — $DFORMER_ROOT carries the method patch."
        exit 0
    elif git apply --check "$PATCH" 2>/dev/null; then
        die "STOCK — $DFORMER_ROOT is unpatched. Run without --check to apply."
    else
        die "UNKNOWN — the checkout matches neither stock nor patched state.
     It may be locally modified, or at a different upstream commit than
     $UPSTREAM_COMMIT. Inspect with: cd $DFORMER_ROOT && git status"
    fi
fi

# --------------------------------------------------------------------------- clone
# NB: test the work tree, not `-d .git` — in a git worktree or a submodule `.git`
# is a FILE, and a `-d` test sends us down the clone path onto a non-empty directory.
if git -C "$DFORMER_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log "existing checkout at $DFORMER_ROOT — reusing it"
else
    log "cloning $UPSTREAM_URL -> $DFORMER_ROOT"
    mkdir -p "$(dirname "$DFORMER_ROOT")"
    git clone "$UPSTREAM_URL" "$DFORMER_ROOT"
fi

cd "$DFORMER_ROOT"

# --------------------------------------------------------------------------- patch
# Order matters: an ALREADY-PATCHED checkout is a dirty tree by definition, so the
# patched test has to run before the dirty-tree guard or re-running this script
# after a successful setup fails on its own handiwork.
if git apply --reverse --check "$PATCH" 2>/dev/null; then
    log "already patched — nothing to do"
else
    log "pinning upstream to $UPSTREAM_COMMIT"
    git fetch --all --tags --quiet || true
    if ! git cat-file -e "${UPSTREAM_COMMIT}^{commit}" 2>/dev/null; then
        die "upstream commit $UPSTREAM_COMMIT not found in $DFORMER_ROOT."
    fi
    if ! git diff --quiet || ! git diff --cached --quiet; then
        die "the checkout at $DFORMER_ROOT has uncommitted changes that are not this
     project's patch; refusing to touch it. Stash or remove them, then re-run."
    fi
    git checkout --quiet --detach "$UPSTREAM_COMMIT"

    log "applying patches/dformer-ugra.patch"
    git apply --check "$PATCH" || die "patch does not apply cleanly at $UPSTREAM_COMMIT"
    git apply "$PATCH"
    log "patch applied"
fi

# --------------------------------------------------------------------------- verify
for f in "${PATCHED_FILES[@]}"; do
    [ -f "$f" ] || die "expected patched file missing: $f"
done
git apply --reverse --check "$PATCH" 2>/dev/null \
    || die "post-apply verification failed"

cat <<EOF

$(log "DFormer is ready at $DFORMER_ROOT")

Next steps:
  1. Datasets and pretrained weights   ->  docs/DATASETS.md
  2. Reproduce the headline table      ->  docs/REPRODUCTION.md
  3. Verify the neutral element        ->  python diagnostics/single_path_invariance.py

EOF
