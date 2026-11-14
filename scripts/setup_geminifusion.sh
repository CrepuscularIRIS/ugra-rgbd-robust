#!/usr/bin/env bash
#
# Fetch the upstream GeminiFusion codebase and apply this project's DAT patch.
#
# patches/geminifusion-dat.patch adds one flag, --dat-preset, to upstream's main.py. With the flag absent
# (the default) behaviour is unchanged; with it, only the depth input of each training batch is replaced by
# D~ = M*D + (1-M)*c (c = 0, see dat/hosts/geminifusion.py). The loss, the parameters and the RGB input are
# untouched. The patch is generated against, and verified to apply cleanly to, UPSTREAM_COMMIT below.
#
# Usage:
#   bash scripts/setup_geminifusion.sh            # clone + pin + patch
#   bash scripts/setup_geminifusion.sh --check    # verify an existing checkout is patched
#
# The host adapter (dat/hosts/geminifusion.py) works on the STOCK or the PATCHED checkout: it never imports
# main.py. Only training with --dat-preset needs the patch.
#
set -euo pipefail

# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

UPSTREAM_URL="https://github.com/JiaDingCN/GeminiFusion.git"
UPSTREAM_COMMIT="1f102f6451ec7cd4546278ef2022f639f979c60b"
PATCH="$UGRA_ROOT/patches/geminifusion-dat.patch"

# Files the patch touches. Used to verify the result.
PATCHED_FILES=(main.py)

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[ -f "$PATCH" ] || die "patch not found: $PATCH"

# --------------------------------------------------------------------------- check
if [ "${1:-}" = "--check" ]; then
    [ -d "$GEMINIFUSION_ROOT" ] || die "no checkout at $GEMINIFUSION_ROOT"
    cd "$GEMINIFUSION_ROOT"
    if git apply --reverse --check "$PATCH" 2>/dev/null; then
        log "PATCHED — $GEMINIFUSION_ROOT carries the DAT patch."
        exit 0
    elif git apply --check "$PATCH" 2>/dev/null; then
        die "STOCK — $GEMINIFUSION_ROOT is unpatched. Run without --check to apply."
    else
        die "UNKNOWN — the checkout matches neither stock nor patched state.
     It may be locally modified, or at a different upstream commit than
     $UPSTREAM_COMMIT. Inspect with: cd $GEMINIFUSION_ROOT && git status"
    fi
fi

# --------------------------------------------------------------------------- clone
# NB: test the work tree, not `-d .git` (a worktree/submodule has a .git FILE).
if git -C "$GEMINIFUSION_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log "existing checkout at $GEMINIFUSION_ROOT — reusing it"
else
    log "cloning $UPSTREAM_URL -> $GEMINIFUSION_ROOT"
    mkdir -p "$(dirname "$GEMINIFUSION_ROOT")"
    git clone "$UPSTREAM_URL" "$GEMINIFUSION_ROOT"
fi

cd "$GEMINIFUSION_ROOT"

# --------------------------------------------------------------------------- patch
# An already-patched checkout is a dirty tree, so test "patched" before the dirty-tree guard.
if git apply --reverse --check "$PATCH" 2>/dev/null; then
    log "already patched — nothing to do"
else
    log "pinning upstream to $UPSTREAM_COMMIT"
    git cat-file -e "${UPSTREAM_COMMIT}^{commit}" 2>/dev/null || git fetch --all --tags --quiet || true
    git cat-file -e "${UPSTREAM_COMMIT}^{commit}" 2>/dev/null \
        || die "upstream commit $UPSTREAM_COMMIT not found in $GEMINIFUSION_ROOT."
    if ! git diff --quiet || ! git diff --cached --quiet; then
        die "the checkout at $GEMINIFUSION_ROOT has uncommitted changes that are not this
     project's patch; refusing to touch it. Stash or remove them, then re-run."
    fi
    git checkout --quiet --detach "$UPSTREAM_COMMIT"

    log "applying patches/geminifusion-dat.patch"
    git apply --check "$PATCH" || die "patch does not apply cleanly at $UPSTREAM_COMMIT"
    git apply "$PATCH"
    log "patch applied"
fi

# --------------------------------------------------------------------------- verify
for f in "${PATCHED_FILES[@]}"; do
    [ -f "$f" ] || die "expected patched file missing: $f"
done
git apply --reverse --check "$PATCH" 2>/dev/null || die "post-apply verification failed"
grep -q -- '--dat-preset' main.py || die "main.py lacks the --dat-preset flag after patching"

cat <<EOF

$(log "GeminiFusion is ready at $GEMINIFUSION_ROOT")

Notes:
  * GeminiFusion has no single-stream mode (num_parallel = 2), but the RGB-only baseline of the paper (constant depth
    in training and testing) works: --dat-preset RGB_ONLY.
  * Train: python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py \\
             --backbone mit_b3 --dataset nyudv2 -c <name> --dat-preset DAT   (see upstream README)
  * Released NYU checkpoints load through dat.hosts.get_host("GeminiFusion-B3").load_weights(...)
EOF
