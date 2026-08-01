# shellcheck shell=bash
#
# Shared environment contract for every script in this repository.
#
# Sourced (never executed) by the run/eval scripts under scripts/ and experiments/.
# Everything here is overridable from the caller's environment, so nothing in this
# repository needs editing to run it on a different machine:
#
#   UGRA_ROOT         repo root                        (auto-detected by the caller)
#   DFORMER_ROOT      patched DFormer checkout         ($UGRA_ROOT/repos/DFormer)
#   GEMINIFUSION_ROOT optional second architecture     ($UGRA_ROOT/repos/GeminiFusion)
#   UGRA_CONDA_ENV    conda environment name           (dformer)
#   CONDA_SH          conda shell hook                 (auto-detected below)
#   UGRA_GPU          default CUDA device              (0)
#
# The paths recorded inside results/**/*.json are the ORIGINAL machine's and are left
# untouched on purpose: those files are frozen evidence, checksummed in
# results/FROZEN_20260728/FREEZE_MANIFEST.md. Rewriting them would break provenance.

# --- repo root ---------------------------------------------------------------
if [ -z "${UGRA_ROOT:-}" ]; then
    echo "env.sh: UGRA_ROOT is unset; source this from a script that sets it." >&2
    return 1 2>/dev/null || exit 1
fi
export UGRA_ROOT

# --- third-party checkouts ---------------------------------------------------
export DFORMER_ROOT="${DFORMER_ROOT:-$UGRA_ROOT/repos/DFormer}"
export GEMINIFUSION_ROOT="${GEMINIFUSION_ROOT:-$UGRA_ROOT/repos/GeminiFusion}"

# --- conda -------------------------------------------------------------------
export UGRA_CONDA_ENV="${UGRA_CONDA_ENV:-dformer}"
if [ -z "${CONDA_SH:-}" ]; then
    for _c in "${CONDA_EXE%/bin/conda}" "$HOME/miniconda3" "$HOME/anaconda3" \
              "$HOME/miniforge3" "$HOME/mambaforge" "/opt/conda" "/usr/local/conda"; do
        if [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ]; then
            CONDA_SH="$_c/etc/profile.d/conda.sh"
            break
        fi
    done
    unset _c
fi
export CONDA_SH="${CONDA_SH:-}"
if [ -z "$CONDA_SH" ] || [ ! -f "$CONDA_SH" ]; then
    echo "env.sh: could not locate conda.sh. Set CONDA_SH=/path/to/etc/profile.d/conda.sh" >&2
fi

# --- GPU ---------------------------------------------------------------------
export UGRA_GPU="${UGRA_GPU:-0}"

# --- invariants that are NOT optional ----------------------------------------
# fp16 is unstable on this architecture; every training run in the paper used fp32.
# See docs/TROUBLESHOOTING.md -> "Mixed precision".
export USE_AMP="${USE_AMP:-0}"

# --- sanity ------------------------------------------------------------------
if [ ! -d "$DFORMER_ROOT/models" ]; then
    echo "env.sh: DFormer not found at $DFORMER_ROOT" >&2
    echo "        run:  bash scripts/setup_dformer.sh" >&2
fi
