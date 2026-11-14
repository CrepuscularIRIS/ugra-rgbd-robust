"""Training presets: the arms compared in the paper, expressed as availability mixtures.

Every arm uses the same architecture, schedule and loss; only the depth that enters the network during
training differs. RGB-only is the reference model of Sec. 3.1: it receives the constant map in place of depth
in training *and* in testing.
"""
from __future__ import annotations

from .availability import Mixture

PRESETS: dict[str, Mixture] = {
    # conventional RGB-D training on complete inputs only (the "C" column of Table 1)
    "C": Mixture(1.0, 0.0, 0.0),
    # RGB-only baseline: depth is always the constant map (Sec. 3.1); evaluate with the missing condition
    "RGB_ONLY": Mixture(0.0, 1.0, 0.0),
    # the reported setting (Sec. 3.4)
    "DAT": Mixture(0.50, 0.25, 0.25),
    # Table 3 ablations. The paper names these rows but does not print their ratios; the two-condition
    # rows below assume an even split of the non-complete mass between the kept conditions -- check
    # against the training logs before quoting them as the paper's exact setting.
    "DAT_NO_MISSING": Mixture(0.50, 0.00, 0.50),  # "w/o fully missing": complete + partial only
    "DAT_NO_PARTIAL": Mixture(0.50, 0.50, 0.00),  # "w/o partial missing": plain modality dropout
    "DAT_75_125_125": Mixture(0.75, 0.125, 0.125),  # mixture-ratio ablation (Table 3, 5th row)
}


def get(name: str) -> Mixture:
    try:
        return PRESETS[name]
    except KeyError as e:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}") from e
