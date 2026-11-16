"""Package corruption families as drop-in training roots (DFormer layout).

    python scripts/make_variants.py --corrupt data/NYU_corrupt --base /data/datasets/rgbd/NYUDepthv2

Each complete family/severity becomes <corrupt>/variants/<FAM>_<SEV>/ holding:
    RGB/   -> absolute symlink to the base package RGB (uncorrupted)
    Label/ -> absolute symlink to the base Label
    Depth/ -> relative symlink to <corrupt>/<FAM>/<SEV>/Depth  (the corruption)
    E/     -> relative symlink to <corrupt>/<FAM>/<SEV>/E      (oracle, eval only)
    train.txt / test.txt  (copies, same ids as the base split)
Point any DFormer-style loader's data_root at the variant directory and it
trains/evaluates on corrupted depth with unchanged RGB, labels and split.
Writes <corrupt>/manifest.json (all variants + severity stats). Idempotent.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

N_EXPECTED = 1449


def _relink(link: Path, target: str) -> None:
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(target)


def _split_ids(base: Path) -> set[int]:
    ids = set()
    for split in ("train.txt", "test.txt"):
        for line in (base / split).read_text().splitlines():
            if line.strip():
                ids.add(int(line.strip().rsplit("/", 1)[-1].split(".")[0]))
    return ids


def package(corrupt: Path, base: Path, n_expected: int = N_EXPECTED) -> list[dict]:
    manifest = []
    expected_ids = _split_ids(base)
    for fam_dir in sorted(p for p in corrupt.iterdir() if p.is_dir() and p.name != "variants"):
        for sev_dir in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
            meta_f = sev_dir / "meta.json"
            depth = sev_dir / "Depth"
            if not meta_f.exists() or not depth.is_dir():
                continue
            meta = json.loads(meta_f.read_text())
            n = len(list(depth.glob("*.png")))
            if n != n_expected:
                print(f"skip partial {fam_dir.name}/{sev_dir.name}: {n}/{N_EXPECTED} frames")
                continue
            # Sol P2-9: counts alone can pass on a mixed partial rebuild; verify ID sets
            d_ids = {p.stem for p in depth.glob("*.png")}
            e_ids = {p.stem for p in (sev_dir / "E").glob("*.png")} if (sev_dir / "E").is_dir() else set()
            want = {str(i) for i in expected_ids}
            if d_ids != want or e_ids != want or meta.get("n") != n:
                print(f"skip inconsistent {fam_dir.name}/{sev_dir.name}: "
                      f"depth {len(d_ids)} / E {len(e_ids)} / meta {meta.get('n')} vs {n} expected ids")
                continue
            vname = f"{fam_dir.name}_{sev_dir.name}"
            vroot = corrupt / "variants" / vname
            vroot.mkdir(parents=True, exist_ok=True)
            _relink(vroot / "RGB", str(base.resolve() / "RGB"))
            _relink(vroot / "Label", str(base.resolve() / "Label"))
            _relink(vroot / "Depth", os.path.relpath(depth, vroot))
            _relink(vroot / "E", os.path.relpath(sev_dir / "E", vroot))
            for split in ("train.txt", "test.txt"):
                shutil.copyfile(base / split, vroot / split)
            manifest.append({
                "name": vname, "family": fam_dir.name, "severity": sev_dir.name,
                "params": meta.get("params"), "n_frames": meta.get("n", n),
                "mean_e_frac": meta.get("mean_e_frac"), "mean_m_eff": meta.get("mean_m_eff"),
            })
            print(f"variant {vname}: e_frac={meta.get('mean_e_frac')} m_eff={meta.get('mean_m_eff')}")
    (corrupt / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _self_check(tmp: Path) -> None:
    base = tmp / "base"
    (base / "RGB").mkdir(parents=True)
    (base / "Label").mkdir()
    (base / "train.txt").write_text("RGB/1.jpg\nRGB/2.jpg\n")
    (base / "test.txt").write_text("RGB/3.jpg\n")
    corrupt = tmp / "corrupt"
    d = corrupt / "F1" / "a05" / "Depth"
    d.mkdir(parents=True)
    (corrupt / "F1" / "a05" / "E").mkdir()
    for i in (1, 2, 3):  # ids must match the split exactly (P2-9 completeness rule)
        (d / f"{i}.png").write_bytes(b"")
        (corrupt / "F1" / "a05" / "E" / f"{i}.png").write_bytes(b"")
    (corrupt / "F1" / "a05" / "meta.json").write_text(
        json.dumps({"params": {"a": 0.05}, "n": 3, "mean_e_frac": 0.1, "mean_m_eff": 0.02}))
    partial = corrupt / "F2" / "a1" / "Depth"
    partial.mkdir(parents=True)
    (partial / "0.png").write_bytes(b"")
    man = package(corrupt, base, n_expected=3)
    assert [m["name"] for m in man] == ["F1_a05"], man
    v = corrupt / "variants" / "F1_a05"
    assert (v / "RGB").is_symlink() and (v / "Depth").is_symlink()
    assert (v / "train.txt").read_text() == "RGB/1.jpg\nRGB/2.jpg\n"
    assert len(list((v / "Depth").iterdir())) == 3
    man2 = package(corrupt, base, n_expected=3)  # idempotent
    assert len(man2) == 1


if __name__ == "__main__":
    import tempfile

    ap = argparse.ArgumentParser()
    ap.add_argument("--corrupt", type=Path, default=Path("data/NYU_corrupt"))
    ap.add_argument("--base", type=Path, default=Path("/data/datasets/rgbd/NYUDepthv2"))
    args = ap.parse_args()
    if args.corrupt.exists():
        package(args.corrupt, args.base)
    else:
        with tempfile.TemporaryDirectory() as td:
            _self_check(Path(td))
        print("make_variants self-check OK")
