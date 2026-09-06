#!/usr/bin/env python
"""Deformably register each subject's DIXON MRI to its ground-truth CT, for L_NGF.

    python scripts/register_mri.py --data_dir <BIC-MAC>/train --out_root mri_reg_ct

Only needed to reproduce Table 1 rows 3 and 4, the two models that use L_NGF. Every
other row trains without it. Roughly 20 minutes per subject on 8 CPU threads, so about
a day for all 75 — run it once and keep the output.

WHAT THIS PRODUCES, AND WHY IT IS TRAINING-ONLY

    <out_root>/<sub>/mri_in_reg.nii.gz     in-phase DIXON, warped onto the CT grid
    <out_root>/<sub>/mri_out_reg.nii.gz    out-phase, warped by the SAME transform

The fixed image is the GROUND-TRUTH CT. That makes this a training-only artefact by
construction — validation and test subjects have no CT to register against. It is not a
limitation here, because the registered MRI enters L_NGF as a TARGET and never as a
model input, so a model trained with it still reads only NAC-PET and the topogram at
inference and remains deployable.

WHAT WE LEARNED DOING THIS, so you do not repeat it

  * The affine stage is nearly a no-op: the released MRI already arrives grossly
    aligned from the rigid translation in the challenge preprocessing. What is left is
    DEFORMABLE, and largest in the head and skull. Hence SyNRA (rigid -> affine -> SyN)
    rather than an affine-only schedule.
  * The residual only shrinks against the CT. Registering the MRI to NAC-PET instead
    does not work: the NAC-PET anatomy sits at a different position, because the CT and
    the PET are acquired around an hour apart.
  * Registration as a MODEL INPUT is a trap. Registering the MRI and feeding it in gave
    us -7.8 % CT MAE with the ground-truth CT as the fixed image, but that operator
    cannot exist at test time; substituting a pseudo-CT made the train and test
    operators differ and the model got WORSE the longer it trained (-1.3 % at 15
    epochs, +17 % at 40). Using the MRI as a training-only NGF target, or feeding it raw
    and unregistered as extra channels, both sidestep this. The paper's models do the
    latter two and never the former.

Requires ANTsPy:  pip install antspyx
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def register_subject(sub: str, data_dir: Path, out_root: Path) -> bool:
    import ants

    features = data_dir / sub / "features"
    ct_path = data_dir / sub / "ct-label" / "ct.nii.gz"
    out = out_root / sub
    out.mkdir(parents=True, exist_ok=True)

    if (out / "mri_in_reg.nii.gz").exists():
        print(f"{sub}: already done", flush=True)
        return True

    if not ct_path.exists():
        print(f"{sub}: SKIPPED — no ground-truth CT, so there is nothing to register "
              f"against. This is expected for validation and test subjects.", flush=True)
        return False

    try:
        fixed = ants.image_read(str(ct_path))
        mri_in = ants.image_read(str(features / "mri_combined_in_phase.nii.gz"))
        mri_out = ants.image_read(str(features / "mri_combined_out_phase.nii.gz"))

        reg = ants.registration(
            fixed=fixed,
            moving=mri_in,
            type_of_transform="SyNRA",              # rigid -> affine -> SyN
            aff_metric="mattes",                    # mutual information: the images are
            syn_metric="mattes",                    # different modalities
        )
        # The in- and out-phase volumes are co-acquired, so the transform estimated on
        # the in-phase applies unchanged to the out-phase.
        transforms = reg["fwdtransforms"]
        ants.image_write(reg["warpedmovout"], str(out / "mri_in_reg.nii.gz"))
        ants.image_write(ants.apply_transforms(fixed, mri_out, transforms),
                         str(out / "mri_out_reg.nii.gz"))
        shutil.copy(transforms[-1], str(out / "mri2target_affine.mat"))
        print(f"{sub}: OK", flush=True)
        return True
    except Exception as exc:                                    # noqa: BLE001
        print(f"{sub}: FAILED — {type(exc).__name__}: {exc}", flush=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_dir", required=True,
                    help="BIC-MAC train/ directory (the subjects with ct-label/)")
    ap.add_argument("--out_root", default="mri_reg_ct",
                    help="output root; point $MRI_REG_ROOT here when training")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="specific subject ids; default is all of them")
    ap.add_argument("--threads", type=int, default=8, help="ITK threads per subject")
    args = ap.parse_args()

    # Must be set before ants (and thus ITK) is imported.
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    subjects = args.subjects or sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and (d / "features").is_dir()
    )

    print(f"{len(subjects)} subjects -> {out_root}  ({args.threads} threads each)")
    print("Roughly 20 min per subject. Run once and keep the output.\n")

    ok = sum(register_subject(s, data_dir, out_root) for s in subjects)
    print(f"\n{ok}/{len(subjects)} registered.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
