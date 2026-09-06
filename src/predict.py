"""Predict a pseudo-CT from one or more MuNet checkpoints.

One model (any single row of Table 1):

    python src/predict.py \
        --features_dir  <subject>/features \
        --output_ct     out/ct.nii.gz \
        --checkpoints   checkpoints/row7_munet4_lor_refine.pth

The submitted fusion (Table 1 row 9) — two models, blended 0.3 / 0.7:

    python src/predict.py \
        --features_dir  <subject>/features \
        --output_ct     out/ct.nii.gz \
        --checkpoints   checkpoints/row4_munet2_ngf_lor.pth \
                        checkpoints/row7_munet4_lor_refine.pth \
        --weights       0.3 0.7

WHY FUSION IS DONE HERE AND NOT BY AVERAGING PET IMAGES. The challenge container emits
a single pseudo-CT and the organizers run the reconstruction themselves, so any
combination of models has to collapse into one volume BEFORE it is scored — we cannot
submit one model's CT with another model's PET. The blend is therefore a fixed convex
combination in Hounsfield Units,

    CT_fused = sum_i w_i * CT_i ,   w_i > 0 ,  sum_i w_i = 1

with the weights fixed in advance rather than learned. We tried learning them: an
8k-parameter per-voxel gate appeared to gain 4.5 %, but that was leakage — both members
had trained on all 75 subjects, so in-sample one member looks much better than it is
out-of-sample. Pinning the gate's spatial mean to the validated weight collapsed its
learned variation to exactly zero.

MEMORY. Members are accumulated one at a time and each is freed before the next loads,
so peak memory is one model plus two volumes regardless of how many members you blend.

THE OUTPUT GRID IS NOT NEGOTIABLE. The saved CT copies the affine and header of
`features/nacpet.nii.gz`. The evaluation code and the reconstruction pipeline both
assume the pseudo-CT sits on exactly that grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from data import get_predict_transforms, get_subject_features
from munet import build_model


def load_meta(checkpoint: Path, override: Path | None = None) -> dict:
    """Read the architecture sidecar written by train.py.

    The sidecar makes a checkpoint self-describing: it records the input channels, the
    patch size and which MuNet additions were enabled, so inference never has to be
    told which config produced the weights. Rebuilding with the wrong flags would fail
    to load (the aux heads and attention blocks carry real parameters).
    """
    path = override or Path(str(checkpoint) + ".meta.json")
    if not path.exists():
        raise FileNotFoundError(
            f"No meta sidecar at {path}. Every released checkpoint ships one; if you "
            f"trained this yourself, train.py writes it next to the weights."
        )
    return json.loads(path.read_text())


def predict_one(checkpoint: Path, features_dir: str, device: str,
                overlap: float, meta_override: Path | None = None) -> np.ndarray:
    """Run one model over a subject and return its pseudo-CT in Hounsfield Units."""
    meta = load_meta(checkpoint, meta_override)
    input_keys = list(meta["input_keys"])
    patch_size = tuple(meta["patch_size"])

    model = build_model(meta).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    subject = get_subject_features(features_dir)
    data = get_predict_transforms(input_keys=input_keys)(subject)
    x = data["input"].unsqueeze(0).to(device)

    # MIXED PRECISION IS PART OF THE RECIPE, NOT AN OPTIMIZATION. The released
    # checkpoints were trained under autocast and every number in the paper was
    # produced by running inference under it too. Forcing fp32 here changes the output
    # by up to ~15 HU (about 0.3 HU on average), which is small but is not what was
    # scored. Keep the autocast.
    #
    # `device="cpu"` keeps the assembled output volume in host memory while
    # `sw_device` does the compute on the GPU. A 512x512x531 float volume is 0.5 GB,
    # and with gaussian blending MONAI holds more than one — on a 24 GB card that is
    # the difference between fitting and not.
    autocast_ctx = torch.amp.autocast("cuda" if device == "cuda" else "cpu")
    with torch.no_grad(), autocast_ctx:
        pred = sliding_window_inference(
            x, roi_size=patch_size, sw_batch_size=1, predictor=model,
            overlap=overlap, mode="gaussian",
            sw_device=device, device="cpu",
        )

    # eval() returns the CT tensor directly; the deep-supervision heads are training-only.
    ct01 = pred[0, 0].float().cpu().numpy()
    a, b = meta["hu_decode"]["a"], meta["hu_decode"]["b"]
    del model, pred, x
    if device == "cuda":
        torch.cuda.empty_cache()
    return ct01 * a + b


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Predict a pseudo-CT from one or more MuNet checkpoints.")
    ap.add_argument("--features_dir", required=True,
                    help="a subject's features/ directory")
    ap.add_argument("--output_ct", required=True,
                    help="where to write ct.nii.gz (Hounsfield Units)")
    ap.add_argument("--checkpoints", nargs="+", required=True, metavar="CKPT",
                    help="one checkpoint, or several to blend")
    ap.add_argument("--weights", nargs="+", type=float, default=None, metavar="W",
                    help="blend weights, one per checkpoint; must sum to 1. "
                         "Default: equal weights.")
    ap.add_argument("--meta", nargs="+", default=None, metavar="META",
                    help="override the meta sidecar path(s), one per checkpoint")
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="sliding-window overlap (default 0.5). Higher is slightly "
                         "better and much slower.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpts = [Path(c) for c in args.checkpoints]
    metas = [Path(m) for m in args.meta] if args.meta else [None] * len(ckpts)
    if len(metas) != len(ckpts):
        raise ValueError(f"got {len(metas)} --meta paths for {len(ckpts)} checkpoints")

    if args.weights is None:
        weights = [1.0 / len(ckpts)] * len(ckpts)
    else:
        weights = list(args.weights)
        if len(weights) != len(ckpts):
            raise ValueError(f"got {len(weights)} weights for {len(ckpts)} checkpoints")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1, got {sum(weights)}")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative (this is a convex blend)")

    print(f"Device: {args.device}")
    print("Blend:  " + " + ".join(f"{w:g} * {c.name}" for w, c in zip(weights, ckpts)))

    fused = None
    for i, (ckpt, w, meta_path) in enumerate(zip(ckpts, weights, metas), start=1):
        print(f"[{i}/{len(ckpts)}] {ckpt.name}  (weight {w:g})")
        ct_hu = predict_one(ckpt, args.features_dir, args.device, args.overlap, meta_path)
        fused = w * ct_hu if fused is None else fused + w * ct_hu

    # Copy the affine AND header from NAC-PET so the output lands on exactly the grid
    # the evaluation and reconstruction code expect.
    ref = nib.load(get_subject_features(args.features_dir)["nacpet"])
    out_path = Path(args.output_ct)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(fused.astype(np.float32), ref.affine, ref.header), out_path)

    print(f"Saved: {out_path}  shape={fused.shape}  "
          f"HU [{fused.min():.0f}, {fused.max():.0f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
