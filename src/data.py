"""Dataset listing and MONAI transform pipelines for training and inference.

The BIC-MAC release lays each subject out as

    <subject>/features/     nacpet, topogram, mri_combined_{in,out}_phase, metadata.json
    <subject>/ct-label/     ct, body_seg, organ_seg, prediction_mask        (train only)

All feature volumes are pre-resampled onto the CT grid, 512 x 512 x 531 at
1.52 x 1.52 x 2.00 mm, so no resampling is needed anywhere in this file. The topogram
is a 2D coronal projection stored as (X, 1, Z) and is tiled across Y — see
`BroadcastTopogramd`.

WHAT THE MODEL SEES vs WHAT THE LOSS SEES. `input_keys` are concatenated into the
`input` tensor the network reads. `prediction_mask`, `organ_seg` and the registered
MRI are loaded and cropped in lockstep with the CT patch, but they are ground-truth
volumes used only to build the loss — the network never sees them, which is what keeps
the model deployable on features/-only validation and test data.

REGISTERED MRI. `L_NGF` compares the predicted CT against a DEFORMABLY REGISTERED
DIXON MRI, which is not part of the released dataset — you produce it once with
`scripts/register_mri.py` (ANTs SyNRA, fixed image = the ground-truth CT). It is
therefore training-only by construction: validation and test subjects have no CT to
register against. That is fine, because the MRI enters `L_NGF` as a target and never as
a model input. Set `$MRI_REG_ROOT` to point elsewhere than `<repo>/mri_reg_ct`.
Only the two configs that switch `L_NGF` on need it.
"""

from __future__ import annotations

import json
import os

from monai.transforms import (
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    SpatialPadd,
)

# Native acquisition spacing (X, Y, Z) in mm. Used by the line-integral loss.
NATIVE_SPACING = (1.52, 1.52, 2.0)

# Where scripts/register_mri.py writes the deformably registered DIXON MRI used by
# L_NGF. Lives outside the dataset so the released data stays pristine.
_MRI_REG_ROOT = os.environ.get(
    "MRI_REG_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mri_reg_ct"),
)

# CT is range-normalized from these Hounsfield Units to [0, 1] for training;
# predict.py inverts with HU = x * 3000 - 1000.
CT_HU_MIN, CT_HU_MAX = -1000, 2000


# ---------------------------------------------------------------------------
# Subject listing
# ---------------------------------------------------------------------------

def get_subject_features(features_dir: str) -> dict:
    """Every model input available for one subject, plus its demographic metadata.

    The registered-MRI paths are added unconditionally as strings; nothing is read
    from disk unless a transform actually lists those keys, so subjects without a
    registration cost nothing.
    """
    paths = {
        "nacpet":                 os.path.join(features_dir, "nacpet.nii.gz"),
        "topogram":               os.path.join(features_dir, "topogram.nii.gz"),
        "mri_combined_in_phase":  os.path.join(features_dir, "mri_combined_in_phase.nii.gz"),
        "mri_combined_out_phase": os.path.join(features_dir, "mri_combined_out_phase.nii.gz"),
        "mri_face_mask":          os.path.join(features_dir, "mri_face_mask.nii.gz"),
    }
    sub_id = os.path.basename(os.path.dirname(features_dir))
    paths["mri_in_reg"] = os.path.join(_MRI_REG_ROOT, sub_id, "mri_in_reg.nii.gz")
    paths["mri_out_reg"] = os.path.join(_MRI_REG_ROOT, sub_id, "mri_out_reg.nii.gz")
    with open(os.path.join(features_dir, "metadata.json")) as f:
        metadata = json.load(f)
    return {**paths, **metadata}


def get_subject_labels(ct_label_dir: str) -> dict:
    """Ground-truth volumes. Training only — validation and test ship none of these.

    `prediction_mask` is the body minus the face and the scanner bed. Those two
    regions are overwritten with ground truth by the reconstruction pipeline before
    it projects, so nothing a model predicts there can affect any metric, and
    penalising them only wastes capacity.
    """
    return {
        "ct":              os.path.join(ct_label_dir, "ct.nii.gz"),
        "body_seg":        os.path.join(ct_label_dir, "body_seg.nii.gz"),
        "organ_seg":       os.path.join(ct_label_dir, "organ_seg.nii.gz"),
        "prediction_mask": os.path.join(ct_label_dir, "prediction_mask.nii.gz"),
    }


def get_dataset(data_dir: str) -> list[dict]:
    """One dict per subject, combining features and labels."""
    subjects = []
    for sub in sorted(os.listdir(data_dir)):
        subject_dir = os.path.join(data_dir, sub)
        if not os.path.isdir(subject_dir):
            continue
        subject = get_subject_features(os.path.join(subject_dir, "features"))
        subject.update(get_subject_labels(os.path.join(subject_dir, "ct-label")))
        subject["subject_id"] = sub
        subjects.append(subject)
    return subjects


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class BroadcastTopogramd(MapTransform):
    """Tile the topogram's singleton Y axis so it stacks with the 3D volumes.

    The topogram is a 2D coronal projection on the CT grid, so it is (C, X, 1, Z)
    after `EnsureChannelFirstd` while every other modality is (C, X, Y, Z). Tiling
    across Y gives every Y plane the same projection, which is the natural way to
    expose a projection image to a 3D convolution. X and Z already match, so this is
    an exact repeat rather than an interpolation.
    """

    def __init__(self, keys, ref_key: str = "nacpet", allow_missing_keys: bool = True):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.ref_key = ref_key

    def __call__(self, data):
        d = dict(data)
        ref_y = d[self.ref_key].shape[2]              # (C, X, Y, Z) -> Y at index 2
        for k in self.key_iterator(d):
            t = d[k]
            if t.shape[2] != ref_y:
                if ref_y % t.shape[2] != 0:
                    raise ValueError(
                        f"BroadcastTopogramd: reference Y={ref_y} is not a multiple "
                        f"of {k} Y={t.shape[2]}"
                    )
                reps = [1] * t.ndim
                reps[2] = ref_y // t.shape[2]
                d[k] = t.repeat(*reps)
        return d


def _normalize_for(key: str):
    """Per-modality normalization, shared by training and inference.

    Edit this function — not the pipelines below — when adding a modality, so the two
    paths cannot drift apart.
    """
    if key == "nacpet":
        # Divide by the standard deviation with the mean held at 0. NAC-PET is
        # non-negative and its zero is meaningful, so we do not re-centre it.
        return NormalizeIntensityd(keys=[key], nonzero=False, channel_wise=True,
                                   subtrahend=[0])
    # MRI phases and the topogram: full per-volume z-score. Their absolute intensities
    # carry no calibration worth preserving.
    return NormalizeIntensityd(keys=[key], nonzero=False, channel_wise=True)


def get_train_transforms(
    patch_size,
    num_samples: int = 2,
    input_keys=("nacpet", "topogram"),
    extra_label_keys=("organ_seg",),
):
    """Training pipeline: load, normalize, concatenate, then sample random patches.

    `extra_label_keys` are ground-truth volumes the loss needs but the model never
    sees. They are cropped by the SAME `RandSpatialCropSamplesd` as the CT so they stay
    voxel-aligned with the patch the loss compares against, and they are deliberately
    not normalized (their values are integer class labels).
    """
    input_keys = list(input_keys)
    extra_label_keys = list(extra_label_keys)
    image_keys = list(dict.fromkeys(
        input_keys + ["ct", "prediction_mask"] + extra_label_keys
    ))

    steps = [
        LoadImaged(keys=image_keys),
        # channel_dim="no_channel" is explicit because train.py disables MetaTensor
        # tracking for speed, leaving no metadata to infer the channel from.
        EnsureChannelFirstd(keys=image_keys, channel_dim="no_channel"),
    ]

    if "topogram" in input_keys:
        steps.append(BroadcastTopogramd(keys=["topogram"], ref_key="nacpet"))

    steps.extend(_normalize_for(k) for k in input_keys)

    # CT label -> [0, 1]. predict.py inverts this to Hounsfield Units.
    steps.append(ScaleIntensityRanged(
        keys=["ct"], a_min=CT_HU_MIN, a_max=CT_HU_MAX, b_min=0.0, b_max=1.0, clip=True))

    steps.append(ConcatItemsd(keys=input_keys, name="input"))

    crop_keys = list(dict.fromkeys(["input", "ct", "prediction_mask"] + extra_label_keys))
    # Pad first so a volume smaller than the patch still yields a full patch. A no-op
    # for BIC-MAC's 512^3 grid, but it makes the pipeline safe for other data.
    steps.append(SpatialPadd(keys=crop_keys, spatial_size=patch_size, mode="constant"))
    steps.append(RandSpatialCropSamplesd(
        keys=crop_keys, roi_size=patch_size, random_size=False, num_samples=num_samples))
    steps.append(EnsureTyped(keys=crop_keys))
    return Compose(steps)


def get_predict_transforms(input_keys=("nacpet", "topogram")):
    """Inference pipeline: load, normalize, concatenate. No labels, no cropping.

    Sliding-window inference handles the patching, so the whole volume is normalized
    once — which also means normalization statistics are computed over the full volume
    exactly as in training.
    """
    input_keys = list(input_keys)
    steps = [
        LoadImaged(keys=input_keys),
        EnsureChannelFirstd(keys=input_keys),
    ]
    if "topogram" in input_keys:
        steps.append(BroadcastTopogramd(keys=["topogram"], ref_key="nacpet"))
    steps.extend(_normalize_for(k) for k in input_keys)
    steps.append(ConcatItemsd(keys=input_keys, name="input"))
    steps.append(EnsureTyped(keys=["input"]))
    return Compose(steps)
