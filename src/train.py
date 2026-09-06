"""Train one MuNet model from a YAML config.

    python src/train.py --config configs/row4_munet2_ngf_lor.yaml

Every model in Table 1 of the paper is this script plus one config file. The config
decides three things and nothing else: which input channels the network reads, which
loss terms are switched on, and the training schedule.

TWO THINGS WORTH KNOWING BEFORE YOU CHANGE ANYTHING.

1. There is no optimizer/epoch resume. `init_checkpoint` loads WEIGHTS only, and the
   cosine schedule always restarts from `learning_rate`. Continuing a run is therefore
   a warm restart (SGDR-style), not a resume — which is exactly how the multi-stage
   models in the paper were built, so it is faithful rather than a limitation. To
   continue an interrupted run, point `init_checkpoint` at its `last_model.pth` and set
   `epochs` to the number of epochs REMAINING.

2. Checkpoint selection. Two files are written every epoch:
       <output_dir>/checkpoints/best_model.pth   saved when the TRAINING loss improves
       <output_dir>/checkpoints/last_model.pth   overwritten unconditionally
   With `holdout_subjects: []` (all 75 subjects used for gradient updates, as in the
   paper) "best" is selection on the training loss, which in our experiments was a poor
   predictor of validation CT. The released checkpoints are LAST-epoch weights. Prefer
   `last_model.pth` unless you have a real held-out split.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.data import CacheDataset, DataLoader
from monai.data.meta_obj import set_track_meta
from tqdm import tqdm

from anper import AnatomicalPerceptionLoss
from data import NATIVE_SPACING, get_dataset, get_train_transforms
from losses import gradient_l1, make_lor_loss, make_mu_l1, ngf_loss, plain_l1
from munet import build_model

# Soft-tissue band in [0,1] CT space, used to gate L_NGF. DIXON MRI is informative for
# soft tissue and carries essentially no bone or air signal, so the term is restricted
# to roughly [-200, 300] HU.
SOFT_TISSUE_LO = (-200.0 + 1000.0) / 3000.0
SOFT_TISSUE_HI = (300.0 + 1000.0) / 3000.0


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    repo = Path(__file__).resolve().parent.parent
    for key in ("data_dir", "output_dir", "init_checkpoint"):
        if cfg.get(key):
            p = Path(cfg[key])
            cfg[key] = str(p if p.is_absolute() else (repo / p).resolve())
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_warm_start(model: torch.nn.Module, path: str) -> None:
    """Load weights from a previous stage, tolerating shape mismatches.

    `strict=False` plus an explicit shape filter, so that changing the number of input
    channels between stages (2 -> 4, say) drops only the stem convolution and keeps
    everything else, instead of failing outright.
    """
    sd = torch.load(path, map_location="cpu", weights_only=True)
    own = model.state_dict()
    keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    dropped = sorted(set(sd) - set(keep))
    missing, unexpected = model.load_state_dict(keep, strict=False)
    print(f"Warm start from: {path}")
    print(f"   loaded={len(keep)}  dropped(shape/absent)={dropped or 'none'}")
    print(f"   missing={len(missing)}  unexpected={len(unexpected)}")


def write_meta_sidecar(cfg: dict, checkpoint_path: Path) -> None:
    """Record the architecture next to the weights.

    predict.py rebuilds the network from this file, so a checkpoint is self-describing
    and you never have to remember which config produced it.
    """
    meta = {
        "model_type": "munet",
        "input_keys": list(cfg["input_keys"]),
        "in_channels": int(cfg["in_channels"]),
        "base_channels": int(cfg.get("base_channels", 32)),
        "bottleneck_dropout": float(cfg.get("bottleneck_dropout", 0.2)),
        "use_attention_skips": bool(cfg.get("use_attention_skips", False)),
        "encoder_attention": bool(cfg.get("encoder_attention", False)),
        "attn_heads": int(cfg.get("attn_heads", 8)),
        "deep_supervision": bool(cfg.get("deep_supervision", False)),
        "patch_size": list(cfg["patch_size"]),
        "hu_decode": {"a": 3000, "b": -1000},
    }
    Path(str(checkpoint_path) + ".meta.json").write_text(json.dumps(meta, indent=2))


def build_losses(cfg: dict, device: str):
    """Instantiate the loss terms this config switches on.

    Every term is opt-in through its weight, so any subset can be run. `loss_type`
    picks the base reconstruction loss; the other four are added when their weight is
    greater than zero.
    """
    loss_type = str(cfg.get("loss_type", "mu_l1"))
    if loss_type == "l1":
        base = plain_l1                      # the organizers' baseline objective
    elif loss_type == "mu_l1":
        base = make_mu_l1(
            bone_weight=float(cfg.get("bone_weight", 3.0)),
            bone_hu=float(cfg.get("bone_hu", 300.0)),
            brain_weight=float(cfg.get("brain_weight", 3.0)),
            organ_weight=float(cfg.get("organ_weight", 2.0)),
            brain_label=int(cfg.get("brain_label", 90)),
            organ_labels=tuple(cfg.get("organ_labels", (1, 5, 7, 51))),
        )
    else:
        raise ValueError(f"loss_type must be 'mu_l1' or 'l1', got {loss_type!r}")

    lor = None
    if float(cfg.get("lor_loss_weight", 0.0)) > 0:
        lor = make_lor_loss(
            n_angles=int(cfg.get("lor_n_angles", 4)),
            fwhm_mm=float(cfg.get("lor_fwhm_mm", 4.0)),
            spacing=NATIVE_SPACING,
            organ_labels=tuple(cfg.get("organ_labels", (1, 5, 7, 51))),
            min_lines=int(cfg.get("lor_min_lines", 32)),
            include_body=bool(cfg.get("lor_include_body", True)),
        )
        x, y = cfg["patch_size"][0], cfg["patch_size"][1]
        if min(x, y) < 384:
            print(f"WARNING: L_LOR is on but the patch is {cfg['patch_size']}. A line "
                  f"integral is only meaningful if the line crosses the whole body; "
                  f"{min(x, y)} voxels spans {min(x, y) * NATIVE_SPACING[0]:.0f} mm "
                  f"against a 400-500 mm body. Use a 448x448x48 slab.")

    anper = None
    if float(cfg.get("anper_loss_weight", 0.0)) > 0:
        anper = AnatomicalPerceptionLoss(
            model_folder=cfg["anper_model_folder"],
            layers=tuple(cfg.get("anper_layers",
                                 ("decoder.stages.0", "decoder.stages.1", "decoder.stages.2"))),
            device=device,
        )
        print(f"L_AnPer teacher: {cfg['anper_model_folder']}")

    return base, lor, anper


def compute_loss(pred, batch, cfg, base_loss, lor, anper, mask, seg, device):
    """Weighted sum of the loss terms, plus a per-term dict for logging."""
    ct_pred = pred["ct"] if isinstance(pred, dict) else pred
    gt_ct = batch["ct"].to(device)
    parts = {}

    total = base_loss(ct_pred, gt_ct, mask, seg)
    parts["base"] = float(total.detach())

    w = float(cfg.get("grad_loss_weight", 0.0))
    if w > 0:
        l = gradient_l1(ct_pred, gt_ct, mask)
        total = total + w * l
        parts["grad"] = float(l.detach())

    w = float(cfg.get("anper_loss_weight", 0.0))
    if anper is not None and w > 0:
        l = anper(ct_pred, gt_ct)
        total = total + w * l
        parts["anper"] = float(l.detach())

    w = float(cfg.get("ngf_loss_weight", 0.0))
    if w > 0 and torch.is_tensor(batch.get("mri_in_reg")):
        # Gate to soft tissue inside the prediction mask, where the MRI is informative.
        soft = (gt_ct > SOFT_TISSUE_LO) & (gt_ct < SOFT_TISSUE_HI)
        gate = mask & soft
        eps_rel = float(cfg.get("ngf_eps_rel", 0.1))
        smooth = float(cfg.get("ngf_smooth", 1.0))
        l = 0.5 * (
            ngf_loss(ct_pred, batch["mri_in_reg"].to(device), gate, eps_rel, smooth)
            + ngf_loss(ct_pred, batch["mri_out_reg"].to(device), gate, eps_rel, smooth)
        )
        total = total + w * l
        parts["ngf"] = float(l.detach())

    w = float(cfg.get("lor_loss_weight", 0.0))
    if lor is not None and w > 0:
        l = lor(ct_pred, gt_ct, mask, seg)
        total = total + w * l
        parts["lor"] = float(l.detach())

    # Deep supervision: the same L_mu on the coarse auxiliary heads, with the ground
    # truth, mask and segmentation downsampled to match.
    ds_aux = pred.get("ds_aux") if isinstance(pred, dict) else None
    ds_weights = cfg.get("ds_weights", [0.5, 0.25])
    if ds_aux and ds_weights:
        for aux, w_ds in zip(ds_aux, ds_weights):
            size = aux.shape[-3:]
            t = torch.nn.functional.interpolate(gt_ct, size=size, mode="trilinear",
                                                align_corners=False)
            m = torch.nn.functional.interpolate(mask.float(), size=size, mode="nearest") > 0.5
            s = (torch.nn.functional.interpolate(seg.float(), size=size, mode="nearest")
                 if seg is not None else None)
            l = base_loss(aux, t, m, s)
            total = total + float(w_ds) * l
            parts[f"ds{size[0]}"] = float(l.detach())

    return total, parts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="path to a YAML config in configs/")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(cfg["output_dir"])
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    input_keys = list(cfg["input_keys"])
    patch_size = tuple(cfg["patch_size"])
    needs_ngf = float(cfg.get("ngf_loss_weight", 0.0)) > 0

    print(f"Config:        {args.config}")
    print(f"Device:        {device}")
    print(f"Inputs:        {input_keys}  (in_channels={cfg['in_channels']})")
    print(f"Patch:         {list(patch_size)} x {cfg.get('train_num_samples', 2)} samples/step")
    print(f"Output dir:    {out_dir}")

    # --- data -------------------------------------------------------------------
    extra_labels = ["organ_seg"] + (["mri_in_reg", "mri_out_reg"] if needs_ngf else [])
    if needs_ngf:
        print("L_NGF is on -> loading the registered MRI as a training-only target "
              "(run scripts/register_mri.py first).")

    subjects = get_dataset(cfg["data_dir"])
    holdout = set(cfg.get("holdout_subjects", []) or [])
    if holdout:
        subjects = [s for s in subjects if s["subject_id"] not in holdout]
        print(f"Holding out {len(holdout)} subjects: {sorted(holdout)}")
    print(f"Training on {len(subjects)} subjects")

    transforms = get_train_transforms(
        patch_size=patch_size,
        num_samples=int(cfg.get("train_num_samples", 2)),
        input_keys=input_keys,
        extra_label_keys=extra_labels,
    )
    # MetaTensor tracking is pure overhead here and costs real time on 512^3 volumes.
    set_track_meta(False)
    dataset = CacheDataset(data=subjects, transform=transforms,
                           cache_rate=float(cfg.get("cache_rate", 0.0)),
                           num_workers=int(cfg.get("num_workers", 4)))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        persistent_workers=bool(cfg.get("persistent_workers", True)),
        # pin_memory crashes with CUDA workers on some driver/torch combinations;
        # it buys little here because the patches are large and few.
        pin_memory=bool(cfg.get("pin_memory", False)),
    )

    # --- model ------------------------------------------------------------------
    model = build_model(cfg).to(device)
    if cfg.get("init_checkpoint"):
        load_warm_start(model, cfg["init_checkpoint"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_params / 1e6:.2f}M")

    base_loss, lor, anper = build_losses(cfg, device)
    active = ["L_mu" if cfg.get("loss_type", "mu_l1") == "mu_l1" else "plain L1"]
    for key, name in (("grad_loss_weight", "L_grad"), ("anper_loss_weight", "L_AnPer"),
                      ("ngf_loss_weight", "L_NGF"), ("lor_loss_weight", "L_LOR")):
        if float(cfg.get(key, 0.0)) > 0:
            active.append(f"{name}({cfg[key]})")
    print(f"Loss terms:    {' + '.join(active)}")

    epochs = int(cfg["epochs"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_dtype = torch.float16 if cfg.get("amp_dtype", "float16") == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    snapshot_every = int(cfg.get("snapshot_every", 0))

    best_loss = float("inf")
    ck_dir = out_dir / "checkpoints"
    write_meta_sidecar(cfg, ck_dir / "last_model.pth")
    write_meta_sidecar(cfg, ck_dir / "best_model.pth")

    print("\nTraining...\n")
    for epoch in range(epochs):
        model.train()
        running, running_parts = 0.0, {}
        pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            x = batch["input"].to(device, non_blocking=True)
            mask = batch["prediction_mask"].to(device) > 0.5
            seg = batch.get("organ_seg")
            seg = seg.to(device) if torch.is_tensor(seg) else None

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                pred = model(x)
                loss, parts = compute_loss(pred, batch, cfg, base_loss, lor, anper,
                                           mask, seg, device)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.detach())
            for k, v in parts.items():
                running_parts[k] = running_parts.get(k, 0.0) + v
            pbar.set_postfix(loss=f"{float(loss.detach()):.4f}")

        scheduler.step()
        n = max(1, len(loader))
        avg = running / n
        detail = "  ".join(f"{k}={v / n:.4f}" for k, v in running_parts.items())
        print(f"Epoch {epoch}  train={avg:.4f}  {detail}", flush=True)

        torch.save(model.state_dict(), ck_dir / "last_model.pth")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), ck_dir / "best_model.pth")
        if snapshot_every and epoch % snapshot_every == 0:
            snap = ck_dir / f"epoch{epoch}.pth"
            torch.save(model.state_dict(), snap)
            write_meta_sidecar(cfg, snap)

        with open(out_dir / "train_log.csv", "a") as f:
            f.write(f"{epoch},{avg}\n")

    print(f"\nDone. Weights: {ck_dir / 'last_model.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
