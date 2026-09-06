"""L_AnPer — anatomical perception loss against a frozen TotalSegmentator network.

A pseudo-CT can have a low voxel-wise error and still be anatomically implausible in
ways the AC-PET metrics punish. This term passes the predicted CT and the ground-truth
CT through a frozen segmentation network and matches intermediate decoder features, in
the spirit of perceptual losses. Because the network was trained to recognise organs,
agreeing in its feature space means agreeing about anatomy rather than about
intensities.

In a matched pair differing by nothing else, adding this term improved organ bias by
5.8 % (3.34 -> 3.14) and the brain-outlier score by 27.4 % (0.0261 -> 0.0189) — the
largest effect on the AC-PET metrics of anything we tried.

Training only. The teacher is never part of the deployed model, so it costs nothing at
inference and does not enter the submission container.

REQUIRES TotalSegmentator's weights on disk; see the README, "Optional: L_AnPer".
Set `map_loss_weight: 0.0` in a config to train without it and skip the dependency.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnatomicalPerceptionLoss(nn.Module):
    """Feature matching against a frozen TotalSegmentator nnU-Net.

    The teacher is task 297 (`total`, 3 mm, PlainConvUNet) trained on 1559 subjects —
    a far stronger anatomical feature extractor than anything we could train on the 75
    subjects released here.

    Inputs are CT in [0, 1]. We invert to HU and apply nnU-Net's CT normalization
    (clip to the [0.5, 99.5] percentiles, then z-score) using the constants baked into
    the downloaded model's `dataset_fingerprint.json`, so the teacher sees what it was
    trained to see.

    SPACING. The teacher was trained at 3 mm and we feed ~1.5 mm patches. That is fine
    because the loss is RELATIVE: prediction and ground truth go through identical
    preprocessing and the identical network, so it measures whether they produce the
    same anatomical activations — not whether the segmentation is correct.

    Gradients flow through the prediction side only; the ground-truth features are
    detached.
    """

    def __init__(
        self,
        model_folder: str,
        layers=("decoder.stages.0", "decoder.stages.1", "decoder.stages.2"),
        device: str = "cuda",
        hu_lo: float = -1000.0,
        hu_span: float = 3000.0,
    ):
        super().__init__()
        model_folder = Path(model_folder)

        # nnU-Net insists these exist; only the results dir is actually read.
        os.environ.setdefault("nnUNet_raw", "/tmp/nnraw")
        os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnprep")
        os.environ["nnUNet_results"] = str(model_folder.parents[1])

        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor = nnUNetPredictor(device=torch.device(device), allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(
            str(model_folder), use_folds=(0,), checkpoint_name="checkpoint_final.pth"
        )
        net = predictor.network
        net.decoder.deep_supervision = False    # return one tensor, not the DS list
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        net.to(device)
        self.net = net

        fp = json.loads((model_folder / "dataset_fingerprint.json").read_text())
        fp = fp["foreground_intensity_properties_per_channel"]["0"]
        self.mean = float(fp["mean"])
        self.std = float(fp["std"])
        self.clip_lo = float(fp["percentile_00_5"])
        self.clip_hi = float(fp["percentile_99_5"])
        self.hu_lo = float(hu_lo)
        self.hu_span = float(hu_span)

        self.layers = tuple(layers)
        self._feats: dict = {}
        mods = dict(net.named_modules())
        missing = [n for n in self.layers if n not in mods]
        if missing:
            raise KeyError(f"AnPer: layers not found in the teacher: {missing}")
        for name in self.layers:
            mods[name].register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(_module, _inp, out):
            self._feats[name] = out
        return hook

    def _prep(self, x01):
        """CT in [0,1] -> HU -> nnU-Net CT normalization."""
        hu = x01 * self.hu_span + self.hu_lo
        hu = torch.clamp(hu, self.clip_lo, self.clip_hi)
        return (hu - self.mean) / self.std

    def _run(self, x):
        self._feats = {}
        self.net(self._prep(x))
        return {k: self._feats[k] for k in self.layers}

    @staticmethod
    def _channel_norm(f, eps=1e-10):
        """Unit-normalize each spatial location across channels."""
        return f / (f.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)

    def forward(self, pred, gt):
        fp = self._run(pred)
        with torch.no_grad():
            fg = self._run(gt)
        loss = pred.new_zeros(())
        for k in self.layers:
            loss = loss + F.l1_loss(self._channel_norm(fp[k]),
                                    self._channel_norm(fg[k]).detach())
        return loss / len(self.layers)
