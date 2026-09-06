"""The four loss terms defined in the paper, plus the Carney HU-to-mu map they share.

    L_mu     region-weighted L1 in Carney attenuation space          make_mu_l1()
    L_grad   gradient-magnitude matching (edge sharpness)            gradient_l1()
    L_NGF    cross-modal boundary agreement with the DIXON MRI       ngf_loss()
    L_LOR    error of line integrals along lines of response         make_lor_loss()

The fifth term, L_AnPer (anatomical perception against a frozen TotalSegmentator),
lives in `anper.py` because it needs nnU-Net.

Everything here operates on CT in [0, 1], the normalization used in training, where
    HU = x * 3000 - 1000.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# The metric's own space
# ---------------------------------------------------------------------------

def carney_mu(x: torch.Tensor) -> torch.Tensor:
    """Carney (2006) bilinear HU -> linear attenuation coefficient, in cm^-1.

    This is the map the CT metric is computed in and the map the reconstruction
    pipeline applies before forward projection, so it is the space our loss should
    measure error in. It is piecewise linear with a knee at 47 HU:

        mu = 9.6e-5 * (HU + 1000)                     for HU < 47
        mu = 5.1e-5 * (HU + 1000) + 4.71e-2           for HU >= 47

    The lower segment is 1.88x steeper, so one HU of error below the knee costs
    almost twice as much mu as one HU above it. An L1 loss in HU therefore misprices
    its own errors relative to the metric; an L1 in mu prices them correctly.

    `x` is CT in [0, 1], hence `x * 3000` is (HU + 1000) and the knee sits at 1047.
    The clamp keeps mu non-negative (HU < -1000 is unphysical); it is inactive
    inside the body mask in practice.
    """
    hu1000 = x * 3000.0
    mu = torch.where(hu1000 < 1047.0, 9.6e-5 * hu1000, 5.10e-5 * hu1000 + 4.71e-2)
    return mu.clamp(min=0.0)


# ---------------------------------------------------------------------------
# Plain L1 — the organizers' baseline objective, for reference and ablation
# ---------------------------------------------------------------------------

def plain_l1(pred, target, mask, seg=None):
    """Masked L1 in normalized [0, 1] CT space. The objective our work replaces.

    Provided so `configs/baseline_unet.yaml` can reproduce the organizers' recipe and
    so any config can switch back with `loss_type: l1`. Two differences from `L_mu`
    matter, and both are the point of the paper:

      * it measures error in HU rather than in mu, so it misprices its own errors
        relative to the CT metric by up to the 1.88x knee ratio;
      * it has no region weights, and without them a from-scratch model never learns
        bone at all. Our from-scratch runs under this loss produced ZERO voxels above
        300 HU and a brain-outlier score of 0.409, roughly six times worse.

    `seg` is accepted and ignored, so the two losses are interchangeable in train.py.
    """
    m = mask.float()
    return ((pred - target).abs() * m).sum() / m.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# L_mu — region-weighted L1 in mu space
# ---------------------------------------------------------------------------

def make_mu_l1(
    bone_weight: float = 3.0,
    bone_hu: float = 300.0,
    brain_weight: float = 3.0,
    organ_weight: float = 2.0,
    brain_label: int = 90,
    organ_labels=(1, 5, 7, 51),
):
    """Region-weighted L1 in Carney-mu space over the body mask. Equation 1 of the paper.

        L_mu = sum_v w_v m_v |mu(x_v) - mu(y_v)|  /  sum_v w_v m_v
        w_v  = max(1, bone, brain, organ)

    The three region factors up-weight what the four scored metrics are sensitive to:

        bone    ground-truth HU > `bone_hu`      x bone_weight
        brain   organ_seg == `brain_label`       x brain_weight
        organ   organ_seg in `organ_labels`      x organ_weight
                (spleen 1, liver 5, pancreas 7, heart 51 — the four organs the
                 organ-bias metric is computed over; we did not pick them, the
                 metric did)

    Taking the MAXIMUM rather than the product bounds the combined weight at
    `max(bone_weight, brain_weight, organ_weight)`, so no voxel can dominate a batch.

    These weights are also what makes training from random initialization work at
    all: under a plain L1 our from-scratch runs produced pseudo-CTs with essentially
    no voxels above 300 HU — the network never learned bone, which wrecks the
    brain-outlier score because that metric is driven by the skull.

    `seg` is the ground-truth organ segmentation, used only to build the weights
    during training. Inference never needs it, so the deployed model still reads
    nothing but the released features. Passing `seg=None` (or leaving both the brain
    and organ weights at 1.0) reduces this to the bone-only variant.
    """
    bone_norm = (bone_hu + 1000.0) / 3000.0            # bone_hu in [0,1] space
    bw, brw, ow = float(bone_weight), float(brain_weight), float(organ_weight)
    brain_label = int(brain_label)
    organ_labels = tuple(int(v) for v in organ_labels)
    use_seg = brw > 1.0 or ow > 1.0

    def _mu_l1(pred, target, mask, seg=None):
        pmu, tmu = carney_mu(pred), carney_mu(target)
        w = torch.where(target > bone_norm,
                        torch.full_like(target, bw), torch.ones_like(target))
        if use_seg and seg is not None:
            if brw > 1.0:
                w = torch.maximum(w, torch.where(
                    seg == brain_label,
                    torch.full_like(target, brw), torch.zeros_like(target)))
            if ow > 1.0:
                in_organ = torch.zeros_like(target, dtype=torch.bool)
                for lab in organ_labels:
                    in_organ = in_organ | (seg == lab)
                w = torch.maximum(w, torch.where(
                    in_organ, torch.full_like(target, ow), torch.zeros_like(target)))
        w = mask.float() * w
        return ((pmu - tmu).abs() * w).sum() / w.sum().clamp(min=1.0)

    return _mu_l1


# ---------------------------------------------------------------------------
# L_grad — edge strength
# ---------------------------------------------------------------------------

def gradient_l1(pred, target, mask):
    """L1 between the first-difference gradients of prediction and target.

    L_mu is a per-voxel loss and is therefore indifferent to whether an edge is sharp
    or smeared, as long as the average intensity is right. This term asks for the same
    edge strength in the same places, which mainly affects the soft-tissue boundaries
    where our error concentrates.

    Masking is done on voxel PAIRS: a difference is scored only where both of its
    endpoints are inside the body mask, so the body/air boundary does not generate a
    spurious gradient.
    """
    def _g(v):
        return (v[:, :, 1:] - v[:, :, :-1],
                v[:, :, :, 1:] - v[:, :, :, :-1],
                v[:, :, :, :, 1:] - v[:, :, :, :, :-1])

    m = mask.float()
    mx = m[:, :, 1:] * m[:, :, :-1]
    my = m[:, :, :, 1:] * m[:, :, :, :-1]
    mz = m[:, :, :, :, 1:] * m[:, :, :, :, :-1]
    pg, tg = _g(pred), _g(target)
    num = (((pg[0] - tg[0]).abs() * mx).sum()
           + ((pg[1] - tg[1]).abs() * my).sum()
           + ((pg[2] - tg[2]).abs() * mz).sum())
    den = (mx.sum() + my.sum() + mz.sum()).clamp(min=1.0)
    return num / den


# ---------------------------------------------------------------------------
# L_NGF — cross-modal boundary agreement
# ---------------------------------------------------------------------------

def _gaussian_blur3d(v, sigma):
    """Separable 3D Gaussian blur, `sigma` in voxels. No-op if sigma <= 0."""
    if not sigma or sigma <= 0:
        return v
    r = max(1, int(round(3.0 * float(sigma))))
    x = torch.arange(-r, r + 1, device=v.device, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2.0 * float(sigma) ** 2))
    k = (k / k.sum()).to(v.dtype)
    c = v.shape[1]
    for axis, pad in ((2, (r, 0, 0)), (3, (0, r, 0)), (4, (0, 0, r))):
        shape = [1, 1, 1, 1, 1]
        shape[axis] = k.numel()
        w = k.view(shape).expand(c, 1, *shape[2:]).contiguous()
        v = F.conv3d(v, w, padding=pad, groups=c)
    return v


def _grad3(v):
    """Per-axis central differences, same shape, ends zeroed."""
    gx = torch.zeros_like(v); gx[:, :, 1:-1] = 0.5 * (v[:, :, 2:] - v[:, :, :-2])
    gy = torch.zeros_like(v); gy[:, :, :, 1:-1] = 0.5 * (v[:, :, :, 2:] - v[:, :, :, :-2])
    gz = torch.zeros_like(v); gz[:, :, :, :, 1:-1] = 0.5 * (v[:, :, :, :, 2:] - v[:, :, :, :, :-2])
    return gx, gy, gz


def ngf_loss(ct_pred, mri, gate, eps_rel: float = 0.1, smooth_sigma: float = 1.0):
    """Normalized Gradient Fields agreement between the predicted CT and the DIXON MRI.

    DIXON MRI has strong soft-tissue contrast and almost no bone signal, which makes it
    a good boundary prior and a poor intensity target. So we compare gradient
    DIRECTIONS and ignore magnitudes, using the cross-product form

        L_vox = |n_ct x n_mri|^2 = |n_ct|^2 |n_mri|^2 - <n_ct, n_mri>^2
        n(I)  = grad I / sqrt(|grad I|^2 + eta^2)

    which is zero when the two images have edges in the same places and orientations,
    regardless of how differently they are scaled. Concretely it is:

      * intensity-invariant  — only direction matters, so CT-HU and DIXON need no
                               intensity correspondence;
      * polarity-invariant   — a bright-to-dark CT edge matches a dark-to-bright MRI edge;
      * zero where an edge exists in only ONE modality, so bone edges the MRI cannot
        see and MRI texture the CT should not have are not penalized;
      * maximal only for co-located but MIS-oriented edges.

    eta is set per sample from the mean gradient energy inside the gate
    (eta^2 = eps_rel^2 * mean|grad I|^2), which makes the term invariant to each
    image's absolute intensity scale — so the raw, un-normalized MRI is a fine input.

    `gate` restricts the term to soft tissue, where the MRI is informative. The MRI
    enters the model through this loss ALONE, so a model trained with it still reads
    only NAC-PET and the topogram at inference.

    Computed in float32 with autocast disabled: the sqrt and the division are not
    numerically safe in fp16.
    """
    with torch.autocast("cuda", enabled=False):
        m = gate.float()
        per = m.sum(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1.0)
        a = _gaussian_blur3d(ct_pred.float(), smooth_sigma)
        b = _gaussian_blur3d(mri.float(), smooth_sigma)
        ax, ay, az = _grad3(a)
        bx, by, bz = _grad3(b)
        a2 = ax * ax + ay * ay + az * az
        b2 = bx * bx + by * by + bz * bz
        e2a = (eps_rel ** 2) * (a2 * m).sum(dim=(1, 2, 3, 4), keepdim=True) / per + 1e-12
        e2b = (eps_rel ** 2) * (b2 * m).sum(dim=(1, 2, 3, 4), keepdim=True) / per + 1e-12
        dot = ax * bx + ay * by + az * bz
        lvox = ((a2 * b2 - dot * dot) / ((a2 + e2a) * (b2 + e2b))).clamp(min=0.0)
        return (lvox * m).sum() / m.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# L_LOR — line integrals along lines of response
# ---------------------------------------------------------------------------

def make_lor_loss(
    n_angles: int = 4,
    fwhm_mm: float = 4.0,
    spacing=(1.52, 1.52, 2.0),
    organ_labels=(1, 5, 7, 51),
    min_lines: int = 32,
    include_body: bool = True,
):
    """Error of transaxial line integrals of the attenuation map.

        L_LOR = mean_theta mean_lines | integral_l ( smooth(mu_pred) - smooth(mu_gt) ) ds |

    WHY A NON-LOCAL TERM. Every other term here is local, and three of the four scored
    metrics are not. The attenuation along a line of response is exp(-integral mu ds),
    so the reconstructed activity in an organ depends on mu integrated along lines that
    cross the WHOLE BODY, not on mu inside that organ. Making an organ's own mu accurate
    therefore does not make its reconstructed activity accurate — which is exactly why
    simply raising `organ_weight` in L_mu failed to move organ bias, and why this term
    was written. It is the only term we found that improved organ bias at all.

    PET lines of response are predominantly transaxial, so we integrate within axial
    slices: rotate each slice by theta and sum along one axis — a Radon transform — at
    `n_angles` angles over [0, pi).

    Two details matter.

    1. The mu difference is smoothed at `fwhm_mm` BEFORE integrating, matching the blur
       the reconstruction pipeline applies before its own forward projection. The term
       then only ever sees the band of spatial frequencies the PET metrics can see.

    2. The difference is masked to the body FIRST. The network is unconstrained outside
       the prediction mask, so unmasked values there would contaminate every line
       integral that passes through.

    WARNING — THE PATCH MUST SPAN THE BODY. A line integral is meaningless unless the
    line crosses the whole body. At 208^3 a patch spans 208 * 1.52 = 316 mm against a
    400-500 mm body, so the integrals are truncated and this loss becomes a poor proxy.
    Use a full-width slab: 448 x 448 x 48 spans 681 mm in x and y at 9.63 M voxels,
    only 1.07x the 9.00 M of a 208^3 cube, so the number of patches per batch need not
    drop. The training script warns if the patch is too narrow.

    Lines are averaged only over those that actually intersect the region, and a region
    is scored only if at least `min_lines` of them do — separately for the whole body
    (which targets whole-body SUV MAE) and for each organ (which targets organ bias).
    """
    sigma_vox = [(fwhm_mm / 2.35482) / s for s in spacing] if fwhm_mm and fwhm_mm > 0 else None
    angles = [math.pi * i / n_angles for i in range(n_angles)]
    organ_labels = tuple(int(v) for v in organ_labels)
    dy_mm = float(spacing[1])
    state = {"smoother": None, "grids": None, "shape": None}

    def _radon(img5):
        """(B,1,X,Y,Z) -> list over angles of (B*Z,1,X), line integrals in mm."""
        b, c, x, y, z = img5.shape
        img = img5.permute(0, 4, 1, 2, 3).reshape(b * z, c, x, y)
        if state["grids"] is None or state["shape"] != (b * z, c, x, y):
            state["grids"] = []
            for th in angles:
                cs, sn = math.cos(th), math.sin(th)
                mat = torch.tensor([[cs, -sn, 0.0], [sn, cs, 0.0]],
                                   device=img5.device, dtype=torch.float32)
                mat = mat.unsqueeze(0).expand(b * z, 2, 3)
                state["grids"].append(
                    F.affine_grid(mat, (b * z, c, x, y), align_corners=False))
            state["shape"] = (b * z, c, x, y)
        out = []
        for g in state["grids"]:
            rot = F.grid_sample(img, g.to(img.dtype), align_corners=False,
                                padding_mode="zeros")
            out.append(rot.sum(dim=3) * dy_mm)              # integrate along Y
        return out

    def _lor(pred, target, mask, seg=None):
        mu_p, mu_t = carney_mu(pred), carney_mu(target)
        if sigma_vox is not None:
            if state["smoother"] is None:
                from monai.networks.layers import GaussianFilter
                state["smoother"] = GaussianFilter(spatial_dims=3, sigma=sigma_vox).to(pred.device)
            mu_p, mu_t = state["smoother"](mu_p), state["smoother"](mu_t)
        m = mask.float()
        d = (mu_p - mu_t) * m                               # mask BEFORE projecting

        regions = [("body", m)] if include_body else []
        if seg is not None:
            regions += [(f"org{lab}", m * (seg == lab).float()) for lab in organ_labels]

        p_d = _radon(d)
        terms, denom = [], 0.0
        for _, r in regions:
            p_r = _radon(r)                                 # path length through the region
            for pd, pr in zip(p_d, p_r):
                hit = (pr > 1e-3).float()                   # lines that cross the region
                n = hit.sum()
                present = (n >= float(min_lines)).float()
                terms.append(present * ((pd.abs() * hit).sum() / n.clamp(min=1.0)))
                denom = denom + present
        if not terms:
            return pred.new_zeros(())
        return torch.stack(terms).sum() / torch.as_tensor(denom).clamp(min=1.0)

    return _lor
