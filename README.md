# Region-Weighted Losses and Model Fusion for Cross-Modal PET Attenuation Correction

Team **ntkhoa** — joint 1st place, [BIC-MAC Challenge](https://bic-mac-challenge.github.io/) (MICCAI 2026).

This repository contains everything needed to reproduce Table 1 of our
[methodology paper](paper/Region_Weighted_Losses_and_Two_Model_Fusion_forCross_Modal_PET_Attenuation_Correction.pdf):
the **MuNet** architecture, the four loss functions we designed, and the two-model
fusion used for the final submission.

The task is to synthesize a pseudo-CT in Hounsfield Units from non-attenuation-corrected
PET, DIXON MRI and a topogram. The organizers then run a fixed STIR reconstruction on
that pseudo-CT, and submissions are scored on **both** CT fidelity and downstream PET
fidelity — which is the tension every design choice here addresses.

**Our result on the 20 unseen test subjects, against the organizers' baseline:**

| Metric (lower is better) | Baseline | **Ours** | |
|---|---:|---:|:--:|
| CT μ-map MAE | 0.00664 | **0.00578** | **−13 %** |
| Brain outlier score | 0.1194 | **0.0201** | **5.9×** |
| Organ bias (MARE %) | 3.93 | **1.85** | **2.1×** |
| Whole-body SUV MAE | 0.0467 | **0.0275** | **1.7×** |

---

## The idea in one paragraph

Two properties of the *fixed* reconstruction pipeline decide which errors are expensive,
and neither is something a network can learn around:

1. The HU→μ conversion is **bilinear with a knee at 47 HU**, and the segment below the
   knee is **1.88× steeper**. One HU of error therefore costs almost twice as much μ in
   soft tissue as in bone, so an L1 loss in HU misprices its own errors relative to the
   metric. We compute the loss in **Carney μ space** instead.
2. The pipeline **blurs the μ-map at 4 mm FWHM before forward projection**. Detail finer
   than that is invisible to the three PET metrics and visible only to the CT metric.

So the loss mattered more than the architecture. The largest single improvement in the
whole project (Table 1 row 1 → row 2) is a **loss change alone**: same backbone, same
input, CT down 12 % and every PET metric better by a factor of 1.4 to 3.7.

---

## Table 1 — what each config reproduces

Validation set, 4 subjects, all metrics lower-is-better. Each row adds one change to
the row above it within its block.

| # | Model | Change | Config | CT μ-MAE | SUV MAE | Organ | Brain |
|---|---|---|---|---:|---:|---:|---:|
| 1 | Baseline | plain L1, NAC-PET only | *(organizers')* | 0.006610 | 0.0586 | 4.56 | 0.0541 |
| 2 | MuNet₂ | `L_mu` + `L_grad` + `L_AnPer` | [`row2_munet2`](configs/row2_munet2.yaml) | 0.005803 | 0.0377 | 2.74 | 0.0148 |
| 3 | | + `L_NGF` | [`row3_munet2_ngf`](configs/row3_munet2_ngf.yaml) | 0.005972 | 0.0352 | 2.58 | 0.0158 |
| 4 | | + `L_LOR` | [`row4_munet2_ngf_lor`](configs/row4_munet2_ngf_lor.yaml) | 0.005958 | 0.0345 | 2.45 | 0.0168 |
| 5 | MuNet₄ | `L_mu` + `L_grad` + `L_AnPer` | [`row5_munet4`](configs/row5_munet4.yaml) | **0.005679** | 0.0383 | 2.80 | 0.0168 |
| 6 | | + `L_LOR` | [`row6_munet4_lor`](configs/row6_munet4_lor.yaml) | 0.005703 | 0.0339 | 2.38 | **0.0078** |
| 7 | | + 100 ep at ⅕ the learning rate | [`row7_munet4_lor_refine`](configs/row7_munet4_lor_refine.yaml) | 0.005710 | — | — | — |
| 8 | Fusion | 0.3 × row 4 + 0.7 × row 6 | *(inference only)* | **0.005627** | 0.0331 | 2.27 | 0.0091 |
| 9 | | **0.3 × row 4 + 0.7 × row 7 — submitted** | *(inference only)* | 0.005633 | **0.0326** | **2.23** | 0.0087 |

Rows 8 and 9 need no training: they are convex blends of two finished models, produced
at inference time by `src/predict.py`.

Row 7's three PET metrics are blank because the challenge's validation phase closed
before that model's own reconstruction could be uploaded. Its effect on all four metrics
is visible in row 9.

---

## Install

```bash
git clone https://github.com/Khoa-NT/BIC-MAC_Challenge_ntkhoa.git
cd BIC-MAC_Challenge_ntkhoa

python -m venv .venv && source .venv/bin/activate
# Install the torch build matching your driver FIRST, from pytorch.org, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Tested with Python 3.12, torch 2.6.0+cu124, MONAI 1.5.2.

**Two optional dependencies**, each needed only for specific rows — see
[Optional dependencies](#optional-dependencies) below.

### Data

Download the [`DEPICT-RH/BIC-MAC`](https://huggingface.co/datasets/DEPICT-RH/BIC-MAC)
dataset. Every config expects `data_dir` to point at `train/`:

```
<data_dir>/sub-XXX/features/    nacpet, topogram, mri_combined_{in,out}_phase, metadata.json
<data_dir>/sub-XXX/ct-label/    ct, body_seg, organ_seg, prediction_mask
```

All feature volumes are pre-resampled onto the CT grid (512 × 512 × 531 at
1.52 × 1.52 × 2.00 mm), so nothing here resamples anything.

### Checkpoints

The six trained models are **~93 MB each, 558 MB total**, so they are not committed to
git. Download the archive and unpack it into `checkpoints/`:

> **📦 [Download the checkpoints (Google Drive)](https://drive.google.com/file/d/1l6XONUm0Cq3OJ9SxqgZzv75fWQNpvAEu/view?usp=sharing)**

```
checkpoints/
├── row2_munet2.pth                 + .meta.json
├── row3_munet2_ngf.pth             + .meta.json
├── row4_munet2_ngf_lor.pth         + .meta.json   ← fusion member, weight 0.3
├── row5_munet4.pth                 + .meta.json
├── row6_munet4_lor.pth             + .meta.json
└── row7_munet4_lor_refine.pth      + .meta.json   ← fusion member, weight 0.7
```

Each `.meta.json` records the architecture, so a checkpoint is self-describing and
`predict.py` never has to be told which config produced it. The `.meta.json` files
**are** committed, so you can inspect the configurations without downloading the weights.

---

## Inference

Reproduce the submitted model (Table 1 row 9):

```bash
python src/predict.py \
    --features_dir  /path/to/BIC-MAC/val/sub-004/features \
    --output_ct     out/sub-004/ct.nii.gz \
    --checkpoints   checkpoints/row4_munet2_ngf_lor.pth \
                    checkpoints/row7_munet4_lor_refine.pth \
    --weights       0.3 0.7
```

Any single row works too — just pass one checkpoint and no weights:

```bash
python src/predict.py \
    --features_dir /path/to/BIC-MAC/val/sub-004/features \
    --output_ct    out/sub-004/ct.nii.gz \
    --checkpoints  checkpoints/row6_munet4_lor.pth
```

About 60 s per model per subject on an RTX 3090 at the default `--overlap 0.5`. The
output copies the affine and header of `features/nacpet.nii.gz`, which is what the
evaluation code and the reconstruction pipeline both require.

**Why fusion happens here rather than on the PET images.** The challenge container emits
a single pseudo-CT and the organizers run the reconstruction themselves, so any
combination of models must collapse into one volume *before* it is scored. The blend is
a fixed convex combination in HU, with the weights fixed in advance. We tried learning
them; an 8k-parameter per-voxel gate appeared to gain 4.5 %, but that was leakage —
both members had trained on all 75 subjects, so in-sample one member looks far better
than it is out-of-sample.

---

## Training

One config per table row:

```bash
python src/train.py --config configs/row2_munet2.yaml
```

Rows 4, 6 and 7 are **later stages of a multi-stage recipe** and warm-start from the row
above, so train the parent first (or download its checkpoint and point
`init_checkpoint` at it). The lineages are:

```
row2                                             300 ep,  lr 3e-4
row3  →  row4                                   1500 ep + 200 ep
row5  →  row6  →  row7                           500 ep + 200 ep + 100 ep
```

All configs are heavily commented — what each knob does, why it has that value, and
what we measured when we changed it.

**Hardware.** Every model was trained on a single 48 GB GPU (RTX A6000). The 208³ configs
peak around 40 GB and the 448×448×48 slab configs about the same. On a 24 GB card, drop
`train_num_samples` to 1 rather than shrinking `patch_size` — see the warning below.

**Time.** Roughly 1 to 13 days per row (each config states its own estimate). Row 3's
1500 epochs is the long pole.

### Run the baseline — with our losses, or any subset of them

Two extra configs let you separate what the **loss** bought from what the
**architecture** bought. Table 1 cannot do that on its own, because its row 1 → row 2
step changes the loss, the architecture *and* the inputs at once.

| Config | Architecture | Loss |
|---|---|---|
| [`baseline_unet`](configs/baseline_unet.yaml) | organizers' plain U-Net | plain L1 *(their recipe)* |
| [`baseline_unet_our_losses`](configs/baseline_unet_our_losses.yaml) | organizers' plain U-Net | **ours** |
| [`row2_munet2`](configs/row2_munet2.yaml) | MuNet | **ours** |

```bash
python src/train.py --config configs/baseline_unet_our_losses.yaml
```

The baseline architecture is not a separate model class — it is MuNet with the three
additions switched off, which makes `src/munet.py` structurally identical to the
organizers' `UNet3D`. So you can toggle it on any config:

```yaml
use_attention_skips: false     # 22.9M params with all three off,
encoder_attention: false       # 24.3M with them on — the additions are
deep_supervision: false        # 1.4M parameters, 5.8% of the model
```

**Every loss term is opt-in through its weight.** Set one to `0.0` and it is skipped
entirely — no dependency, no cost:

| Key | Term | Needs |
|---|---|---|
| `loss_type: mu_l1` or `l1` | `L_mu` (region-weighted, μ space) or plain L1 | — |
| `grad_loss_weight` | `L_grad` | — |
| `anper_loss_weight` | `L_AnPer` | `nnunetv2` + TotalSegmentator weights |
| `ngf_loss_weight` | `L_NGF` | `scripts/register_mri.py` output |
| `lor_loss_weight` | `L_LOR` | a 448 × 448 × 48 slab patch |

⚠ **We never published a number for `baseline_unet_our_losses`** — it is not in the
paper. Treat whatever you get as your own measurement, and give it the same epoch budget
as whatever you compare it against, or you will be measuring training length rather than
architecture.

### Three things that will bite you

**The slab patch is load-bearing for `L_LOR`.** A line integral is meaningless unless the
line crosses the whole body. A 208³ patch spans 316 mm against a 400–500 mm body, so the
integrals are truncated and the loss becomes a poor proxy. That is why rows 4, 6 and 7
use a 448 × 448 × 48 slab — it spans 681 mm in x/y at only 1.07× the voxel count of the
cube. `train.py` warns if you make the patch too narrow.

**Do not shrink `patch_size` to fit a smaller GPU.** Patches are sampled uniformly at
random from a 512 × 512 × 531 volume, and the body fills only part of it. At 208³ or the
slab, a patch essentially always intersects the body; at 64³ many land entirely in air,
where the mask is empty and the loss is exactly zero, so those samples contribute no
gradient at all. It fails silently — the loss just looks small. Reduce
`train_num_samples` instead.

**"Best" checkpoints are selected on the *training* loss.** With `holdout_subjects: []`
(all 75 subjects used for gradient updates, as in the paper) there is no validation
signal, so `best_model.pth` is simply the lowest training loss — which in our experiments
was a poor predictor of validation CT. **The released checkpoints are last-epoch
weights.** Prefer `last_model.pth` unless you set up a real held-out split.

### Optional dependencies

**`L_AnPer`** (rows 2–7, `anper_loss_weight: 0.1`) needs `nnunetv2` and the
TotalSegmentator task-297 weights:

```bash
pip install nnunetv2 TotalSegmentator
totalsegmentator --help          # triggers the weight download on first run
```

Then point `anper_model_folder:` in each config at
`…/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres`.

To train without it, set `anper_loss_weight: 0.0`. You will not reproduce the table
exactly — it was worth −5.8 % organ bias and −27.4 % brain outlier in a matched pair,
the largest effect on the PET metrics of anything we tried.

**`L_NGF`** (rows 3 and 4 only) needs a deformably registered DIXON MRI, which is not
part of the released dataset. Produce it once with ANTs:

```bash
pip install antspyx
python scripts/register_mri.py --data_dir /path/to/BIC-MAC/train --out_root mri_reg_ct
```

About 20 minutes per subject on 8 threads, so roughly a day for all 75. Run it once and
keep the output. It is training-only by construction (the fixed image is the
ground-truth CT), which is fine because the MRI enters `L_NGF` as a *target* and never as
a model input — a model trained with it still reads only NAC-PET and the topogram at
inference.

---

## What is in here

```
src/
├── munet.py        MuNet: the organizers' residual 3D U-Net + our three additions
├── losses.py       L_mu, L_grad, L_NGF, L_LOR, and the Carney HU→μ map
├── anper.py        L_AnPer — perceptual loss vs a frozen TotalSegmentator
├── data.py         subject listing + MONAI transforms for training and inference
├── train.py        one model, one config
└── predict.py      single-model and fused inference
configs/            one YAML per table row, plus two baseline configs
scripts/
└── register_mri.py ANTs SyNRA MRI→CT registration (only needed for L_NGF)
checkpoints/        the six trained models (download separately) + meta sidecars
paper/              the methodology paper
```

### The method, briefly

**MuNet** is the organizers' baseline U-Net plus three additions — attention gates on the
skips, self-attention at the two deepest stages, and two deep-supervision heads. Together
they are **1.4 M parameters, 5.8 % of the model** (24.3 M vs 22.9 M). All three are
*identity at initialization*, so a model carrying them reproduces its plain-U-Net parent
exactly at epoch 0. Only two variants appear in the paper, differing solely in input
width: **MuNet₂** reads NAC-PET + topogram, **MuNet₄** adds the two raw DIXON phases.

**The losses:**

| | What it does |
|---|---|
| `L_mu` | Region-weighted L1 in Carney μ space. Weights ×3 bone, ×3 brain, ×2 the four organs the organ-bias metric is computed over. These weights are also what makes from-scratch training work: under a plain L1 our from-scratch runs produced **zero** voxels above 300 HU — the network never learned bone. |
| `L_grad` | Matches gradient magnitudes. `L_mu` is per-voxel and cannot tell a sharp edge from a smeared one. |
| `L_AnPer` | Matches decoder features of a frozen TotalSegmentator. Agreeing in the feature space of a network trained to recognise organs means agreeing about *anatomy*, not intensities. |
| `L_NGF` | Cross-modal boundary agreement with the DIXON MRI, comparing gradient *directions* and ignoring magnitudes. DIXON has soft-tissue contrast and almost no bone signal, making it a good boundary prior and a poor intensity target. |
| `L_LOR` | Error of line integrals along lines of response. **The only term that ever improved organ bias.** Attenuation acts as exp(−∫μ ds) along lines crossing the whole body, so an organ's reconstructed activity does not depend on μ inside that organ — which is exactly why raising the organ weight in `L_mu` did nothing, and why this term exists. |

### The DIXON MRI is fed raw and unregistered — on purpose

Registering it and feeding it in was worth −7.8 % CT MAE **with the ground-truth CT as
the fixed image**, but that operator cannot exist at test time. Substituting a pseudo-CT
made the train and test operators differ, and the model got *worse the longer it
trained*: −1.3 % at 15 epochs, **+17 % at 40**. A learned registration network with a
consistent operator closed only 13.6 % of the gap. Feeding the MRI raw sidesteps the
problem instead of solving it: the residual misalignment is small relative to the
bottleneck's receptive field, so the network absorbs it internally rather than depending
on an operator we cannot keep consistent.

---

## Citation

```bibtex
@misc{nguyen2026regionweightedlossesmodelfusion,
      title={Region-Weighted Losses and Model Fusion for Cross-Modal PET Attenuation Correction}, 
      author={Khoa Tuan Nguyen and Joris Vankerschaver and Wesley De Neve},
      year={2026},
      eprint={2608.21881},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.21881}, 
}
```

## Acknowledgements

Thanks to the BIC-MAC organizers for the dataset, the reconstruction pipeline and the
challenge. The baseline U-Net and the STIR reconstruction code are theirs; see
[bic-mac-challenge/challenge-codebase](https://github.com/bic-mac-challenge/challenge-codebase).
