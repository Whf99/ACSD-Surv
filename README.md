# ACSD-Surv

**Anatomically conditioned surface diffusion for survival prediction.**

ACSD-Surv combines tumor-surface geometry, volumetric images, and clinical covariates in a clinical-residual proportional-hazards model. It learns directional diffusion on triangular tumor surfaces and controls prediction sensitivity to contour perturbations through a fixed Sobolev metric.

## Overview

The model comprises four components:

1. **Surface diffusion:** a bounded, trace-normalized anisotropic operator conditioned on patient anatomical axes.
2. **Multiscale representation:** curvature heat responses, heat traces, directional extents, and spatial descriptors. Three diffusion scales produce 43 descriptors, followed by a smooth neural readout.
3. **Clinical-residual prediction:** a training-fold clinical Cox score anchors a zero-initialized multimodal residual; a frozen MedicalNet encoder supplies image features.
4. **Contour sensitivity regularization:** probability-level or log-risk Sobolev penalties evaluated through an exact rank-one factorization.

The survival prediction is

$$S_i(t)=\exp\left[-\widehat\Lambda_0(t)\exp\left(\eta_{\mathrm{clin}}(q_i)+f_\omega([q_i,z_i,h_i])\right)\right].$$

Training uses Cox partial likelihood, IPCW Brier loss, and optional sensitivity regularization. The baseline hazard and censoring distribution are estimated from training patients. Validation Brier loss selects the checkpoint; the test fold is evaluated afterward.

## Installation

The reference implementation uses Python 3.9 and PyTorch 2.5.1. The dense surface calculations use double precision; the training runner currently supports CPU execution.

```bash
git clone https://github.com/Whf99/ACSD-Surv.git
cd ACSD-Surv
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -r requirements.txt
```

For image-based variants, obtain the MedicalNet ResNet-18 source and 23-dataset pretrained weights from [MedicalNet](https://github.com/Tencent/MedicalNet). Place the files as follows:

```text
assets/medicalnet/
├── resnet.py
└── resnet_18_23dataset.pth
```

Set `MEDICALNET_ASSETS` if they are stored elsewhere. `ACSD_FEATURE_CACHE` optionally specifies a writable image-feature cache directory. Feature caches distinguish encoder source and weight versions. The clinical-plus-geometry variant does not require MedicalNet assets.

## Quick start

Run a small clinical-plus-geometry experiment with generated images, surfaces, and survival labels:

```bash
python examples/synthetic_run.py --output outputs/synthetic
```

The example runs the same preprocessing, Cox fitting, training, and evaluation code used for cohort experiments. It demonstrates execution only; synthetic results are not clinical performance estimates. Each run requires a new output directory.

## Data preparation

Patient data, split assignments, reference coordinates, and trained checkpoints are not included. Obtain the appropriate dataset permissions before preparing inputs.

```text
data/NPC/
├── images/
├── masks/
├── references.csv
└── splits/
    ├── fold_1/{train,val,test}.csv
    ├── fold_2/{train,val,test}.csv
    └── ...
```

Each split CSV requires `patient_id`, `image_path`, `label_path`, `stage`, `time`, and `recur_label`. Image and mask paths are relative to `data_root`. HECKTOR additionally uses `T-stage` and `N-stage` and requires primary-tumor masks. NIfTI images and masks must have matching dimensions and physical affines.

| Field | Convention |
|---|---|
| `time` | Positive follow-up duration in months, measured from treatment initiation |
| `recur_label` | 1 for the cohort endpoint event; 0 for right censoring |
| `stage` | 1–4 or unknown; HECKTOR uses the additional T/N fields |
| `reference_csv` | Lesion-independent cranio-cervical origin, patient RAS axes, and reference length in physical units |

The endpoint is recurrence-free survival for NPC/HECKTOR and progression-free survival for Foshan. Confirm endpoint definitions and time origins against the source clinical protocol. See [data conventions](docs/data.md) for the reference schema and preprocessing details.

## Training

Configure dataset paths and clinical provenance in a private copy of `configs/NPC.json`, `configs/HECKTOR.json`, or `configs/Foshan.json`. Paths inside the configuration are relative to that configuration file; `--output` is relative to the invoking shell.

```bash
python scripts/train.py --cohort NPC --fold 1 --arm D \
  --config configs/NPC.json --output outputs/npc_run
```

| Arm | Inputs | Sensitivity penalty |
|---|---|---|
| `H` | Clinical | None |
| `A` | Clinical + image | None |
| `G` | Clinical + geometry | None |
| `D` | Clinical + geometry + image | None |
| `E_S` | Clinical + geometry + image | Survival-probability sensitivity |
| `E_eta` | Clinical + geometry + image | Log-risk sensitivity |

For `E_S` or `E_eta`, set a positive `training.geometry_rho` in the configuration. For unregularized variants, set it to zero. Run each patient-level fold separately using `--fold 1` through `--fold 5` and aggregate fold results with `acsd_surv.metrics.summarize_folds`.

The main numerical options are documented in [configuration and outputs](docs/configuration.md). Changing these settings can change results; retain the effective configuration for every experiment.

## Outputs and evaluation

Each fold produces:

```text
<output>/<cohort>/fold_<k>/<arm>/
├── effective_config.json
├── selected.pt
├── inference_state.npz
├── test.npz
├── evaluation_details.json
├── result.json
└── history.json
```

`inference_state.npz` stores the clinical transform, Cox coefficients, and selected training baseline hazard. Evaluation reports Harrell's C-index, cumulative/dynamic iAUC, and IPCW IBS. Unsupported time-dependent metrics are recorded with their reasons instead of silently changing the evaluation grid. Probability-level loss weights may be supplied explicitly; reported IBS uses the standard time-integrated evaluation convention.

Default configuration windows are 12–36 months for NPC/HECKTOR and 24–48 months for Foshan. Specify horizons before test evaluation and establish event/control support on training and validation patients.

## Repository structure

```text
acsd_surv/
├── data/              # Cohort input adapters and physical-space preprocessing
├── geometry.py        # FEM operators and multiscale surface responses
├── model.py           # Clinical-residual predictor
├── survival.py        # PH objective and baseline hazard
├── regularization.py  # Sobolev sensitivity and contour perturbations
├── image_encoder.py   # Frozen MedicalNet features
├── training.py        # Fold training and model selection
└── metrics.py         # Survival evaluation and fold aggregation
configs/               # Cohort configurations
docs/                  # Input conventions and configuration reference
examples/              # Generated-data example
scripts/               # Command-line entry point
tests/                 # Numerical and integration tests
```

## Tests and reproducibility

```bash
python -m pytest -q
```

Tests cover FEM structure, shape derivatives, spectral caching, rank-one regularizer gradients, and synthetic training with a real clinical Cox fit. Image-branch integration tests use a small generated encoder fixture; they do not validate the external MedicalNet weights. Complete clinical result reproduction requires the corresponding authorized data, fixed splits, reference extraction procedure, and experimental configurations.

## Dependencies and acknowledgements

The implementation uses [PyTorch](https://pytorch.org/), [scikit-survival](https://scikit-survival.readthedocs.io/), NiBabel, SciPy, and scikit-image. MedicalNet source and weights are external dependencies and retain their original terms. See [third-party notices](THIRD_PARTY_NOTICES.md).

Publication and citation details will be added when available.
