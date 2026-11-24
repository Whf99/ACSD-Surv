# Configuration and outputs

All path fields are resolved relative to the JSON configuration file. Command-line --output is resolved relative to the invoking shell. A run cannot reuse a nonempty fold/arm output directory.

| Option | Meaning |
|---|---|
| evaluation_grid | Positive ordered times in months |
| horizon_weights | Positive weights summing to one; null uses normalized trapezoidal weights |
| ipcw_clip | Lower bound for censoring probability in the training loss |
| clinical_anchor_alpha | Ridge coefficient for the training clinical Cox fit |
| training.lr | Adam learning rate |
| training.l2_lambda | Explicit L2 coefficient over trainable model parameters |
| training.geometry_rho | Fixed sensitivity coefficient; positive for E_S/E_eta, zero otherwise |
| training.epochs / patience | Maximum updates and validation early-stopping patience |
| training.batch_size | Preprocessing/image-loading batch size |
| training.replay_block_size | Score-gradient replay block size; Cox risk sets remain full-training-set |
| model.sobolev_ell / sobolev_power | Fixed perturbation metric parameters |
| model.geometry.learn_operator | Enable learned conductivity; false uses isotropic diffusion |
| model.geometry.isotropic_cache_dir | Optional fixed-mesh eigenbasis cache |

The residual terminal layer is initialized to zero. The frozen image backbone and fitted clinical anchor are excluded from L2. The baseline hazard is differentiable during training and fixed to training-split estimates during validation/test prediction and contour sensitivity calculations.

At sobolev_power=0, the perturbation metric reduces to diagonal normalized lumped mass, so evaluation needs no dense linear solve. Fractional nonnegative powers are supported; the smooth-surface C2 stability condition requires power>3. Fixed-isotropic eigenbasis caches are bypassed when differentiating through mesh coordinates.

Sensitive generated outputs include selected.pt, inference_state.npz, test.npz and per-patient perturbation.json. They are ignored by Git. result.json contains fold metrics; evaluation_details.json includes metric availability and censoring-support information. A missing metric remains null and is not replaced by a fabricated score. Five-fold aggregation requires five valid folds for an official mean and sample standard deviation.
