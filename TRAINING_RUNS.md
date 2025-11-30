# HyperPerturb Training Runs and Hyperparameter Notes

This file tracks training runs, learning rates, and hyperparameter behavior for the hyperbolic architecture.

Each time you provide logs from a run, we can append a new entry here.

---

## Run 1 – Baseline advanced trainer (before LR tuning)

- **Date / context:** 2025-11-30, advanced trainer (`hyperpreturb.models.train.train_model`) via `scripts/train_model.py`.
- **Effective learning rate:** `5e-4` (CLI default at the time).
- **Optimizer:** `HyperbolicAdam` with `QuantumAnnealer` schedule, curvature `1.0`.
- **Losses:**
  - Policy: `safe_kl_divergence(name="policy_kld")` wrapping `KLDivergence`.
  - Value: `MeanSquaredError(name="value_mse")`.
- **Key metrics (epoch 1):**
  - `loss ≈ 4.15–4.53`
  - `policy_kld` / `loss_fn_loss ≈ 4.0–4.4`
  - `policy_mae ≈ 0.077–0.084`
  - `value_mae ≈ 0.40`
  - `value_mse_loss ≈ 0.245`
- **Behavior:**
  - Epoch 1: finite losses.
  - Epoch 2 onward: all losses and metrics become `nan` despite KL stabilization, indicating optimizer step size is still too large for this hyperbolic setup.
- **Assessment:**
  - Hyperbolic architecture trains but is numerically fragile at high learning rates.
  - Reducing the effective learning rate (and possibly adding gradient clipping) is necessary for stable training.

---

## Current defaults (after CLI update)

- **CLI default learning rate:** `1e-5` in `scripts/train_model.py`.
- **Model-side default learning rate:** `1e-5` in `hyperpreturb/models/train.py`.
- **Expected effect:**
  - Much smaller initial step size for `HyperbolicAdam + QuantumAnnealer`.
  - Should prevent immediate divergence to `nan` after the first epoch and allow observation of true convergence behavior.

Append future runs below with: date, LR, curvature, key metrics from first few epochs, and qualitative stability notes.
