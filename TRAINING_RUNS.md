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

---

## Run 2 – Advanced trainer with LR = 1e-5 (CLI default)

- **Date / context:** 2025-11-30, advanced trainer after lowering CLI LR.
- **Effective learning rate schedule:**
  - `initial_lr = 1e-5` via `QuantumAnnealer`.
  - Logged `learning_rate` at epoch 1: `≈ 9.68e-06` (matches schedule).
- **Other hyperparameters:**
  - Epochs: `30` (single graph, no validation split).
  - Batch size: `1` graph-level batch (hard-coded in `train_model`).
  - Curvature: `1.0` (Poincaré ball).
  - Optimizer: `HyperbolicAdam` + `QuantumAnnealer`.
  - Losses: `safe_kl_divergence` (policy), `MeanSquaredError` (value).
- **Key metrics (epoch 1):**
  - `loss ≈ 4.18`
  - `loss_fn_loss` (policy KL) ≈ `4.05`
  - `policy_mae ≈ 0.078`
  - `value_mae ≈ 0.420`
  - `value_mse_loss ≈ 0.259`
- **Behavior:**
  - Epoch 1: finite and reasonable losses with small LR.
  - Epoch 2 onward: all reported metrics (`loss`, `loss_fn_loss`, `policy_mae`, `value_mae`, `value_mse_loss`) become `nan` again despite:
    - Much smaller LR.
    - Numerically-stabilized `safe_kl_divergence`.
- **Assessment on hyperbolic architecture:**
  - Instability is no longer purely attributable to the nominal learning rate; even with ~1e-5, the combination of hyperbolic graph convolutions and the current target scaling appears to produce exploding activations or gradients.
  - Next likely stabilizing steps:
    - Constrain the policy head outputs more aggressively (e.g., explicit `softmax` with temperature and clipping inside the model, not just in the loss).
    - Consider per-layer or global gradient clipping inside `HyperbolicAdam`.
    - Normalize or rescale `per_gene_pert_reward` / `per_gene_pert_dist` targets further (e.g., temperature scaling or log-space losses) so that targets and logits live on a gentler scale.

---

## Architectural change – Policy head stabilization

- **Change:**
  - In `HyperPerturbModel` (`hyperpreturb/models/__init__.py`), the policy head dense layer now uses `activation="softmax"` and its outputs are additionally clipped to `[1e-6, 1]` and renormalized along the last axis.
- **Motivation:**
  - Force the policy head to emit well-formed per-gene probability distributions over perturbations, rather than unconstrained logits that can explode in magnitude.
  - Reduce the risk of `nan` values when combined with KL-based losses, even when logits become extreme due to hyperbolic graph dynamics.
- **Interaction with losses:**
  - Works together with `safe_kl_divergence` (which also clips and renormalizes) to provide two layers of numerical protection: one at the model output, one at the loss.
- **Expectation for future runs:**
  - With LR ≈ `1e-5` and the stabilized policy head, early epochs should maintain finite losses instead of going `nan` at epoch 2.

Append future runs below with: date, LR, curvature, key metrics from first few epochs, and qualitative stability notes.
