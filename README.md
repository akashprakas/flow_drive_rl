# Flow_drive_rl

An ablation study on [DiffusionDriveV2](https://github.com/hustvl/DiffusionDrive) that swaps its DDIM diffusion sampler for a rectified flow matching model while keeping everything else backbone, anchor init, GRPO training loop, PDM reward identical. Built on the [NAVSIM](https://github.com/autonomousvision/navsim) benchmark.

---

## What This Project Is

DiffusionDriveV2 achieves 91.2 PDMS / 85.5 EPDMS using DDIM + GRPO. **flow_drive_rl** makes a single architectural change: replace the DDIM denoiser with a **rectified flow matching ODE**, keeping the TransFuser backbone, 20 K-Means anchors, and GRPO fine-tuning pipeline unchanged. This isolates the effect of the generative model on RL training stability and final performance.

The core research question: *does GRPO converge faster and achieve higher EPDMS with a flow matching policy than with a diffusion policy, all else being equal?*

---

## Architecture Overview

```
4 cameras + LiDAR
      │
  TransFuser Backbone (ResNet-34 + BEV)
      │
  BEV Features [B, 256, H, W]
      │
  Flow Trajectory Head
      │
  20 K-Means Anchors × G groups = 80 proposals
      │
  Rectified Flow ODE: t: 0 (noise) → 1 (data)
      │
  PDM Scorer → best trajectory
```

The trajectory head works in a normalized coordinate space (`norm_odo`/`denorm_odo`) and uses the 20 K-Means anchors pre-computed from the navtrain split (`kmeans_navsim_traj_20.npy`) to initialize the ODE from near-feasible starting points rather than pure Gaussian noise.

---

## Codebase Structure

```
navsim/agents/flow_drive_agent/
  flow_model.py       — IL flow model (FlowTransfuserModel + FlowTrajectoryHead)
  flow_agent.py       — Lightning wrapper for IL training
  flow_rl_model.py    — RL flow model (FlowRLTransfuserModel + FlowRLTrajectoryHead)
  flow_rl_agent.py    — Lightning wrapper for GRPO fine-tuning

navsim/planning/script/config/common/agent/
  flow_agent.yaml     — Hydra config for IL agent

scripts/
  training/           — Training launch scripts
  evaluation/         — PDM score evaluation scripts
```

---

## Stage 1A: IL Pretraining with Flow Matching

**File:** `flow_model.py` / `flow_agent.py`

The IL model replaces DiffusionDrive's DDIM denoiser with a rectified flow formulation. The key differences:

**Training:**
- Sample `t ~ U(0, 0.05)` (truncated — only small noise levels, analogous to diffusion's first 50/1000 timesteps)
- Interpolate: `x_t = (1 - t) * x_0 + t * noise`, where `x_0` is the K-Means anchor (data side)
- The decoder directly predicts the clean trajectory `x_0` (flow target: `v = x_0 - noise`)
- Loss: classification (which anchor is closest to GT) + regression (L1 on predicted poses)

**Inference:**
- Start from `x_t = (1 - 0.02) * anchor + 0.02 * 0` (tiny perturbation from anchors)
- Single decoder pass — no iterative ODE at inference, just one-shot prediction
- Select best mode by argmax of classification scores

**What this gives:** a multimodal trajectory generator that proposes 20 diverse candidates (one per K-Means anchor) and selects the highest-confidence one. This is the IL baseline that GRPO then fine-tunes.

---

## Stage 1B: GRPO Fine-tuning

**File:** `flow_rl_model.py` / `flow_rl_agent.py`

GRPO replaces the supervised loss with a PDM reward signal. The backbone is **frozen** — only the trajectory head is trained during RL.

### Rollout

For each scene, the model generates `G × 20 = 40` trajectory proposals (G=2 groups, same 20 anchors repeated with different noise seeds):

1. Initialize from anchors + Gaussian noise: `x_t = (1 - σ_init) * anchor + σ_init * noise`
2. Run 5 Euler ODE steps with **stochastic transitions** (Gaussian Euler):
   ```
   v = (x_0_pred - x_t) / t          # velocity toward predicted clean trajectory
   mean = x_t + v * dt
   x_next = mean + σ_step * z         # z ~ N(0,I) — adds exploration noise
   log_prob += -0.5 * ||z||²          # Gaussian log-prob for importance ratio
   ```
3. Score all 40 final trajectories with the PDM scorer (collision, drivable area, TTC, ego progress)

### Advantage Computation

Advantages use **intra-group z-score normalization** — each trajectory is compared only against the other trajectories that started from the same anchor group. This gives a relative signal: "was this trajectory better than others exploring the same driving mode?"

Additional rules:
- Any trajectory with `NC < 1.0` or `DAC < 1.0` (collision or off-road) gets advantage `= -1.0` regardless of z-score
- Only trajectories with positive advantage contribute to the policy gradient (clamp at 0)

### PPO Loss

Standard clipped surrogate with `clip_ratio=0.1`:
```
ratio = exp(log_prob_new - log_prob_old)
loss_rl = -min(ratio * A, clip(ratio, 0.9, 1.1) * A)
```

Plus two regularization terms:
- **IL loss**: L1 between predicted trajectory and GT (weight 0.8 when any positive reward exists, 1.0 otherwise)
- **KL penalty**: `ratio - 1 - log(ratio)` averaged over steps (coefficient 0.1)

---


## What's Implemented 

### Done
- [x] IL flow matching model with K-Means anchor init
- [x] `norm_odo`/`denorm_odo` normalization (borrowed from GuideFlow)
- [x] GRPO training loop: Gaussian Euler rollout, PDM scoring, intra-group z-score, PPO clipping
- [x] Backbone freezing during RL (only trajectory head trains)
- [x] Safety masking (collision/off-road → hard -1.0)
- [x] IL regularization + KL penalty alongside RL loss
