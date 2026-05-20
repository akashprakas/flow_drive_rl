"""Flow matching model with GRPO/REINFORCE RL training.

Uses Gaussian Euler steps for stochastic policy with time-dependent noise:
  sigma_t = sigma_base * sqrt(t / (1-t))   (marginal-preserving SDE, Flow-GRPO Eq. 8)
  x_{next} = mean * (1 + sigma_t * z)      (multiplicative, DDV2-style)
  log_prob = -0.5 * ||z||^2  (where z = (x_next - mean) / (sigma_t * |mean|))
"""
from typing import Dict, List, Optional
import copy
import math
import numpy as np
import pickle
import lzma
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from navsim.agents.diffusiondrive.modules.conditional_unet1d import SinusoidalPosEmb
from navsim.agents.diffusiondrive.modules.blocks import linear_relu_ln, bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer
from navsim.agents.diffusiondrive.transfuser_model_v2 import (
    AgentHead,
    CustomTransformerDecoder,
    CustomTransformerDecoderLayer,
)
from navsim.evaluate.pdm_score import pdm_score_para
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorerConfig
from navsim.planning.metric_caching.metric_cache import MetricCache
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    WeightedMetricIndex as WIdx,
)
#copied diffusiondrive v2
def _pairwise_subscores(scorer):
    """
    Extract all individual sub-scores from a PDMScorer that has already run
    score_proposals() in batch mode.

    Replicates the same GT-relative progress normalization as _pairwise_scores
    so that each sub-metric value is consistent with the final scalar score.
    Index 0 in the scorer is the PDM reference trajectory inserted by
    pdm_score_para; the returned arrays are sliced to [1:] so they align with
    the model trajectories passed to pdm_score_para.

    Returns:
        dict[str, np.ndarray], each of shape (G,) where G = num_proposals - 1.
        Keys: no_collision, drivable_area, progress, ttc, comfort,
              dir_weighted, final.
    """
    mm   = scorer._multi_metrics                # (3, N) — binary multiplicative metrics
    wm   = scorer._weighted_metrics.copy()      # must copy before modifying progress in-place
    prod = mm.prod(axis=0)                      # (N,) — product of all multiplicative metrics

    wcoef  = scorer._config.weighted_metrics_array
    thresh = scorer._config.progress_distance_threshold
    prog_raw = scorer._progress_raw             # (N,) — raw ego progress values

    # Normalize progress relative to GT (index 0), matching _pairwise_scores exactly.
    # If max(gt_prog, proposal_prog) > threshold: normalize by that max.
    # Otherwise: 0 if there was a collision/infraction, 1 if the car was stationary.
    raw_prog    = prog_raw * prod
    raw_prog_gt = raw_prog[0]
    max_pair    = np.maximum(raw_prog_gt, raw_prog[1:])
    norm_prog   = np.where(
        max_pair > thresh,
        raw_prog[1:] / (max_pair + 1e-6),
        np.where(prod[1:] == 0.0, 0.0, 1.0),
    ).astype(np.float64)
    wm[WeightedMetricIndex.PROGRESS, 1:] = norm_prog

    # Exclude TWO_FRAME_EXTENDED_COMFORT (computed later by scene aggregator)
    mask = np.ones(len(wcoef), dtype=bool)
    mask[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = False
    wscore = (wm[mask] * wcoef[mask, None]).sum(axis=0) / wcoef[mask].sum()

    return {
        "no_collision"  : mm[MultiMetricIndex.NO_COLLISION,        1:].copy(),
        "drivable_area" : mm[MultiMetricIndex.DRIVABLE_AREA,       1:].copy(),
        "progress"      : wm[WeightedMetricIndex.PROGRESS,         1:].copy(),
        "ttc"           : wm[WeightedMetricIndex.TTC,              1:].copy(),
        "comfort"       : wm[WeightedMetricIndex.HISTORY_COMFORT,  1:].copy(),
        "dir_weighted"  : wm[WeightedMetricIndex.LANE_KEEPING,     1:].copy(),
        "final"         : prod[1:] * wscore[1:],
    }

#copied diffusiondrive v2
def _pairwise_scores(scorer) -> np.ndarray:
    """
    Recompute per-proposal PDM scores using the intermediate state cached by
    a PDMScorer after score_proposals() has been called in batch mode.

    The scorer is called once with the PDM reference at index 0 followed by the
    model trajectories. This function pulls out the cached tensors and
    re-derives each model trajectory's final score relative to that reference.

    Args:
        scorer: PDMScorer instance with populated _multi_metrics,
                _weighted_metrics, and _progress_raw after score_proposals().

    Returns:
        np.ndarray of shape (N-1,) float32 — one PDM score per proposal (GT excluded).
    """
    # Retrieve cached intermediate tensors from the scorer
    mm   = scorer._multi_metrics            # (M_mul, N) — binary multiplicative metrics
    wm   = scorer._weighted_metrics.copy()  # (M_wgt, N) — copy so we can overwrite progress
    prog_raw = scorer._progress_raw         # (N,) — unnormalized ego progress distances
    weight_coef = scorer._config.weighted_metrics_array  # (M_wgt,) — per-metric weights

    N = mm.shape[1]                         # total proposals = 1 (GT) + G candidates
    assert N >= 2, "Need at least GT + 1 proposal"

    # Product of all binary multiplicative metrics (no_collision × drivable_area × ...)
    multi_prod = mm.prod(axis=0)            # (N,)

    # Normalize progress for each candidate relative to GT only.
    # This differs from the default scorer behavior which normalizes globally.
    raw_prog    = prog_raw * multi_prod     # zero out progress if any binary metric failed
    raw_prog_gt = raw_prog[0]

    max_pair    = np.maximum(raw_prog_gt, raw_prog[1:])           # (G,)
    thresh      = scorer._config.progress_distance_threshold

    # If the larger of (gt_prog, proposal_prog) exceeds threshold: use ratio.
    # Otherwise: 0 if proposal had a collision/infraction, 1 if both were stationary.
    norm_prog   = np.where(
        max_pair > thresh,
        raw_prog[1:] / (max_pair + 1e-6),
        np.where(multi_prod[1:] == 0.0, 0.0, 1.0),
    ).astype(np.float64)                                         # (G,)

    # Overwrite the progress row with the GT-relative normalized values
    wm[WIdx.PROGRESS, 1:] = norm_prog

    # Exclude TWO_FRAME_EXTENDED_COMFORT (computed later by scene aggregator, always 0 here)
    mask = np.ones(len(weight_coef), dtype=bool)
    mask[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = False

    # Weighted sum of soft metrics (matches PDMScorer._aggregate_pdm_scores)
    weighted_scores = (wm[mask, 1:] * weight_coef[mask, None]).sum(axis=0)
    weighted_scores /= weight_coef[mask].sum()                   # (G,)

    # Final PDM score = product of binary metrics × weighted soft metric score
    final_scores = multi_prod[1:] * weighted_scores              # (G,)

    return final_scores.astype(np.float32)                       # (G,)


# --- Persistent process pool for parallel PDM reward scoring ---
# ProcessPoolExecutor bypasses the Python GIL, giving true CPU parallelism for
# the shapely/numpy-heavy PDM scoring.
# * spawn context: child processes start fresh (safe when parent has a CUDA context)
# * initializer: pre-imports heavy modules so the first real task has no cold-start
# * Pool is created lazily on first use and kept alive for the process lifetime
_PDM_N_WORKERS: int = 8  # one process per batch item at batch_size=8
_pdm_executor = None  # type: Optional[object]  # ProcessPoolExecutor


def _pdm_worker_init():
    """Pre-import expensive navsim/nuplan modules in each worker process."""
    # Importing flow_rl_model itself would be circular; import just the deps we need.
    import lzma as _lzma  # noqa: F401
    import pickle as _pkl  # noqa: F401
    import numpy as _np  # noqa: F401
    from navsim.evaluate.pdm_score import pdm_score_para as _ps  # noqa: F401
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer as _SC  # noqa: F401
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator as _SIM  # noqa: F401


def _get_pdm_executor():
    """Return the persistent ProcessPoolExecutor, creating it on first call."""
    global _pdm_executor
    if _pdm_executor is None:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        ctx = multiprocessing.get_context("spawn")
        _pdm_executor = ProcessPoolExecutor(
            max_workers=_PDM_N_WORKERS,
            mp_context=ctx,
            initializer=_pdm_worker_init,
        )
    return _pdm_executor


def _score_one_item(args):
    """Score one batch item in a worker process. No shared state with parent.

    Creates PDMSimulator/PDMScorer locally — cheap since they hold only config.

    Args:
        args: (cache_path, pred_trajs [N,8,3], gt_traj [8,3],
               scoring_config PDMScorerConfig, proposal_sampling TrajectorySampling)

    Returns:
        (scores np.ndarray [N+1], subscores dict[str, np.ndarray [N+1]])
        Index 0..N-1 = model predictions, index N = GT trajectory.
    """
    import lzma as _lzma
    import pickle as _pkl
    import numpy as _np
    from navsim.evaluate.pdm_score import pdm_score_para
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

    cache_path, pred_trajs, gt_traj, scoring_config, proposal_sampling = args

    simulator = PDMSimulator(proposal_sampling)
    scorer = PDMScorer(proposal_sampling, scoring_config)

    with _lzma.open(str(cache_path), "rb") as f:
        metric_cache = _pkl.load(f)

    all_trajs = _np.concatenate([pred_trajs, gt_traj[None]], axis=0)  # [N+1, 8, 3]
    _ = pdm_score_para(
        metric_cache=metric_cache,
        model_trajectory=all_trajs,
        future_sampling=proposal_sampling,
        simulator=simulator,
        scorer=scorer,
    )
    # Re-import helpers since _pairwise_scores/_pairwise_subscores are defined
    # in this module; they are available because the worker imported flow_rl_model.
    from navsim.agents.flow_drive_agent.flow_rl_model import (
        _pairwise_scores,
        _pairwise_subscores,
    )
    scores = _pairwise_scores(scorer)
    subscores = _pairwise_subscores(scorer)
    return scores, subscores


class FlowRLTransfuserModel(nn.Module):
    """Flow matching model with RL training capability."""

    def __init__(self, config: TransfuserConfig):
        super().__init__()

        self._query_splits = [1, config.num_bounding_boxes]
        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(config.bev_features_channels, config.bev_features_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.bev_features_channels, config.num_bev_classes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Upsample(size=(config.lidar_resolution_height // 2, config.lidar_resolution_width), mode="bilinear", align_corners=False),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model, nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn, dropout=config.tf_dropout, batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(num_agents=config.num_bounding_boxes, d_ffn=config.tf_d_ffn, d_model=config.tf_d_model)

        self._trajectory_head = FlowRLTrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )
        self.bev_proj = nn.Sequential(*linear_relu_ln(256, 1, 1, 320))

    def forward(self, features: Dict[str, torch.Tensor], targets=None, metric_cache_path=None, token=None):
        camera_feature = features["camera_feature"]
        lidar_feature = features["lidar_feature"]
        status_feature = features["status_feature"]
        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1).permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:, :-1].permute(0, 2, 1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)
        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2, -1).permute(0, 2, 1))
        cross_bev_feature = cross_bev_feature.permute(0, 2, 1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)
        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        trajectory = self._trajectory_head(
            trajectory_query, agents_query, cross_bev_feature, bev_spatial_shape,
            status_encoding[:, None], targets=targets, metric_cache_path=metric_cache_path, token=token,
        )
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)
        return output


class FlowRLTrajectoryHead(nn.Module):
    """Flow matching trajectory head with Gaussian Euler RL."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str, config: TransfuserConfig):
        super().__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.ego_fut_mode = 20

        # RL parameters
        self.num_groups = 4
        self.n_steps = 5
        self.t_start = 0.15
        self.sigma_base = 0.12
        self.temporal_discount = 0.8
        self.clip_ratio = 0.1
        self.kl_coeff = 0.1
        self.il_coeff = 0.1
        # Multiple PPO optimizer epochs require manual optimization; in automatic
        # Lightning mode, replaying here only duplicates the same backward pass.
        self.ppo_epochs = 1

        plan_anchor = np.load(plan_anchor_path)
        self.plan_anchor = nn.Parameter(torch.tensor(plan_anchor, dtype=torch.float32), requires_grad=False)

        self.plan_anchor_encoder = nn.Sequential(*linear_relu_ln(d_model, 1, 1, 512), nn.Linear(d_model, d_model))
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(d_model), nn.Linear(d_model, d_model * 4), nn.Mish(), nn.Linear(d_model * 4, d_model))

        diff_decoder_layer = CustomTransformerDecoderLayer(num_poses=num_poses, d_model=d_model, d_ffn=d_ffn, config=config)
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)
        self.loss_computer = LossComputer(config)

        # Frozen reference decoder for drift-MSE KL (Flow-GRPO regularization)
        # KL(pi_theta || pi_ref) = ||mean_new - mean_ref||^2 / (2 * sigma_t^2)
        self._ref_diff_decoder = copy.deepcopy(self.diff_decoder)
        self._ref_diff_decoder.requires_grad_(False)
        self.kl_ref_coeff = 0.1

        # PDM scorer for reward computation (used to pass config/sampling to worker threads)
        proposal_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
        self._simulator = PDMSimulator(proposal_sampling)
        self._scorer = PDMScorer(proposal_sampling)

    def norm_odo(self, odo_info_fut):
        x = odo_info_fut[..., 0:1]
        y = odo_info_fut[..., 1:2]
        x = 2 * (x + 1.2) / 56.9 - 1
        y = 2 * (y + 20) / 46 - 1
        return torch.cat([x, y], dim=-1)

    def denorm_odo(self, odo_info_fut):
        x = odo_info_fut[..., 0:1]
        y = odo_info_fut[..., 1:2]
        x = (x + 1) / 2 * 56.9 - 1.2
        y = (y + 1) / 2 * 46 - 20
        return torch.cat([x, y], dim=-1)

    def forward(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets=None, metric_cache_path=None, token=None):
        if self.training:
            if metric_cache_path is None:
                raise ValueError("FlowRL training requires metric_cache_path from the cache-only dataloader.")
            if targets is None:
                raise ValueError("FlowRL training requires targets for IL regularization and GT reward comparison.")
            return self.forward_train_rl(ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets, metric_cache_path)
        return self.forward_test(ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets=targets)

    def _run_decoder(self, x_t_norm, t_val, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, bs, n_modes):
        """Run decoder on current noisy state, return predicted clean trajectory."""
        noisy_traj_points = self.denorm_odo(x_t_norm)
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, n_modes, -1)

        timesteps = torch.full((bs,), int(t_val * 1000), device=x_t_norm.device, dtype=torch.long)
        time_embed = self.time_mlp(timesteps).view(bs, 1, -1)

        poses_reg_list, poses_cls_list = self.diff_decoder(
            traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, None,
        )
        return poses_reg_list, poses_cls_list

    def _run_decoder_ref(self, x_t_norm, t_val, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, bs, n_modes):
        """Run frozen reference decoder — same inputs as _run_decoder but uses _ref_diff_decoder."""
        noisy_traj_points = self.denorm_odo(x_t_norm)
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, n_modes, -1)

        timesteps = torch.full((bs,), int(t_val * 1000), device=x_t_norm.device, dtype=torch.long)
        time_embed = self.time_mlp(timesteps).view(bs, 1, -1)

        poses_reg_list, poses_cls_list = self._ref_diff_decoder(
            traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, None,
        )
        return poses_reg_list, poses_cls_list

    def _sigma_t(self, t_val):
        """Time-dependent noise scale following the marginal-preserving SDE schedule.

        sigma_t = sigma_base * sqrt(t / (1-t)), matching Flow-GRPO Eq. 8.
        At t=0.15: 0.12 * sqrt(0.176) = 0.050
        At t=0.03: 0.12 * sqrt(0.031) = 0.021
        """
        return self.sigma_base * math.sqrt(t_val / (1.0 - t_val))

    def _latent_log_prob(self, z):
        """Standard normal log-prob on the 2D latent z ~ N(0, I).

        Args:
            z: [B, N, 1, 2] — the two latent random variables per mode.
        Returns:
            [B, N] — log-prob per mode (sum over the 2 independent coordinates).
        """
        return -0.5 * (z ** 2).sum(dim=(-2, -1))

    def _infer_z(self, x_next_sampled, mean, sigma_t):
        """Infer the latent z that would produce x_next_sampled under the current mean.

        Inverts: x_next = mean * (1 + sigma_t * z_broadcast)
        Solves:  z = (x_next / mean - 1) / sigma_t

        Since z is shared across waypoints [B, N, 1, 2], we average the inferred
        z across the 8 waypoints to get a robust estimate. The result is clamped
        to [-5, 5] to prevent overflow when mean is near zero but x_next is not.

        Note: this is a latent-likelihood surrogate, not an exact trajectory density.
        The shared-noise transform maps 2 latent dims into 16D trajectory state,
        making the true trajectory density singular. The latent formulation gives
        correct PPO gradients for policy improvement despite this.

        Args:
            x_next_sampled: [B, N, 8, 2]
            mean: [B, N, 8, 2]
            sigma_t: scalar

        Returns:
            z_inferred: [B, N, 1, 2], clamped to [-5, 5]
        """
        safe_mean = mean.clone()
        safe_mean[safe_mean.abs() < 1e-4] = 1e-4 * safe_mean[safe_mean.abs() < 1e-4].sign()
        safe_mean[safe_mean == 0] = 1e-4
        z_per_wp = (x_next_sampled / safe_mean - 1.0) / sigma_t
        z_inferred = z_per_wp.mean(dim=-2, keepdim=True)  # [B, N, 1, 2]
        z_inferred = z_inferred.clamp(-5.0, 5.0)
        return z_inferred

    def _bezier_xyyaw(self, xy8: torch.Tensor) -> torch.Tensor:
        """Compute heading from Bézier curve tangent direction (same as DDV2).

        Treats the 8 waypoints as control points, prepends (0,0) as origin,
        computes the analytical derivative of the 8th-order Bézier curve,
        and derives yaw = atan2(dy, dx) at each waypoint.

        Args:
            xy8: [B, N, 8, 2] — x,y positions in world coordinates.
        Returns:
            [B, N, 8, 3] — x, y, yaw (radians).
        """
        B, N, T, _ = xy8.shape
        device, dtype = xy8.device, xy8.dtype

        origin = torch.zeros(B, N, 1, 2, device=device, dtype=dtype)
        ctrl = torch.cat([origin, xy8], dim=-2)  # [B, N, 9, 2]
        n = ctrl.shape[-2] - 1  # 8

        delta = ctrl[..., 1:, :] - ctrl[..., :-1, :]  # [B, N, 8, 2]

        binom = torch.tensor(
            [math.comb(n - 1, i) for i in range(n)], device=device, dtype=dtype
        )  # (8,)

        t = torch.arange(1, n + 1, device=device, dtype=dtype) / n  # (8,)

        t_pow = t.view(-1, 1) ** torch.arange(0, n, device=device, dtype=dtype)
        one_pow = (1 - t).view(-1, 1) ** torch.arange(n - 1, -1, -1, device=device, dtype=dtype)
        basis = binom * t_pow * one_pow  # (8, 8)

        delta_exp = delta.unsqueeze(-3)  # [B, N, 1, 8, 2]
        basis_exp = basis.view(1, 1, 8, 8, 1)  # [1, 1, 8, 8, 1]

        deriv = n * (delta_exp * basis_exp).sum(dim=-2)  # [B, N, 8, 2]

        dx, dy = deriv[..., 0], deriv[..., 1]
        yaw = torch.atan2(dy, dx).unsqueeze(-1)  # [B, N, 8, 1]

        return torch.cat([xy8, yaw], dim=-1)  # [B, N, 8, 3]

    def _compute_pdm_rewards(self, trajectories, metric_cache_path, targets):
        """Score trajectories using PDM. trajectories: [B, N, 8, 3] in world coords.

        Batch items are scored in parallel via a persistent ThreadPoolExecutor.
        Each worker thread reuses its own PDMSimulator/PDMScorer (thread-local),
        so there is no shared mutable state and no per-step construction overhead.

        Returns: proposal_scores [B, N], gt_scores [B], sub_rewards_list
        """
        bs = trajectories.shape[0]

        args_list = []
        for b in range(bs):
            cache_path = metric_cache_path[b] if isinstance(metric_cache_path, (list, tuple)) else metric_cache_path
            gt_traj = targets["trajectory"][b].cpu().numpy()   # [8, 3]
            pred_trajs = trajectories[b].cpu().numpy()          # [N, 8, 3]
            args_list.append((
                cache_path,
                pred_trajs,
                gt_traj,
                self._scorer._config,
                self._simulator.proposal_sampling,
            ))

        executor = _get_pdm_executor()
        results = list(executor.map(_score_one_item, args_list))

        all_proposal_scores = []
        all_gt_scores = []
        all_subscores = []
        for scores, subscores in results:
            # scores: [N+1] — model predictions (0..N-1) + GT (N)
            proposal_scores = scores[:-1]  # [N]
            gt_score = scores[-1]          # scalar
            all_proposal_scores.append(proposal_scores)
            all_gt_scores.append(gt_score)
            all_subscores.append({k: v[:-1] for k, v in subscores.items()})  # exclude GT

        proposal_scores_tensor = torch.tensor(np.stack(all_proposal_scores), dtype=torch.float32, device=trajectories.device)
        gt_scores_tensor = torch.tensor(np.array(all_gt_scores), dtype=torch.float32, device=trajectories.device)
        return proposal_scores_tensor, gt_scores_tensor, all_subscores

    def _compute_advantages(self, rewards, gt_rewards, sub_rewards_list, bs):
        """Compute group-relative advantages with GT filtering, safety masking, and temporal discount."""
        G = self.num_groups
        N = self.ego_fut_mode
        device = rewards.device

        # Reshape: [B, G*N] → [B, G, N]
        reward_group = rewards.view(bs, G, N)
        reward_gt = gt_rewards.view(bs, 1, 1)  # [B, 1, 1] for broadcasting

        # Z-score normalization within each anchor across groups
        mean_r = reward_group.mean(dim=1, keepdim=True)
        std_r = reward_group.std(dim=1, keepdim=True)
        advantages = (reward_group - mean_r) / (std_r + 1e-4)

        # GT filtering: only reinforce trajectories that beat GT (DDV2 line 892)
        # Disabled on navmini (too sparse), enable for full navtrain
        # mask_better_than_gt = (reward_group > (reward_gt - 1e-6))  # [B, G, N]
        # advantages = advantages.clamp(min=0) * mask_better_than_gt.float()
        advantages = advantages.clamp(min=0)

        # Safety masking: collision or off-road → advantage = -1
        for b in range(bs):
            sub = sub_rewards_list[b]
            nc = torch.tensor(sub["no_collision"], dtype=torch.float32, device=device).view(G, N)
            dac = torch.tensor(sub["drivable_area"], dtype=torch.float32, device=device).view(G, N)
            unsafe = (nc != 1.0) | (dac != 1.0)
            advantages[b][unsafe] = -1.0

        # Flatten back: [B, G, N] → [B, G*N]
        advantages = advantages.view(bs, G * N)

        # Temporal discount: [n_steps] weights (later/cleaner steps weighted more)
        discount = torch.tensor(
            [self.temporal_discount ** (self.n_steps - i - 1) for i in range(self.n_steps)],
            device=device,
        )
        # Expand to [B, G*N, n_steps]
        advantages = advantages.unsqueeze(-1) * discount.unsqueeze(0).unsqueeze(0)

        return advantages

    def forward_train_rl(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets, metric_cache_path):
        bs = ego_query.shape[0]
        device = ego_query.device
        G = self.num_groups
        N = self.ego_fut_mode
        n_modes = G * N

        # === PASS 1: Rollout (no grad) ===
        with torch.no_grad():
            # Initialize anchors for G groups: [20, 8, 2] → [B, G, 20, 8, 2] → [B, G*20, 8, 2]
            plan_anchor = self.plan_anchor.unsqueeze(0).unsqueeze(0).repeat(bs, G, 1, 1, 1)  # [B, G, 20, 8, 2]
            plan_anchor = plan_anchor.reshape(bs, G * self.ego_fut_mode, 8, 2)  # [B, G*20, 8, 2]
            x_0_norm = self.norm_odo(plan_anchor)

            # Initial multiplicative noise: per-mode, shared across waypoints (DDV2-style)
            # This scales each trajectory coherently rather than adding independent per-waypoint jitter
            noise_x = torch.randn(bs, G * self.ego_fut_mode, 1, 1, device=device) * self.t_start + 1.0
            noise_y = torch.randn(bs, G * self.ego_fut_mode, 1, 1, device=device) * self.t_start + 1.0
            noise_mul = torch.cat([noise_x, noise_y], dim=-1).expand_as(x_0_norm)  # [B, G*20, 8, 2]
            x_t = x_0_norm * noise_mul
            x_t = torch.clamp(x_t, -1, 1)

            dt = self.t_start / self.n_steps
            chain_states = [x_t.clone()]
            sampled_next_states = []
            old_log_probs = []

            for step in range(self.n_steps):
                t_val = self.t_start - step * dt  # decreasing: 0.15, 0.12, ...

                # Decoder predicts clean trajectory
                poses_reg_list, poses_cls_list = self._run_decoder(
                    x_t, t_val, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, bs, n_modes,
                )
                x_0_pred = poses_reg_list[-1][..., :2]  # [B, G*20, 8, 2] world coords
                x_0_pred_norm = self.norm_odo(x_0_pred)

                # Velocity: direction toward predicted clean sample
                v_norm = (x_0_pred_norm - x_t) / max(t_val, 1e-5)

                # Time-dependent noise: more exploration at high t, less at low t
                sigma_t = self._sigma_t(t_val)

                # Multiplicative Euler step (DDV2-style): per-mode noise shared across waypoints
                # This preserves trajectory shape while exploring speed/lateral scaling
                mean = x_t + v_norm * dt
                z_x = torch.randn(bs, n_modes, 1, 1, device=device)
                z_y = torch.randn(bs, n_modes, 1, 1, device=device)
                z_latent = torch.cat([z_x, z_y], dim=-1)  # [B, G*20, 1, 2]
                z_mul = z_latent.expand_as(mean)  # [B, G*20, 8, 2]
                x_next_sampled = mean * (1.0 + sigma_t * z_mul)
                x_next = torch.clamp(x_next_sampled, -1, 1)

                log_prob = self._latent_log_prob(z_latent)  # [B, G*20]
                old_log_probs.append(log_prob)
                sampled_next_states.append(x_next_sampled.clone())
                chain_states.append(x_next.clone())
                x_t = x_next

            # Final trajectory: recompute heading from x,y geometry (DDV2-style Bézier tangent)
            final_xy = poses_reg_list[-1][..., :2]  # [B, G*20, 8, 2]
            final_traj = self._bezier_xyyaw(final_xy)  # [B, G*20, 8, 3]
            final_cls = poses_cls_list[-1]  # [B, G*20]... actually [B, n_modes]

            # PDM scoring
            rewards, gt_rewards, sub_rewards_list = self._compute_pdm_rewards(final_traj.detach(), metric_cache_path, targets)

            # Compute advantages
            advantages = self._compute_advantages(rewards, gt_rewards, sub_rewards_list, bs)  # [B, G*N, n_steps]

            # Stack old log probs: [B, G*N, n_steps]
            old_log_probs = torch.stack(old_log_probs, dim=-1)

        # === PASS 2: PPO loss on stored rollouts ===
        dt = self.t_start / self.n_steps
        adv = advantages.detach()
        total_loss = torch.zeros((), device=device)
        rl_loss_values = []
        il_loss_values = []
        kl_values = []
        kl_ref_values = []

        # WTA IL: find closest anchor mode per batch element (only that mode gets IL gradient)
        with torch.no_grad():
            anchor_dist = torch.linalg.norm(
                targets["trajectory"][:, None, :, :2] - self.plan_anchor[None, :, :, :], dim=-1
            ).mean(dim=-1)  # [B, 20]
            best_mode = anchor_dist.argmin(dim=-1)  # [B]

        for ppo_epoch in range(self.ppo_epochs):
            new_log_probs = []
            kl_drift_per_step = []
            il_losses = []

            for step in range(self.n_steps):
                t_val = self.t_start - step * dt
                x_curr = chain_states[step].detach()
                x_next_sampled = sampled_next_states[step].detach()

                poses_reg_list, poses_cls_list = self._run_decoder(
                    x_curr, t_val, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, bs, n_modes,
                )
                x_0_pred = poses_reg_list[-1][..., :2]
                x_0_pred_norm = self.norm_odo(x_0_pred)

                v_norm = (x_0_pred_norm - x_curr) / max(t_val, 1e-5)
                mean = x_curr + v_norm * dt

                # Time-dependent noise (must match rollout)
                sigma_t = self._sigma_t(t_val)

                # Drift-MSE KL to frozen reference decoder
                # Uses constant sigma_t^2 denominator (stable, avoids explosion near zero coords)
                with torch.no_grad():
                    ref_poses_reg_list, _ = self._run_decoder_ref(
                        x_curr, t_val, ego_query, agents_query, bev_feature,
                        bev_spatial_shape, status_encoding, bs, n_modes,
                    )
                    x_0_pred_ref = ref_poses_reg_list[-1][..., :2]
                    x_0_pred_ref_norm = self.norm_odo(x_0_pred_ref)
                    v_norm_ref = (x_0_pred_ref_norm - x_curr) / max(t_val, 1e-5)
                    mean_ref = x_curr + v_norm_ref * dt

                kl_drift = ((mean - mean_ref) ** 2).mean() / (2.0 * sigma_t ** 2)
                kl_drift_per_step.append(kl_drift)

                # Infer latent z from stored x_next_sampled under new mean, score under N(0,I)
                z_inferred = self._infer_z(x_next_sampled, mean, sigma_t)  # [B, N, 1, 2]
                new_log_prob = self._latent_log_prob(z_inferred)  # [B, N]
                new_log_probs.append(new_log_prob)

                # WTA IL: only regularize the closest-to-GT mode (across G groups)
                for poses_reg in poses_reg_list:
                    poses_reshaped = poses_reg.view(bs, G, N, 8, 3)  # [B, G, 20, 8, 3]
                    idx = best_mode[:, None, None, None, None].expand(-1, G, 1, 8, 3)
                    best_mode_pred = poses_reshaped.gather(2, idx).squeeze(2)  # [B, G, 8, 3]
                    traj_l1 = F.l1_loss(best_mode_pred[..., :2], targets["trajectory"][:, None, :, :2], reduction='mean')
                    il_losses.append(traj_l1)

            new_log_probs = torch.stack(new_log_probs, dim=-1)
            kl_drift_loss = torch.stack(kl_drift_per_step).mean()
            il_loss = torch.stack(il_losses).mean()

            # PPO clipped surrogate
            log_ratio_raw = new_log_probs - old_log_probs.detach()
            log_ratio = log_ratio_raw.clamp(-5.0, 5.0)
            ratio = torch.exp(log_ratio)

            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            per_token_loss = -torch.min(ratio * adv, clipped_ratio * adv)

            mask_nonzero = adv != 0
            rl_loss_per_batch = (per_token_loss * mask_nonzero).sum(dim=(1, 2)) / mask_nonzero.sum(dim=(1, 2)).clamp_min(1)

            kl_div = (ratio - 1.0 - log_ratio).mean(dim=(1, 2))

            epoch_loss = rl_loss_per_batch + self.kl_coeff * kl_div + self.kl_ref_coeff * kl_drift_loss + self.il_coeff * il_loss
            total_loss = total_loss + epoch_loss.mean()
            rl_loss_values.append(rl_loss_per_batch.mean())
            il_loss_values.append(il_loss)
            kl_values.append(kl_div.mean())
            kl_ref_values.append(kl_drift_loss)

        # Average over PPO epochs
        loss = total_loss / self.ppo_epochs

        # Best trajectory for output
        with torch.no_grad():
            mode_idx = final_cls.argmax(dim=-1)
            mode_idx_exp = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
            best_reg = torch.gather(final_traj, 1, mode_idx_exp).squeeze(1)

        # Diagnostic: fraction of log-ratios that hit the clamp (indicates z saturation)
        with torch.no_grad():
            clamp_frac = (log_ratio_raw.abs() > 5.0).float().mean()

        return {
            "trajectory": best_reg,
            "trajectory_loss": loss,
            "trajectory_loss_dict": {
                "rl_loss": torch.stack(rl_loss_values).mean(),
                "il_loss_wta": torch.stack(il_loss_values).mean(),
                "kl_div": torch.stack(kl_values).mean(),
                "kl_ref": torch.stack(kl_ref_values).mean(),
                "mean_reward": rewards.mean(),
                "log_ratio_clamp_frac": clamp_frac,
            },
        }

    def forward_test(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets=None):
        """Deterministic inference — same as flow IL model."""
        bs = ego_query.shape[0]
        device = ego_query.device

        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        x_0_norm = self.norm_odo(plan_anchor)
        noise = torch.zeros_like(x_0_norm)
        x_t = (1 - 0.02) * x_0_norm + 0.02 * noise
        x_t = torch.clamp(x_t, -1, 1)

        noisy_traj_points = self.denorm_odo(x_t)
        ego_fut_mode = noisy_traj_points.shape[1]

        t_tensor = torch.full((bs,), int(0.02 * 1000), device=device, dtype=torch.long)
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

        time_embed = self.time_mlp(t_tensor).view(bs, 1, -1)

        poses_reg_list, poses_cls_list = self.diff_decoder(
            traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, None,
        )

        poses_reg = poses_reg_list[-1]
        poses_cls = poses_cls_list[-1]

        # Recompute heading from x,y geometry (Bézier tangent, same as training)
        poses_xy = poses_reg[..., :2]  # [B, 20, 8, 2]
        poses_with_yaw = self._bezier_xyyaw(poses_xy)  # [B, 20, 8, 3]

        mode_idx = poses_cls.argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
        best_reg = torch.gather(poses_with_yaw, 1, mode_idx).squeeze(1)

        output = {"trajectory": best_reg}
        if targets is not None:
            output["trajectory_loss"] = F.l1_loss(best_reg, targets["trajectory"])
        return output
