"""Flow matching model with GRPO/REINFORCE RL training.

Uses Gaussian Euler steps for stochastic policy:
  x_{next} = x_t + v_theta * dt + sigma_step * z
  log_prob = -0.5 * ||z||^2
"""
from typing import Dict, List, Optional
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

from navsim.agents.diffusiondrivev2.diffusiondrivev2_model_rl import _pairwise_scores, _pairwise_subscores


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
        self.num_groups = 2
        self.n_steps = 5
        self.sigma_init = 0.05
        self.sigma_step = 0.02
        self.temporal_discount = 0.8
        self.il_weight_positive = 0.8
        self.il_weight_negative = 1.0
        self.clip_ratio = 0.1
        self.kl_coeff = 0.1
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

        # PDM scorer for reward computation
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
        return self.forward_test(ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding)

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

    def _gaussian_log_prob(self, diff):
        """Gaussian transition log-prob up to constants shared by old/new policies."""
        return -0.5 * ((diff / self.sigma_step) ** 2).sum(dim=(-2, -1))

    def _compute_pdm_rewards(self, trajectories, metric_cache_path, targets):
        """Score trajectories using PDM. trajectories: [B, N, 8, 3] in world coords.
        Returns: proposal_scores [B, N], gt_scores [B], sub_rewards_list
        """
        bs = trajectories.shape[0]
        all_proposal_scores = []
        all_gt_scores = []
        all_subscores = []

        for b in range(bs):
            cache_path = metric_cache_path[b] if isinstance(metric_cache_path, (list, tuple)) else metric_cache_path
            with lzma.open(str(cache_path), "rb") as f:
                metric_cache = pickle.load(f)

            # Include GT as last trajectory for comparison
            gt_traj = targets["trajectory"][b].cpu().numpy()  # [8, 3]
            pred_trajs = trajectories[b].cpu().numpy()  # [N, 8, 3]
            all_trajs = np.concatenate([pred_trajs, gt_traj[None]], axis=0)  # [N+1, 8, 3]

            pdm_score_para(
                metric_cache=metric_cache,
                model_trajectory=all_trajs,
                future_sampling=self._simulator.proposal_sampling,
                simulator=self._simulator,
                scorer=self._scorer,
            )

            scores = _pairwise_scores(self._scorer)  # [N+1] (proposals + GT vs pdm_trajectory)
            subscores = _pairwise_subscores(self._scorer)

            # Separate proposal scores from GT score
            proposal_scores = scores[:-1]  # [N]
            gt_score = scores[-1]  # scalar

            all_proposal_scores.append(proposal_scores)
            all_gt_scores.append(gt_score)
            all_subscores.append({k: v[:-1] for k, v in subscores.items()})  # exclude GT from subscores

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

            # Initial noise: additive per-waypoint (more exploration for small G)
            # For full navtrain with G=4, consider switching to multiplicative
            noise = torch.randn_like(x_0_norm)
            x_t = (1 - self.sigma_init) * x_0_norm + self.sigma_init * noise
            x_t = torch.clamp(x_t, -1, 1)

            dt = self.sigma_init / self.n_steps
            chain_states = [x_t.clone()]
            sampled_next_states = []
            old_log_probs = []

            for step in range(self.n_steps):
                t_val = self.sigma_init - step * dt  # decreasing: 0.05, 0.04, ...

                # Decoder predicts clean trajectory
                poses_reg_list, poses_cls_list = self._run_decoder(
                    x_t, t_val, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, bs, n_modes,
                )
                x_0_pred = poses_reg_list[-1][..., :2]  # [B, G*20, 8, 2] world coords
                x_0_pred_norm = self.norm_odo(x_0_pred)

                # Velocity: direction toward predicted clean sample
                v_norm = (x_0_pred_norm - x_t) / max(t_val, 1e-5)

                # Stochastic Euler step. The next decoder state is clamped, but
                # PPO must score the unclamped Gaussian sample that was drawn.
                mean = x_t + v_norm * dt
                z = torch.randn_like(x_t)
                x_next_sampled = mean + self.sigma_step * z
                x_next = torch.clamp(x_next_sampled, -1, 1)

                log_prob = self._gaussian_log_prob(x_next_sampled - mean)  # [B, G*20]
                old_log_probs.append(log_prob)
                sampled_next_states.append(x_next_sampled.clone())
                chain_states.append(x_next.clone())
                x_t = x_next

            # Final trajectory from last decoder pass (with heading)
            final_traj = poses_reg_list[-1]  # [B, G*20, 8, 3]
            final_cls = poses_cls_list[-1]  # [B, G*20]... actually [B, n_modes]

            # PDM scoring
            rewards, gt_rewards, sub_rewards_list = self._compute_pdm_rewards(final_traj.detach(), metric_cache_path, targets)

            # Compute advantages
            advantages = self._compute_advantages(rewards, gt_rewards, sub_rewards_list, bs)  # [B, G*N, n_steps]

            # Stack old log probs: [B, G*N, n_steps]
            old_log_probs = torch.stack(old_log_probs, dim=-1)

        # === PASS 2: PPO loss on stored rollouts ===
        dt = self.sigma_init / self.n_steps
        adv = advantages.detach()
        target_traj = targets["trajectory"].unsqueeze(1).expand(-1, n_modes, -1, -1)
        total_loss = torch.zeros((), device=device)
        rl_loss_values = []
        il_loss_values = []
        kl_values = []

        for ppo_epoch in range(self.ppo_epochs):
            new_log_probs = []
            il_losses = []

            for step in range(self.n_steps):
                t_val = self.sigma_init - step * dt
                x_curr = chain_states[step].detach()
                x_next_sampled = sampled_next_states[step].detach()

                poses_reg_list, poses_cls_list = self._run_decoder(
                    x_curr, t_val, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, bs, n_modes,
                )
                x_0_pred = poses_reg_list[-1][..., :2]
                x_0_pred_norm = self.norm_odo(x_0_pred)

                v_norm = (x_0_pred_norm - x_curr) / max(t_val, 1e-5)
                mean = x_curr + v_norm * dt

                new_log_prob = self._gaussian_log_prob(x_next_sampled - mean)
                new_log_probs.append(new_log_prob)

                for poses_reg in poses_reg_list:
                    traj_l1 = F.l1_loss(poses_reg[..., :2], target_traj[..., :2], reduction='none')
                    il_losses.append(traj_l1.mean(dim=(1, 2, 3)))

            new_log_probs = torch.stack(new_log_probs, dim=-1)
            il_loss = torch.stack(il_losses, dim=0).mean(dim=0)

            # PPO clipped surrogate
            log_ratio = new_log_probs - old_log_probs.detach()
            ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))

            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            per_token_loss = -torch.min(ratio * adv, clipped_ratio * adv)

            mask_nonzero = adv != 0
            rl_loss_per_batch = (per_token_loss * mask_nonzero).sum(dim=(1, 2)) / mask_nonzero.sum(dim=(1, 2)).clamp_min(1)

            kl_div = (ratio - 1.0 - log_ratio).mean(dim=(1, 2))

            has_positive = (adv > 0).any(dim=(1, 2))
            il_weight = torch.where(has_positive, torch.tensor(self.il_weight_positive, device=device), torch.tensor(self.il_weight_negative, device=device))

            epoch_loss = rl_loss_per_batch + self.kl_coeff * kl_div + il_weight * il_loss
            total_loss = total_loss + epoch_loss.mean()
            rl_loss_values.append(rl_loss_per_batch.mean())
            il_loss_values.append(il_loss.mean())
            kl_values.append(kl_div.mean())

        # Average over PPO epochs
        loss = total_loss / self.ppo_epochs

        # Best trajectory for output
        with torch.no_grad():
            mode_idx = final_cls.argmax(dim=-1)
            mode_idx_exp = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
            best_reg = torch.gather(final_traj, 1, mode_idx_exp).squeeze(1)

        return {
            "trajectory": best_reg,
            "trajectory_loss": loss,
            "trajectory_loss_dict": {
                "rl_loss": torch.stack(rl_loss_values).mean(),
                "il_loss": torch.stack(il_loss_values).mean(),
                "kl_div": torch.stack(kl_values).mean(),
                "mean_reward": rewards.mean(),
            },
        }

    def forward_test(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding):
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
        mode_idx = poses_cls.argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        return {"trajectory": best_reg}
