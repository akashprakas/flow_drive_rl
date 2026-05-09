from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from navsim.agents.diffusiondrive.modules.conditional_unet1d import SinusoidalPosEmb
import torch.nn.functional as F
from navsim.agents.diffusiondrive.modules.blocks import linear_relu_ln, bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer
from navsim.agents.diffusiondrive.transfuser_model_v2 import (
    AgentHead,
    CustomTransformerDecoder,
    CustomTransformerDecoderLayer,
)
from typing import Any, List, Dict, Optional, Union


class FlowTransfuserModel(nn.Module):
    """DiffusionDrive with flow matching instead of DDIM."""

    def __init__(self, config: TransfuserConfig):
        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = FlowTrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1, 320),
        )

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
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

        trajectory = self._trajectory_head(trajectory_query, agents_query, cross_bev_feature, bev_spatial_shape, status_encoding[:, None], targets=targets, global_img=None)
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output


class FlowTrajectoryHead(nn.Module):
    """Flow matching trajectory head — truncated, in normalized space.

    Mirrors DiffusionDrive's approach:
    - Works in normalized [-1, 1] coordinate space (norm_odo/denorm_odo)
    - Truncated noise: t sampled from U(0, t_max) with t_max=0.05 (small perturbation)
    - At inference, starts from t=t_start with slight noise and refines in 1-2 steps
    - Model predicts clean trajectory (x_0) directly
    """

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str, config: TransfuserConfig):
        super(FlowTrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.ego_fut_mode = 20

        # Truncated flow: only add small noise (analogous to diffusion's timesteps 0-50/1000)
        self.t_max = 0.05
        # At inference, start from this noise level
        self.t_start = 0.02
        self.num_inference_steps = 2

        plan_anchor = np.load(plan_anchor_path)

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )  # 20, 8, 2
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 512),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)

        self.loss_computer = LossComputer(config)

    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_x = 2 * (odo_info_fut_x + 1.2) / 56.9 - 1
        odo_info_fut_y = 2 * (odo_info_fut_y + 20) / 46 - 1
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)

    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_x = (odo_info_fut_x + 1) / 2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1) / 2 * 46 - 20
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)

    def forward(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets=None, global_img=None) -> Dict[str, torch.Tensor]:
        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets, global_img)
        else:
            return self.forward_test(ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img)

    def forward_train(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets=None, global_img=None) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device

        # x_0 = normalized plan anchors (data side, t=0)
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)  # [B, 20, 8, 2]
        x_0_norm = self.norm_odo(plan_anchor)  # [B, 20, 8, 2] in [-1, 1]

        # Sample t ~ U(0, t_max) — truncated, small noise only
        t = torch.rand(bs, device=device) * self.t_max  # [B] in [0, 0.05]

        # Flow interpolation in normalized space: x_t = (1-t)*x_0 + t*noise
        noise = torch.randn_like(x_0_norm)  # unit Gaussian in normalized space
        t_expand = t[:, None, None, None]  # [B, 1, 1, 1]
        x_t_norm = (1 - t_expand) * x_0_norm + t_expand * noise

        # Clamp to valid range
        x_t_norm = torch.clamp(x_t_norm, -1, 1)

        # Convert back to world coords for the decoder (BEV attention needs world coords)
        noisy_traj_points = self.denorm_odo(x_t_norm)  # [B, 20, 8, 2]

        ego_fut_mode = noisy_traj_points.shape[1]

        # Encode noisy trajectory positions
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

        # Encode timestep — scale t to [0, 50] range to match diffusion's timestep range
        timesteps = (t * 1000).long()  # t_max=0.05 → max timestep=50
        time_embed = self.time_mlp(timesteps)
        time_embed = time_embed.view(bs, 1, -1)

        # Run stacked decoder — predicts clean poses (x, y, heading) directly
        poses_reg_list, poses_cls_list = self.diff_decoder(
            traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, global_img
        )

        # Compute loss on predicted clean trajectory vs ground truth
        trajectory_loss_dict = {}
        ret_traj_loss = 0
        for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
            trajectory_loss = self.loss_computer(poses_reg, poses_cls, targets, plan_anchor)
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss
            ret_traj_loss += trajectory_loss

        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
        return {"trajectory": best_reg, "trajectory_loss": ret_traj_loss, "trajectory_loss_dict": trajectory_loss_dict}

    def forward_test(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device

        # Use the mean perturbation at inference for repeatable PDM scores.
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        x_0_norm = self.norm_odo(plan_anchor)
        noise = torch.zeros_like(x_0_norm)
        x_t_norm = (1 - self.t_start) * x_0_norm + self.t_start * noise
        x_t_norm = torch.clamp(x_t_norm, -1, 1)

        # Convert to world coords for BEV attention
        noisy_traj_points = self.denorm_odo(x_t_norm)
        ego_fut_mode = noisy_traj_points.shape[1]

        # Single pass: the model directly predicts clean trajectory
        t_tensor = torch.full((bs,), int(self.t_start * 1000), device=device, dtype=torch.long)

        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

        time_embed = self.time_mlp(t_tensor)
        time_embed = time_embed.view(bs, 1, -1)

        poses_reg_list, poses_cls_list = self.diff_decoder(
            traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, global_img
        )

        # Directly use decoder output — it predicts clean x_0
        poses_reg = poses_reg_list[-1]
        poses_cls = poses_cls_list[-1]
        mode_idx = poses_cls.argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        return {"trajectory": best_reg}
