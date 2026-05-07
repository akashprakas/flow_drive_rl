from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torchvision.models as models
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision import transforms

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class FrontCamFeatureBuilder(AbstractFeatureBuilder):
    """Feature builder: front camera image + ego status."""

    # ResNet-18 expects 224x224 but we use 256x256 for a slightly larger field
    IMAGE_H = 256
    IMAGE_W = 256

    def get_unique_name(self) -> str:
        return "front_cam_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        # current frame is the last element in the history list
        cam_f0 = agent_input.cameras[-1].cam_f0.image  # (H, W, 3) uint8

        resized = cv2.resize(cam_f0, (self.IMAGE_W, self.IMAGE_H))
        image_tensor = transforms.ToTensor()(resized)  # (3, H, W) float32 in [0,1]
        image_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )(image_tensor)

        ego_status = agent_input.ego_statuses[-1]
        status_tensor = torch.cat([
            torch.tensor(ego_status.ego_velocity, dtype=torch.float32),      # (2,)
            torch.tensor(ego_status.ego_acceleration, dtype=torch.float32),  # (2,)
            torch.tensor(ego_status.driving_command, dtype=torch.float32),   # (4,) one-hot
        ])  # (8,)  -- 2 velocity + 2 acceleration + 4 driving_command

        return {
            "image": image_tensor,
            "ego_status": status_tensor,
        }


class FrontCamTargetBuilder(AbstractTargetBuilder):
    """Target builder: future trajectory waypoints."""

    def __init__(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "front_cam_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self._trajectory_sampling.num_poses
        )
        return {"trajectory": torch.tensor(future_trajectory.poses, dtype=torch.float32)}


class FrontCamModel(torch.nn.Module):
    """ResNet-18 image encoder fused with ego status, predicting waypoints."""

    def __init__(self, num_waypoints: int, ego_status_dim: int = 8, hidden_dim: int = 256):
        super().__init__()

        # image encoder: ResNet-18 backbone, drop the final FC layer
        resnet = models.resnet18(weights=None)
        self._encoder = torch.nn.Sequential(*list(resnet.children())[:-1])  # output: (B, 512, 1, 1)
        image_feature_dim = 512

        # MLP head fusing image features + ego status → waypoints
        self._head = torch.nn.Sequential(
            torch.nn.Linear(image_feature_dim + ego_status_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_waypoints * 3),  # (x, y, heading) per waypoint
        )

    def forward(self, image: torch.Tensor, ego_status: torch.Tensor) -> torch.Tensor:
        image_features = self._encoder(image).flatten(1)           # (B, 512)
        fused = torch.cat([image_features, ego_status], dim=-1)    # (B, 512+8)
        waypoints = self._head(fused)                              # (B, num_waypoints*3)
        return waypoints


class FrontCamAgent(AbstractAgent):
    """Camera-only agent: ResNet-18 front camera + ego status → waypoints."""

    def __init__(
        self,
        lr: float = 1e-4,
        hidden_dim: int = 256,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        super().__init__(trajectory_sampling)
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._model = FrontCamModel(
            num_waypoints=trajectory_sampling.num_poses,
            hidden_dim=hidden_dim,
        )

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self._checkpoint_path is None:
            return
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=map_location)["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig(
            cam_f0=[3],       # index 3 = current frame (most recent in 4-step history)
            cam_l0=False, cam_l1=False, cam_l2=False,
            cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False,
            lidar_pc=False,
        )

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [FrontCamFeatureBuilder()]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [FrontCamTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        waypoints = self._model(
            features["image"].to(torch.float32),
            features["ego_status"].to(torch.float32),
        )
        return {"trajectory": waypoints.reshape(-1, self._trajectory_sampling.num_poses, 3)}

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return torch.optim.Adam(self._model.parameters(), lr=self._lr)
