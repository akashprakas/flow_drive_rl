from typing import Any, List, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.flow_drive_agent.flow_rl_model import FlowRLTransfuserModel
from navsim.agents.diffusiondrive.transfuser_callback import TransfuserCallback
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim


def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type_name = cfg.pop('type')
    return getattr(obj, type_name)(**cfg, **kwargs)


class FlowRLAgent(AbstractAgent):
    """Flow matching agent with GRPO/REINFORCE RL training."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()

        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._transfuser_model = FlowRLTransfuserModel(config)
        self.init_from_pretrained()
        self._freeze_backbone()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._set_frozen_modules_eval()
        return self

    def init_from_pretrained(self):
        if self._checkpoint_path:
            if torch.cuda.is_available():
                checkpoint = torch.load(self._checkpoint_path)
            else:
                checkpoint = torch.load(self._checkpoint_path, map_location=torch.device('cpu'))

            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('agent.', ''): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
        else:
            print("No checkpoint path provided. Initializing from scratch.")

    def _freeze_backbone(self):
        """Freeze everything except the trajectory head."""
        for name, param in self._transfuser_model.named_parameters():
            if '_trajectory_head' not in name:
                param.requires_grad = False

        self._set_frozen_modules_eval()

        trainable = sum(p.numel() for p in self._transfuser_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._transfuser_model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} parameters ({100*trainable/total:.1f}%)")

    def _set_frozen_modules_eval(self):
        for name, module in self._transfuser_model.named_modules():
            if name and not name.startswith('_trajectory_head'):
                module.eval()
        self._transfuser_model._trajectory_head.train()
        for module in self._transfuser_model._trajectory_head.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if torch.cuda.is_available():
            state_dict = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features, targets=None, metric_cache_path=None, token=None):
        return self._transfuser_model(features, targets=targets, metric_cache_path=metric_cache_path, token=token)

    def compute_loss(self, features, targets, predictions) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        device = next(self.parameters()).device
        loss_dict['loss'] = predictions.get('trajectory_loss', torch.zeros((), device=device))

        traj_loss_dict = predictions.get('trajectory_loss_dict', {})
        for k, v in traj_loss_dict.items():
            loss_dict[k] = v

        return loss_dict

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        params = [p for p in self._transfuser_model._trajectory_head.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self._lr, weight_decay=1e-4)
        scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=10, warmup_epochs=1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [TransfuserCallback(self._config)]
