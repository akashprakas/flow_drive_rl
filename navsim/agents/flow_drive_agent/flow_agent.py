from typing import Any, List, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.flow_drive_agent.flow_model import FlowTransfuserModel
from navsim.agents.diffusiondrive.transfuser_callback import TransfuserCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from navsim.agents.diffusiondrive.transfuser_loss import transfuser_loss
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
from navsim.common.dataclasses import  SensorConfig


def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)


class FlowAgent(AbstractAgent):
    """Agent using flow matching for trajectory prediction."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__(trajectory_sampling=config.trajectory_sampling)

        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._transfuser_model = FlowTransfuserModel(config)
        self.init_from_pretrained()

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
                print(f"Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        else:
            print("No checkpoint path provided. Initializing from scratch.")

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        return self._transfuser_model(features, targets=targets)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return transfuser_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return self.get_coslr_optimizers()

    def get_coslr_optimizers(self):
        optimizer_cfg = dict(
            type=self._config.optimizer_type,
            lr=self._lr,
            weight_decay=self._config.weight_decay,
            paramwise_cfg=self._config.opt_paramwise_cfg,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)

        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)

        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg['name']]

            for k, v in self._transfuser_model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg['name'].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                    if in_param_group:
                        pass
                if in_param_group:
                    params.append(v)
        else:
            params = self._transfuser_model.parameters()

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)

        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg['name'].items()):
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg['lr'] * pg_cfg['lr_mult']
                optimizer.add_param_group({'params': pg, **cfg})

        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=100,
            warmup_epochs=3,
        )

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        best_ckpt = ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename="best",
            save_last=True,
        )
        return [TransfuserCallback(self._config), best_ckpt]
