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
from navsim.agents.diffusiondrive.transfuser_loss import transfuser_loss
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig



def build_from_config(obj,cfg,**kwargs):
    if cfg is None:
        return None
    
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)

    type = cfg.pop("type")
    return getattr(obj, type)(**cfg, **kwargs)


class FlowAgent(AbstractAgent):
    def  __int__(self,config: TransfuserConfig,lr: float, checkpoint_path : Optional[str] = None):
        super().__init__()

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

    
