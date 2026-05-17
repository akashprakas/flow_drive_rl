import inspect
from typing import Dict, Tuple

import pytorch_lightning as pl
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        # Check once at init whether this agent's forward accepts targets / metric cache
        _fwd_params = inspect.signature(self.agent.forward).parameters
        self._forward_accepts_targets = "targets" in _fwd_params
        self._forward_accepts_metric_cache = "metric_cache_path" in _fwd_params

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        if len(batch) == 2:
            features, targets = batch
            pdm_token_path, token = None, None
        else:
            features, targets, pdm_token_path, token = batch

        if self._forward_accepts_metric_cache and pdm_token_path is not None:
            prediction = self.agent.forward(features, targets, metric_cache_path=pdm_token_path, token=token)
        elif self._forward_accepts_targets:
            prediction = self.agent.forward(features, targets)
        else:
            prediction = self.agent.forward(features)

        loss_or_dict = self.agent.compute_loss(features, targets, prediction)

        # Support agents that return a scalar loss (legacy) or a dict (diffusiondrive-style)
        if isinstance(loss_or_dict, dict):
            loss_dict = loss_or_dict
        else:
            loss_dict = {"loss": loss_or_dict}

        try:
            current_batch_size = next(iter(features.values())).shape[0]
        except (StopIteration, AttributeError):
            current_batch_size = 1

        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"{logging_prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=current_batch_size)

        return loss_dict["loss"]

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
