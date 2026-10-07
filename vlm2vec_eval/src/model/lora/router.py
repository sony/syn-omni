from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .global_vars import MODALITY_MASK_VAR


class Router(nn.Module, ABC):
    """
    Abstract base class for routing mechanisms.
    Subclasses must implement both forward() and get_weights().
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Defines the computation to get raw logits.
        """
        pass

    @abstractmethod
    def get_weights(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns normalized weights (probabilities) for lora selection.
        """
        pass


class LinearRouter(Router):
    # A simple linear-based router implementation.

    def __init__(self, hidden_dim: int, num_experts: int, routing_mode: str = "soft"):
        super().__init__()
        self.layer = nn.Linear(hidden_dim, num_experts)
        self.routing_mode = routing_mode

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.layer(x)

    def get_weights(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.forward(x)

        if self.routing_mode == 'soft':
            return torch.softmax(logits, dim=-1)  # (B, seq, 4)

        elif self.routing_mode == 'hard':
            # Straight-Through Estimator (STE) for hard-routing
            soft_w = torch.softmax(logits, dim=-1)
            expert_idx = torch.argmax(soft_w, dim=-1)
            hard_w = F.one_hot(expert_idx, num_classes=soft_w.size(-1)).to(soft_w.dtype)
            # STE Trick: Hard-routing in forward, Soft-routing gradients in backward
            return hard_w - soft_w.detach() + soft_w

        else:
            raise ValueError(f"Unknown routing mode: {self.routing_mode}")


class ModalityRouter(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        routing_mode: str = "soft",
        routing_activation_soft: str = "softmax",
        filter_absent_modality: bool = True,
        normalize_routing_weight: bool = True
    ):
        super().__init__()
        self.num_experts = num_experts
        self.routing_mode = routing_mode
        self.routing_activation_soft = routing_activation_soft
        self.filter_absent_modality = filter_absent_modality
        self.normalize_routing_weight = normalize_routing_weight

        if routing_mode == "soft":
            self.layer = nn.Linear(hidden_dim, num_experts)
        else:
            self.layer = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: (batch, hidden_dim) -> return: (batch, num_experts)
        assert self.layer is not None
        return self.layer(x)

    def get_weights(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_dim)
        modality_mask: (batch, num_experts) - Binary mask (0 or 1)
        """
        modality_mask = MODALITY_MASK_VAR.get()
        assert modality_mask is not None

        if self.routing_mode == 'soft':
            x_pooled = x[:, -1]  # last token pooling:  (batch, dim)
            logits = self.forward(x_pooled)  # logits: (batch, num_experts)

            # logits masking
            if self.filter_absent_modality:
                logits = logits.masked_fill(modality_mask == 0, float('-inf'))

            # activation
            if self.routing_activation_soft == "softmax":
                weights = torch.softmax(logits, dim=-1)
            else:
                assert self.routing_activation_soft == "sigmoid"
                # sigmoid + normalization (sum to 1)
                weights = torch.sigmoid(logits)
                if self.normalize_routing_weight:
                    weights = weights / (weights.sum(dim=-1, keepdim=True))

        elif self.routing_mode == 'hard':
            # hard routing, just averaging different modalities in a fixed ratio
            # 1. / num_present_modalities
            modality_mask = modality_mask.to(dtype=x.dtype, device=x.device)
            weights = modality_mask
            if self.normalize_routing_weight:
                weights = weights / (weights.sum(dim=-1, keepdim=True))

        else:
            raise ValueError(f"Unknown routing mode: {self.routing_mode}")

        return weights.unsqueeze(1)


class ProgressiveModalityRouter(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        loss_type: str = "margin",  # "margin" or "bce"
        loss_weight: float = 1.0,
        margin: float = 0.2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.loss_type = loss_type.lower()
        self.loss_weight = loss_weight
        self.margin = margin
        if loss_weight > 0.0:
            self.layer = nn.Linear(hidden_dim, num_experts)
        else:
            self.layer = None

        # to analyize later on
        self._saved_routing_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, hidden_dim) -> return logits: (batch, num_experts)
        return self.layer(x)

    def get_weights(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_dim)
        """

        # for eval: return sigmoid mask for soft routing only
        if self.loss_weight > 0.0:
            x_pooled = x[:, -1]
            logits = self.forward(x_pooled)
            target_mask = torch.sigmoid(logits)
        else:
            # hard routing; not used in evaluation phase
            modality_mask = MODALITY_MASK_VAR.get()
            assert modality_mask is not None
            target_mask = modality_mask.to(dtype=x.dtype)

        # save routing weights
        # self._saved_routing_weights = target_mask.float().detach().cpu().clone()

        # Return shape: [batch, 1, num_experts] for seq_len broadcasting
        return target_mask.unsqueeze(1)


def build_router(hidden_dim: int, num_experts: int, **kwargs):
    # configuration
    router_type = kwargs.get("router_type", "linear")
    routing_mode = kwargs.get("routing_mode", "soft")

    # hard-coding for interpretability
    if router_type == "linear":
        router = LinearRouter(hidden_dim, num_experts, routing_mode)

    elif router_type == "modality":
        routing_activation_soft = kwargs.get("routing_activation_soft", "softmax")
        filter_absent_modality = kwargs.get("filter_absent_modality", True)
        normalize_routing_weight = kwargs.get("normalize_routing_weight", True)

        router = ModalityRouter(
            hidden_dim,
            num_experts,
            routing_mode=routing_mode,
            routing_activation_soft=routing_activation_soft,
            filter_absent_modality=filter_absent_modality,
            normalize_routing_weight=normalize_routing_weight
        )

    elif router_type == "progressive":
        router_loss_type = kwargs.get("router_loss_type", "margin")
        router_loss_weight = kwargs.get("router_loss_weight", 0.0)
        router_loss_margin = kwargs.get("router_loss_margin", 0.2)

        router = ProgressiveModalityRouter(
            hidden_dim,
            num_experts,
            loss_type=router_loss_type,
            loss_weight=router_loss_weight,
            margin=router_loss_margin
        )

    else:
        raise ValueError(f"Unknown router type: {router_type}")
    return router
