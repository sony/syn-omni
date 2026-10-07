import math

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# from .global_vars import MODALITY_MASK_VAR
from .router import build_router


class LoRAAdaptation(nn.Module):
    """
    Unified LoRA/DoRA block. 
    Can wrap a base_layer (for DoRA/Standard LoRA) or act as a standalone delta provider.
    """

    def __init__(
        self,
        base_layer: Optional[nn.Linear],
        rank: int,
        alpha: int,
        dropout_p: float = 0.1,
        use_dora: bool = False,
        in_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        activation: Optional[Callable] = None
    ):
        super().__init__()
        self.base_layer = base_layer
        # will use dora only as standalone layer. not moe, not stacked.
        self.use_dora = use_dora and (base_layer is not None)
        # disable dropout when dora is activated
        dropout_p = 0.0 if self.use_dora else dropout_p
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Setup dimensions
        if base_layer is not None:
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features

            # Freeze base weight
            self.base_layer.weight.requires_grad = False
            if self.base_layer.bias is not None:
                self.base_layer.bias.requires_grad = False
        else:
            assert in_dim is not None and out_dim is not None
            self.in_features = in_dim
            self.out_features = out_dim

        # Core parameters
        self.lora_A = nn.Parameter(torch.empty(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, self.out_features))
        self.lora_dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # DoRA Magnitude (m)
        if use_dora and base_layer is not None:
            self.magnitude = nn.Parameter(
                torch.norm(self.base_layer.weight.data, p=2, dim=1).detach()
            )
        else:
            self.register_parameter("magnitude", None)
            self.use_dora = False
        if activation is None:
            activation = nn.Identity()
        self.activation = activation

    def get_delta(self, x):
        # input dropout -> A -> activation -> B -> scale (alpha/rank)
        # NOTE: activation position: rank space for sure?
        return self.activation(self.lora_dropout(x) @ self.lora_A) @ self.lora_B * self.scaling

    def forward(self, x):
        if self.base_layer is None:
            assert not self.use_dora
            return self.get_delta(x)

        if self.use_dora:
            # DoRA Logic: W' = m * (W + BA) / ||W + BA||
            delta_w = (self.lora_B.transpose(0, 1) @ self.lora_A.transpose(0, 1)) * self.scaling
            combined_weight = self.base_layer.weight + delta_w
            column_norm = torch.norm(combined_weight, p=2, dim=1).detach()
            scale_factor = self.magnitude / (column_norm + 1e-6)
            dora_weight = combined_weight * scale_factor.unsqueeze(1)
            return F.linear(x, dora_weight, self.base_layer.bias)
        else:
            # Standard Path: Base(x) + Delta(x)
            return self.base_layer(x) + self.get_delta(x)

    @torch.no_grad()
    def get_merged_weight(self) -> torch.Tensor:
        """
        Compute the merged weight matrix based on LoRA or DoRA logic.
        """
        # Calculate Delta W: (B @ A.T) * scaling -> [Out, In]
        delta_w = (self.lora_B.transpose(0, 1) @ self.lora_A.transpose(0, 1)) * self.scaling
        if self.base_layer is None:
            return delta_w

        if self.use_dora:
            raise NotImplementedError  # not use dora

            # DoRA Merge: W_merged = m * (W + delta_W) / ||W + delta_W||
            combined_w = self.base_layer.weight + delta_w
            column_norm = torch.norm(combined_w, p=2, dim=1, keepdim=True)

            # Apply magnitude scaling
            merged_w = self.magnitude.unsqueeze(1) * (combined_w / (column_norm + 1e-6))
            return merged_w
        else:
            # Standard LoRA Merge: W_merged = W + delta_W
            return self.base_layer.weight + delta_w

    def to_linear(self, name: str) -> nn.Linear:
        # Convert this adaptation layer back to a standard nn.Linear.
        merged_w = self.get_merged_weight()
        new_linear = nn.Linear(
            self.in_features, self.out_features, bias=(self.base_layer.bias is not None)
        )
        new_linear.weight.data.copy_(merged_w)
        if self.base_layer.bias is not None:
            new_linear.bias.data.copy_(self.base_layer.bias.data)

        # Ensure the new layer is in the same device/dtype
        new_linear.to(device=merged_w.device, dtype=merged_w.dtype)
        return new_linear


class MixtureOfLoRA(nn.Module):
    """ Batch computation version """

    def __init__(self, base_layer: nn.Linear = None, in_features=None, out_features=None, **kwargs):
        super().__init__()
        # Configuration
        self.rank = kwargs.get("rank", 16)
        self.alpha = kwargs.get("alpha", 32)
        self.scaling = self.alpha / self.rank
        self.num_experts = kwargs.get("num_specific_lora", 4)
        dropout_p = kwargs.get("lora_dropout", 0.1)

        # Base layer setup
        self.base_layer = base_layer
        if base_layer is not None:
            self.base_layer.weight.requires_grad = False
            if self.base_layer.bias is not None:
                self.base_layer.bias.requires_grad = False
            in_features = base_layer.in_features
            out_features = base_layer.out_features

        assert (in_features is not None) and (out_features is not None)
        self.in_features = in_features
        self.out_features = out_features

        # Router
        self.router = build_router(hidden_dim=in_features, num_experts=self.num_experts, **kwargs)

        # [num_experts, in_features, rank] -> [num_experts, rank, out_features]
        self.lora_A = nn.Parameter(torch.empty(self.num_experts, in_features, self.rank))
        self.lora_B = nn.Parameter(torch.empty(self.num_experts, self.rank, out_features))

        self.reset_parameters()
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

        # logging
        self._saved_res_b = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def get_delta(self, x, return_intermediate=False, **kwargs):

        # [Batch, Seq, NumExperts]
        routing_weights = self.router.get_weights(x)
        x_dropped = self.dropout(x)

        # x_dropped: [Batch, Seq, In_Features]
        # lora_A: [NumExperts, In_Features, Rank]
        # lora_B: [NumExperts, Rank, Out_Features]

        # e: experts, b: batch, s: seq, i: in_features, r: rank, o: out_features
        res_a = torch.einsum("bsi,eir->ebsr", x_dropped, self.lora_A)
        res_b = torch.einsum("ebsr,ero->ebso", res_a, self.lora_B)

        # [NumExperts, Batch, Seq, Out_Features] -> [Batch, Seq, NumExperts, Out_Features]
        res_b = res_b.permute(1, 2, 0, 3)

        # routing_weights: [Batch, Seq, NumExperts] -> [Batch, Seq, 1, NumExperts]
        # combined: [Batch, Seq, 1, Out_Features] -> [Batch, Seq, Out_Features]
        combined_delta = torch.matmul(routing_weights.unsqueeze(-2), res_b).squeeze(-2)
        if return_intermediate:
            return combined_delta * self.scaling, res_b
        return combined_delta * self.scaling, None

    def forward(self, x, **kwargs):
        delta = self.get_delta(x, **kwargs)
        if self.base_layer is None:
            return delta
        return self.base_layer(x) + delta

    def to_linear(self, name: str):
        # merge is impossible, as the effective lora parameter changes depending on its input
        return None


class MoELoRAwithShared(nn.Module):

    def __init__(self, base_layer: nn.Linear = None, **kwargs):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Configuration
        shared_rank = kwargs.get("rank", 16)
        shared_alpha = kwargs.get("alpha", 32)
        specific_rank_ratio = kwargs.get("specific_rank_ratio", 2)

        # share the dropout layer across shared_lora and moe lora
        dropout_p = kwargs.get("lora_dropout", 0.1)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

        self.shared_lora = LoRAAdaptation(
            base_layer=None,
            rank=shared_rank,
            alpha=shared_alpha,
            dropout_p=0.0,
            in_dim=in_features,
            out_dim=out_features
        )
        self.is_merged = False  # is True when evaluation-only mode

        # Experts
        scaling = int(shared_alpha / shared_rank)
        spec_rank = int(shared_rank / specific_rank_ratio)
        spec_alpha = spec_rank * scaling

        # re-assigning dropout, rank, and alpha values for expert lora
        kwargs.pop("rank", 16)
        kwargs.pop("alpha", 32)
        kwargs.pop("lora_dropout", 0.1)
        self.expert_moe = MixtureOfLoRA(
            base_layer=None,
            in_features=in_features,
            out_features=out_features,
            rank=spec_rank,
            alpha=spec_alpha,
            lora_dropout=0.0,
            **kwargs
        )

        # A-matrices orthogonal loss
        self.ortho_loss_weight = kwargs.get("ortho_loss_weight", 0.0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.base_layer(x)
        # base_out = out.clone()
        x_dropped = self.dropout(x)
        shared_out = None
        if not self.is_merged:
            shared_out = self.shared_lora(x_dropped)
            out = out + shared_out
        expert_output, expert_delta = self.expert_moe.get_delta(x_dropped)

        return out + expert_output

    @torch.no_grad()
    def merge_shared_and_unload(self, name):
        if self.is_merged:
            return

        shared_delta_w = self.shared_lora.get_merged_weight()
        self.base_layer.weight.data += shared_delta_w

        self.is_merged = True
        del self.shared_lora

    def to_linear(self, name: str):
        self.merge_shared_and_unload(name)
        return None
