from collections import defaultdict
from contextlib import nullcontext
from typing import Any, List

import torch
import torch.nn as nn

from grad_cache import GradCache
from grad_cache.context_managers import RandContext
from torch import Tensor

from .modeling.lora.global_vars import get_averaged_aux_losses, provide_aux_loss_context


class AuxGradCache(GradCache):
    """
    Custom GradCache that supports multiple internal Auxiliary Losses.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dictionary to store running sum of each aux loss for logging
        self.running_aux_losses = defaultdict(float)

    def forward_backward(
        self,
        model: nn.Module,
        model_inputs: List[Any],
        cached_gradients: List[Tensor],
        random_states: List[RandContext],
        no_sync_except_last: bool = False
    ):
        if no_sync_except_last:
            sync_contexts = [model.no_sync for _ in range(len(model_inputs) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_inputs))]

        # Chunk scaling factor to match global batch mean
        num_chunks = len(model_inputs)
        aux_loss_scale = 1.0 / num_chunks

        for x, state, gradient, sync_context in zip(
            model_inputs, random_states, cached_gradients, sync_contexts
        ):
            with sync_context():
                with state:
                    # Open context for current chunk
                    with provide_aux_loss_context():

                        y = self.model_call(model, x)

                        # Retrieve collected aux losses
                        chunk_aux_losses = get_averaged_aux_losses()

                        total_chunk_aux_loss = 0.0
                        if chunk_aux_losses:
                            for loss_name, loss_val in chunk_aux_losses.items():
                                scaled_loss = loss_val * aux_loss_scale
                                total_chunk_aux_loss += scaled_loss

                                # Save detached float value for trainer logging
                                self.running_aux_losses[loss_name] += scaled_loss.detach().item()

                reps = self.get_reps(y)

                # Main contrastive surrogate
                surrogate = torch.dot(reps.flatten(), gradient.flatten())

                # Add aux loss only if it exists (keeps autograd graph clean)
                if isinstance(total_chunk_aux_loss, torch.Tensor) or total_chunk_aux_loss != 0.0:
                    final_surrogate = surrogate + total_chunk_aux_loss
                else:
                    final_surrogate = surrogate

                final_surrogate.backward()
