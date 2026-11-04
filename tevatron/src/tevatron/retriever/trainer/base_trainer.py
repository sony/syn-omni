import logging
import os

from typing import Optional

import torch
import torch.distributed as dist

from transformers.trainer import TRAINING_ARGS_NAME, Trainer

logger = logging.getLogger(__name__)


class TevatronTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def _save(self, output_dir: Optional[str] = None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        output_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        if self.args.training_component == "custom_lora":
            # skip saving checkpoint here; self.model.save() includes saving the checkpoint
            state_dict = {}
        else:
            state_dict = self.model.state_dict()

        prefix = 'encoder.'
        assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        self.model.encoder.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save(output_dir)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # we do not use mm order and modality masks info in default trainer; its for grad_cache
        (query, _, _), (passage, _, _) = inputs
        return model(query=query, passage=passage).loss

    def training_step(self, *args):
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    def _save_loss(self, loss_dict):
        self._stored_losses = loss_dict

    def log(self, logs, *args, **kwargs):
        if hasattr(self, "_stored_losses"):
            logs.update(self._stored_losses)
            delattr(self, "_stored_losses")
        super().log(logs, *args, **kwargs)
