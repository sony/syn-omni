import logging

import torch

from torch import Tensor
from torch import distributed as dist
from torch.nn import functional as F

from tevatron.retriever.grad_cache import AuxGradCache
from tevatron.retriever.modeling.lora import provide_routing_alpha

from .base_trainer import TevatronTrainer
from .gc_splitter import split_dense_inputs

logger = logging.getLogger(__name__)


class SimpleContrastiveLoss:

    def __init__(self, temperature: float = 0.02):
        self.temperature = temperature

    def __call__(
        self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean', **kwargs
    ):
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long
            )
        logits = torch.matmul(x, y.transpose(0, 1))

        return F.cross_entropy(logits / self.temperature, target, reduction=reduction)


class DistributedContrastiveLoss(SimpleContrastiveLoss):

    def __init__(self, temperature: float = 0.02, n_target: int = 0, scale_loss: bool = True):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__(temperature=temperature)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        # pop optional arguments for logging
        kwargs.pop("query_masks", None)  # query_masks
        kwargs.pop("passage_masks", None)  # passage_masks

        # loss function as usual
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)

        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.world_size

        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)


def get_dense_rep(x):
    if x.q_reps is None:
        return x.p_reps
    else:
        return x.q_reps


class GradCacheTrainer(TevatronTrainer):

    def __init__(self, *args, **kwargs):
        super(GradCacheTrainer, self).__init__(*args, **kwargs)
        logger.info('Initializing Gradient Cache Trainer')

        loss_fn_cls = DistributedContrastiveLoss if self.is_ddp else SimpleContrastiveLoss
        loss_fn = loss_fn_cls(temperature=self.model.temperature)

        self.gc = AuxGradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None
        )
        self.loss_fn = loss_fn
        self.alpha_scheduler = self.model.alpha_scheduler

    def training_step(self, model, inputs, num_items_in_batch: int = None) -> torch.Tensor:
        model.train()
        (queries, query_masks, queries_mm), (passages, passage_masks,
                                             passages_mm) = self._prepare_inputs(inputs)

        # masks also chunked
        queries["modality_mask"] = query_masks
        passages["modality_mask"] = passage_masks

        queries, passages = {'query': queries}, {'passage': passages}
        query_inputs, passage_inputs = (queries, queries_mm), (passages, passages_mm)

        _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]

        if self.alpha_scheduler is not None:
            max_steps = max(1, self.state.max_steps)
            t = self.state.global_step / max_steps
            current_alpha = self.alpha_scheduler.step_scheduler(t)
        else:
            current_alpha = None

        with provide_routing_alpha(current_alpha):
            # lora/router.py
            loss = self.gc(
                query_inputs,
                passage_inputs,
                no_sync_except_last=_distributed,
                query_masks=query_masks,
                passage_masks=passage_masks
            )

        self._last_alpha = current_alpha  # for logging

        return loss / self._dist_loss_scale_factor

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        # Flush and log all accumulated auxiliary losses
        if hasattr(self, "gc") and hasattr(self.gc, "running_aux_losses"):
            for loss_name, loss_val in self.gc.running_aux_losses.items():
                logs[f"loss_{loss_name}"] = loss_val

            # Reset after logging
            self.gc.running_aux_losses.clear()

        if hasattr(self, "_last_alpha") and self._last_alpha is not None:
            logs["routing_alpha"] = self._last_alpha

        super().log(logs, *args, **kwargs)
