import contextvars

from contextlib import contextmanager

# Declare context variables
MODALITY_MASK_VAR = contextvars.ContextVar("modality_mask", default=None)
# AUX_LOSS_VAR = contextvars.ContextVar("aux_loss_var", default=None)
# ROUTING_ALPHA_VAR = contextvars.ContextVar("routing_alpha", default=0.0)


@contextmanager
def provide_modality_mask(mask):
    token = MODALITY_MASK_VAR.set(mask)
    try:
        yield
    finally:
        MODALITY_MASK_VAR.reset(token)


"""
@contextmanager
def provide_aux_loss_context():
    token = AUX_LOSS_VAR.set(defaultdict(list))
    try:
        yield
    finally:
        AUX_LOSS_VAR.reset(token)


def add_aux_loss(name: str, loss: Tensor):
    if not torch.is_grad_enabled():
        return

    loss_dict = AUX_LOSS_VAR.get()
    if loss_dict is not None:
        loss_dict[name].append(loss)


def get_averaged_aux_losses() -> dict:
    loss_dict = AUX_LOSS_VAR.get()
    if loss_dict is None or len(loss_dict) == 0:
        return {}

    averaged_losses = {}
    for name, loss_list in loss_dict.items():
        averaged_losses[name] = sum(loss_list) / len(loss_list)

    return averaged_losses


@contextmanager
def provide_routing_alpha(alpha: float):
    token = ROUTING_ALPHA_VAR.set(alpha)
    try:
        yield
    finally:
        ROUTING_ALPHA_VAR.reset(token)
"""
