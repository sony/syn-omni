import math


class AlphaScheduler:

    def __init__(
        self,
        warmup_ratio: float = 0.1,  # 0~10%: alpha=0
        transition_ratio: float = 0.5,  # 10~50%: alpha 0 -> max_alpha
        schedule_type: str = "cosine"  # "cosine" or "linear"
    ):
        self.warmup_ratio = warmup_ratio
        self.transition_ratio = transition_ratio
        self.transition_end = warmup_ratio + transition_ratio
        self.schedule_type = schedule_type
        self.max_alpha = 1

    def step_scheduler(self, t: float) -> float:
        """
        Calculates the progressive routing alpha based on the training progress ratio 't'.
        t: Training progress ratio (0.0 to 1.0), i.e., current_step / total_steps
        """
        # Stabilization (Hard mask only)
        if t <= self.warmup_ratio:
            return 0.0

        # Fine-tuning (Hold max alpha)
        if t >= self.transition_end:
            return self.max_alpha

        # Phase 2: Transition
        # Normalize 't' to a [0.0, 1.0] scale within the transition window
        progress = (t - self.warmup_ratio) / self.transition_ratio

        if self.schedule_type == "linear":
            return self.max_alpha * progress

        elif self.schedule_type == "cosine":
            # Smooth S-curve transition using cosine (0.5 * (1 - cos(pi * x)))
            return self.max_alpha * 0.5 * (1.0 - math.cos(math.pi * progress))

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


def build_routing_scheduler(lora_config):
    if lora_config["router_type"] != "progressive":
        return None
    warmup_ratio = lora_config.get("router_warmup_ratio", 0.1)
    transition_ratio = lora_config.get("router_transition_ratio", 0.5)
    schedule_type = lora_config.get("router_schedule_type", "cosine")

    return AlphaScheduler(
        warmup_ratio=warmup_ratio, transition_ratio=transition_ratio, schedule_type=schedule_type
    )
