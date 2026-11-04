from .gc_trainer import GradCacheTrainer

TRAINER_DICT = {
    # "base": TevatronTrainer,
    "base": GradCacheTrainer,
    "custom_lora": GradCacheTrainer,
}


def build_trainer(trainer_name: str, *args, **kwargs):
    trainer_name = trainer_name or "base"
    assert trainer_name in TRAINER_DICT
    return TRAINER_DICT[trainer_name](*args, **kwargs)
