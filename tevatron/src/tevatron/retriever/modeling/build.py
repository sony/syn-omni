from .dense import CustomLoRA_MMDenseModel, MultiModalDenseModel

MODEL_DICT = {
    "base": MultiModalDenseModel,
    "custom_lora": CustomLoRA_MMDenseModel,
}


def build_model(model_name: str, *args, **kwargs):
    model_name = model_name or "base"
    assert model_name in MODEL_DICT, f"unknown model type: {model_name}"
    return MODEL_DICT[model_name].build(*args, **kwargs)
