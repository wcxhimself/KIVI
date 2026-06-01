import os
from .generic_api_generator import GenericAPIGenerator
from .model_config import discover_models
from .config_based_generator import ConfigBasedGenerator

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")
_configs_loaded = False
_MODEL_REGISTRY = {}


def _ensure_configs_loaded():
    global _configs_loaded
    if _configs_loaded:
        return
    _configs_loaded = True

    configs = discover_models(CONFIG_DIR)
    for cfg in configs:
        name = cfg.name
        if name not in _MODEL_REGISTRY:
            _MODEL_REGISTRY[name] = cfg


def list_available_models():
    """Return dict of {model_name: description} from YAML configs."""
    _ensure_configs_loaded()
    models = {}
    for name, cfg in _MODEL_REGISTRY.items():
        models[name] = cfg.description or name
    return models


def create_video_generator(model_name: str, **kwargs):
    """
    Factory function to instantiate a video generator from YAML configuration.

    Mode dispatch:
      - "segment" / "interactive" / "single_prompt" → ConfigBasedGenerator (shell subprocess)
      - "api"               → GenericAPIGenerator (REST polling)
    """
    _ensure_configs_loaded()

    model_name_lower = model_name.lower()
    cfg = _MODEL_REGISTRY.get(model_name_lower)
    if cfg is None:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Available: {list(list_available_models().keys())}\n"
            f"To add a new model, create a YAML file in configs/ "
            f"(see configs/wan22.yaml for example)."
        )

    if cfg.mode == "api":
        return GenericAPIGenerator(config=cfg, **kwargs)

    return ConfigBasedGenerator(config=cfg, **kwargs)