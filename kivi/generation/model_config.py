import os
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any


def _expand_env_var(value: str) -> str:
    """Expand ${ENV_VAR:-default} syntax in a string."""
    pattern = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")

    def _replace(match):
        var_name = match.group(1)
        default = match.group(2)
        return os.environ.get(var_name, default if default is not None else "")

    return pattern.sub(_replace, value)


def _resolve_dict(d: dict) -> dict:
    """Recursively expand env vars in all string values of a dict."""
    result = {}
    for k, v in d.items():
        if isinstance(v, str):
            result[k] = _expand_env_var(v)
        elif isinstance(v, dict):
            result[k] = _resolve_dict(v)
        elif isinstance(v, list):
            result[k] = [
                _expand_env_var(item) if isinstance(item, str) else item
                for item in v
            ]
        else:
            result[k] = v
    return result


def discover_models(config_dir: str) -> List["ModelConfig"]:
    """Auto-discover all .yaml files in config directory, return ModelConfig objects."""
    configs = []
    config_path = Path(config_dir)
    if not config_path.is_dir():
        return configs

    for yaml_file in sorted(config_path.glob("*.yaml")):
        if yaml_file.name.startswith(("_", ".")):
            continue
        try:
            config = ModelConfig.from_yaml(str(yaml_file))
            configs.append(config)
        except Exception as e:
            print(f"[WARN] Failed to load {yaml_file}: {e}")

    return configs


class ModelConfig:
    """Parsed representation of a model's YAML configuration."""

    def __init__(self, data: dict, source_path: str = ""):
        self._raw = data
        self.source_path = source_path

        self.name: str = data["name"]
        self.description: str = data.get("description", "")
        self.mode: str = data["mode"]  # "segment", "interactive", "single_prompt", "api"
        self.fps: int = data.get("fps", 24)
        self.max_num_frames: int = data.get("max_num_frames", 257)
        self.frame_snapping: str = data.get("frame_snapping", "none")
        self.code_dir: str = data.get("code_dir", "")

        self.model_path: dict = _resolve_dict(data.get("model_path", {}))
        self.environment: dict = data.get("environment", {})

        # Mode-specific data
        self.segment: dict = data.get("segment", {})
        self.multi_gpu: dict = data.get("multi_gpu", {})
        self.generation: dict = data.get("generation", {})
        self.output_patterns: List[str] = data.get("output_patterns", [])
        self.api: dict = data.get("api", {})

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data or "name" not in data:
            raise ValueError(f"Invalid model config: missing 'name' in {path}")
        return cls(data, source_path=path)

    def get_model_path(self, key: str = "default") -> str:
        """Resolve a model path by key. Falls back to the first available value."""
        if key in self.model_path:
            return self.model_path[key]
        # Try all keys in order until we find one
        for k, v in self.model_path.items():
            return v
        return ""

    def get_env(self, code_dir_abs: str = "") -> Dict[str, str]:
        """Build environment variables with {code_dir} and {inherit} substituted."""
        env = {}
        for k, v in self.environment.items():
            v_expanded = _expand_env_var(v)
            v_expanded = v_expanded.replace("{code_dir}", code_dir_abs or self.code_dir)
            # {inherit} means prepend to existing env value
            if "{inherit}" in v_expanded:
                existing = os.environ.get(k, "")
                v_expanded = v_expanded.replace("{inherit}:", "").replace("{inherit}", "")
                if existing:
                    v_expanded = f"{v_expanded}:{existing}" if v_expanded else existing
            env[k] = v_expanded
        return env