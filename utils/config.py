"""Configuration management utilities."""

from typing import Any, Dict, Optional
import yaml
import os


class Config:
    """Hierarchical configuration container with dot notation access."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)

        value = self._config.get(name)
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set value using dot notation."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config

    def update(self, other: Dict[str, Any]) -> None:
        """Update configuration."""
        self._config.update(other)

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    @classmethod
    def from_args(cls, args: Any, base_config: Optional['Config'] = None) -> 'Config':
        """Create configuration from argparse namespace."""
        config = base_config or cls()
        for key, value in vars(args).items():
            if value is not None:
                config.set(key, value)
        return config

    def __repr__(self) -> str:
        return f"Config({self._config})"


def merge_configs(base_path: str, *override_paths: str) -> Config:
    """Merge multiple configuration files."""
    config = Config.from_yaml(base_path)

    for path in override_paths:
        if os.path.exists(path):
            override = Config.from_yaml(path)
            config._config = deep_merge(config._config, override._config)

    return config


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
