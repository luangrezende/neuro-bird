"""
Configuration management with singleton pattern.
"""
import os
from typing import Any, Dict

import yaml

class Config:
    _instance = None
    _config_data: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config_data:
            self.load_config()
    
    def load_config(self, config_path: str = "config.yaml") -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if hasattr(self, '_config_data') and self._config_data:
            yaml_settings = self.get('utils.yaml_settings', {})
        else:
            yaml_settings = {}
        
        encoding = yaml_settings.get('encoding', 'utf-8')
        with open(config_path, 'r', encoding=encoding) as file:
            self._config_data = yaml.safe_load(file) or {}
    
    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self._config_data
        
        try:
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        return self._config_data[section]

config = Config()
