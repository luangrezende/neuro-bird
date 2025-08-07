import yaml
import os
from typing import Dict, Any, Optional

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
            
        with open(config_path, 'r', encoding='utf-8') as file:
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
        return self._config_data.get(section, {})
    
    def set(self, key_path: str, value: Any) -> None:
        keys = key_path.split('.')
        data = self._config_data
        
        for key in keys[:-1]:
            if key not in data:
                data[key] = {}
            data = data[key]
        
        data[keys[-1]] = value
    
    def save_config(self, config_path: str = "config.yaml") -> None:
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(self._config_data, file, default_flow_style=False, indent=2)
    
    @property
    def all(self) -> Dict[str, Any]:
        return self._config_data
    
    def validate_config(self) -> bool:
        required_sections = ['app', 'vision', 'agent', 'training', 'environment', 'utils']
        
        for section in required_sections:
            if section not in self._config_data:
                raise ValueError(f"Missing required configuration section: {section}")
        
        vision_config = self._config_data.get('vision', {})
        ocr_config = vision_config.get('ocr', {}) if isinstance(vision_config, dict) else {}
        gpu_enabled = ocr_config.get('gpu', False) if isinstance(ocr_config, dict) else False
        
        if not gpu_enabled:
            raise ValueError("GPU must be enabled for OCR processing")
        
        return True

config = Config()
