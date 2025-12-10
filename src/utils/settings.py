"""
Stone Detection Service - Configuration Manager

This module provides centralized configuration management for the entire project.
It loads settings from config/config.yaml and provides easy access to all parameters.

Usage:
    from src.utils.settings import config, Config
    
    # Access settings directly
    yolo_path = config.models.yolo.path
    batch_size = config.training.batch_size
    
    # Or get nested values with defaults
    lr = config.get("training.learning_rate", default=0.001)
    
    # Get full section as dict
    gpu_config = config.get_section("hardware.gpu")
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from src/utils/settings.py to project root
    current = Path(__file__).resolve()
    # Go up: settings.py -> utils -> src -> project_root
    return current.parent.parent.parent


class DotDict(dict):
    """
    Dictionary that allows dot notation access to nested keys.
    
    Example:
        d = DotDict({'a': {'b': 1}})
        print(d.a.b)  # 1
    """
    
    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
            if isinstance(value, dict):
                return DotDict(value)
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value
    
    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")


class Config:
    """
    Configuration manager for Stone Detection Service.
    
    Loads configuration from YAML file and provides easy access methods.
    Supports dot notation, nested key access, and default values.
    
    Attributes:
        paths: Project path configurations
        models: Model configurations (YOLO, MobileSAM, U-Net, etc.)
        training: Training parameters
        inference: Inference settings
        hardware: GPU and hardware optimization settings
        processing: Processing pipeline settings
        logging: Logging configuration
        data: Data loading and preprocessing settings
        service: API service configuration
    """
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls, config_path: Optional[str] = None) -> 'Config':
        """Singleton pattern - only one config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize configuration.
        
        Args:
            config_path: Optional path to config file. If None, uses default.
        """
        if self._initialized:
            return
            
        self._project_root = get_project_root()
        
        if config_path is None:
            config_path = self._project_root / "config" / "config.yaml"
        else:
            config_path = Path(config_path)
            
        self._config_path = config_path
        self._load_config()
        self._initialized = True
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self._config_path}"
            )
        
        with open(self._config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f) or {}
        
        # Resolve relative paths to absolute
        self._resolve_paths()
    
    def _resolve_paths(self) -> None:
        """Convert relative paths in config to absolute paths."""
        if 'paths' not in self._config:
            return
            
        for key, value in self._config['paths'].items():
            if isinstance(value, str):
                # Convert to absolute path relative to project root
                abs_path = self._project_root / value
                self._config['paths'][key] = str(abs_path)
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Dot-separated key path (e.g., "models.yolo.path")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            lr = config.get("training.learning_rate", default=0.001)
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a configuration section as a dictionary.
        
        Args:
            section: Section name (e.g., "models.yolo")
            
        Returns:
            Section as dictionary or empty dict if not found
        """
        value = self.get(section, {})
        return dict(value) if isinstance(value, dict) else {}
    
    def get_path(self, key: str) -> Path:
        """
        Get a path configuration value as Path object.
        
        Args:
            key: Dot-separated key path
            
        Returns:
            Path object
        """
        path_str = self.get(key)
        if path_str is None:
            raise KeyError(f"Path not found in config: {key}")
        return Path(path_str)
    
    def __getattr__(self, key: str) -> Any:
        """Allow dot notation access to top-level sections."""
        if key.startswith('_'):
            return super().__getattribute__(key)
        
        if key in self._config:
            value = self._config[key]
            if isinstance(value, dict):
                return DotDict(value)
            return value
        
        raise AttributeError(f"Config has no section '{key}'")
    
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return self._project_root
    
    @property
    def as_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return dict(self._config)
    
    def __repr__(self) -> str:
        return f"Config(path={self._config_path})"


# Global config instance - import this in your scripts
config = Config()


# ==============================================================================
# Convenience functions for common operations
# ==============================================================================

def get_model_path(model_name: str) -> Path:
    """
    Get the path to a model file.
    
    Args:
        model_name: Name of the model (yolo, mobilesam, unet, se_unet)
        
    Returns:
        Absolute path to model file
    """
    path = config.get(f"models.{model_name}.path")
    if path is None:
        raise KeyError(f"Model path not found: {model_name}")
    
    abs_path = config.project_root / path
    return abs_path


def get_output_dir(subdir: Optional[str] = None) -> Path:
    """
    Get output directory, optionally with subdirectory.
    
    Args:
        subdir: Optional subdirectory name
        
    Returns:
        Path to output directory
    """
    base = config.get_path("paths.results_dir")
    if subdir:
        return base / subdir
    return base


def get_data_dir(subdir: Optional[str] = None) -> Path:
    """
    Get data directory, optionally with subdirectory.
    
    Args:
        subdir: Optional subdirectory name (raw, processed, annotations, datasets)
        
    Returns:
        Path to data directory
    """
    if subdir:
        key = f"paths.{subdir}"
        path = config.get(key)
        if path:
            return Path(path)
    
    return config.get_path("paths.data_root")


def ensure_dirs() -> None:
    """Ensure all output directories exist."""
    dirs_to_create = [
        config.get("paths.results_dir"),
        config.get("paths.visualizations"),
        config.get("paths.json_output"),
        config.get("paths.reports"),
        config.get("paths.logs_dir"),
    ]
    
    for dir_path in dirs_to_create:
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Usage examples (for documentation)
# ==============================================================================

if __name__ == "__main__":
    # Example usage
    print(f"Project root: {config.project_root}")
    print(f"YOLO model path: {config.models.yolo.path}")
    print(f"Training batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.get('training.learning_rate', 0.001)}")
    print(f"GPU memory fraction: {config.hardware.gpu.memory_fraction}")
    
    # Get model path
    yolo_path = get_model_path("yolo")
    print(f"YOLO absolute path: {yolo_path}")
    
    # Ensure directories exist
    ensure_dirs()
    print("Output directories created/verified")
