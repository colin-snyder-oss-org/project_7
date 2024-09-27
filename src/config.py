# src/config.py
from dataclasses import dataclass, field
from typing import Any, Dict, List
import yaml
import os

@dataclass
class Config:
    # General Configurations
    seed: int = 42
    device: str = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

    # Data Configurations
    data_path: str = 'data/processed/'
    batch_size: int = 64
    num_workers: int = 4

    # Model Configurations
    model_name: str = 'ComplexGNNModel'
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 64
    num_layers: int = 3

    # Training Configurations
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 5e-4

    # Additional configurations can be loaded from a YAML file
    extra_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        config_file = 'config.yaml'
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
                self.__dict__.update(config_dict)
