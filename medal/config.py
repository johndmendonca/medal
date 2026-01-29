"""
Configuration management for MEDAL framework.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class PathConfig:
    """Configuration for directory paths."""
    batches_to_process: str = "batches_to_process"
    completed_batches: str = "completed_batches"
    submitted_batches: str = "submitted_batches"
    dialogues: str = "dialogues"
    data: str = "data"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                path = getattr(self, attr_name)
                if isinstance(path, str):
                    Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    temperature: float = 0.9
    top_p: float = 0.95
    frequency_penalty: float = 1.0
    presence_penalty: float = 0.6
    max_tokens: int = 512
    response_format: Optional[Dict[str, Any]] = None


@dataclass
class APIConfig:
    """Configuration for API providers."""
    provider: str = "openai"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_requests_per_minute: float = 1000.0
    max_tokens_per_minute: float = 100000.0
    max_attempts: int = 5
    token_encoding_name: str = "cl100k_base"
    
    def __post_init__(self):
        """Set default API key from environment if not provided."""
        if self.api_key is None:
            env_key_map = {
                "openai": "OPENAI_API_KEY",
                "google": "GEMENI_KEY",
                "openrouter": "OPENROUTER_KEY",
            }
            env_key = env_key_map.get(self.provider, "OPENAI_API_KEY")
            self.api_key = os.getenv(env_key)


@dataclass
class VLLMConfig:
    """Configuration for VLLM inference."""
    tensor_parallel_size: int = 4
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    trust_remote_code: bool = True


@dataclass
class Config:
    """Main configuration class."""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        
        return cls(
            paths=PathConfig(**config_dict.get("paths", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            api=APIConfig(**config_dict.get("api", {})),
            vllm=VLLMConfig(**config_dict.get("vllm", {})),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "paths": {
                "batches_to_process": self.paths.batches_to_process,
                "completed_batches": self.paths.completed_batches,
                "submitted_batches": self.paths.submitted_batches,
                "dialogues": self.paths.dialogues,
                "data": self.paths.data,
            },
            "model": {
                "temperature": self.model.temperature,
                "top_p": self.model.top_p,
                "frequency_penalty": self.model.frequency_penalty,
                "presence_penalty": self.model.presence_penalty,
                "max_tokens": self.model.max_tokens,
            },
            "api": {
                "provider": self.api.provider,
                "max_requests_per_minute": self.api.max_requests_per_minute,
                "max_tokens_per_minute": self.api.max_tokens_per_minute,
                "max_attempts": self.api.max_attempts,
                "token_encoding_name": self.api.token_encoding_name,
            },
            "vllm": {
                "tensor_parallel_size": self.vllm.tensor_parallel_size,
                "pipeline_parallel_size": self.vllm.pipeline_parallel_size,
                "gpu_memory_utilization": self.vllm.gpu_memory_utilization,
                "max_model_len": self.vllm.max_model_len,
            },
        }
