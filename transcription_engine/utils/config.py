# File: transcription_engine/utils/config.py
"""
Configuration management for the Open Transcription Engine.
Handles loading and validation of configuration settings from YAML files,
with support for different environments and GPU configurations.
"""

import os
import yaml
import torch
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WhisperConfig:
    """Configuration for Whisper model settings."""
    model_size: str  # tiny, base, small, medium, large
    device: str  # cpu, cuda, mps
    language: str  # en, auto, etc.
    batch_size: int
    compute_type: str  # float32, float16, int8


@dataclass
class AudioConfig:
    """Configuration for audio recording and processing."""
    sample_rate: int
    channels: int
    chunk_size: int
    format: str  # wav, mp3, etc.
    device_index: Optional[int]


@dataclass
class RedactionConfig:
    """Configuration for redaction settings."""
    sensitive_phrases_file: Path
    redaction_char: str
    min_phrase_length: int
    fuzzy_threshold: float


@dataclass
class SecurityConfig:
    """Security and privacy related configuration."""
    encrypt_audio: bool
    encrypt_transcripts: bool
    encryption_key_file: Optional[Path]
    audit_logging: bool


@dataclass
class SystemConfig:
    """Main configuration class containing all settings."""
    whisper: WhisperConfig
    audio: AudioConfig
    redaction: RedactionConfig
    security: SecurityConfig
    output_dir: Path
    temp_dir: Path
    max_audio_length: int  # maximum length in seconds to process at once


class ConfigurationManager:
    """Manages loading and accessing configuration settings."""

    DEFAULT_CONFIG_PATH = Path("config/default.yml")

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config: Optional[SystemConfig] = None

    def load_config(self) -> SystemConfig:
        """Load and validate configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}, creating default config")
            self._create_default_config()

        with open(self.config_path) as f:
            config_dict = yaml.safe_load(f)

        self.config = self._parse_config(config_dict)
        self._validate_config()
        self._setup_device()
        return self.config

    def _create_default_config(self) -> None:
        """Create default configuration file if none exists."""
        default_config = {
            'whisper': {
                'model_size': 'base',
                'device': 'auto',
                'language': 'en',
                'batch_size': 16,
                'compute_type': 'float16'
            },
            'audio': {
                'sample_rate': 16000,
                'channels': 1,
                'chunk_size': 1024,
                'format': 'wav',
                'device_index': None
            },
            'redaction': {
                'sensitive_phrases_file': 'config/sensitive_phrases.txt',
                'redaction_char': '*',
                'min_phrase_length': 2,
                'fuzzy_threshold': 0.85
            },
            'security': {
                'encrypt_audio': True,
                'encrypt_transcripts': True,
                'encryption_key_file': None,
                'audit_logging': True
            },
            'output_dir': 'output',
            'temp_dir': 'temp',
            'max_audio_length': 3600  # 1 hour
        }

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(default_config, f)

    def _parse_config(self, config_dict: Dict) -> SystemConfig:
        """Parse configuration dictionary into dataclass instances."""
        whisper_config = WhisperConfig(**config_dict['whisper'])
        audio_config = AudioConfig(**config_dict['audio'])
        redaction_config = RedactionConfig(
            **{**config_dict['redaction'],
               'sensitive_phrases_file': Path(config_dict['redaction']['sensitive_phrases_file'])}
        )
        security_config = SecurityConfig(
            **{**config_dict['security'],
               'encryption_key_file': Path(config_dict['security']['encryption_key_file'])
               if config_dict['security']['encryption_key_file'] else None}
        )

        return SystemConfig(
            whisper=whisper_config,
            audio=audio_config,
            redaction=redaction_config,
            security=security_config,
            output_dir=Path(config_dict['output_dir']),
            temp_dir=Path(config_dict['temp_dir']),
            max_audio_length=config_dict['max_audio_length']
        )

    def _validate_config(self) -> None:
        """Validate configuration settings."""
        if self.config is None:
            raise ValueError("Configuration not loaded")

        # Validate directories exist or create them
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)

        # Validate Whisper settings
        valid_models = {'tiny', 'base', 'small', 'medium', 'large'}
        if self.config.whisper.model_size not in valid_models:
            raise ValueError(f"Invalid model size. Must be one of {valid_models}")

        # Validate audio settings
        if self.config.audio.sample_rate not in {8000, 16000, 22050, 44100, 48000}:
            raise ValueError("Invalid sample rate")

        if self.config.audio.channels < 1:
            raise ValueError("Number of channels must be positive")

        # Validate security settings
        if self.config.security.encrypt_audio or self.config.security.encrypt_transcripts:
            if not self.config.security.encryption_key_file:
                raise ValueError("Encryption key file must be specified when encryption is enabled")

    def _setup_device(self) -> None:
        """Configure the computation device (CPU, CUDA, or MPS)."""
        if self.config is None:
            raise ValueError("Configuration not loaded")

        if self.config.whisper.device == 'auto':
            if torch.backends.mps.is_available():
                self.config.whisper.device = 'mps'
                logger.info("Using MPS (Metal Performance Shaders) for computation")
            elif torch.cuda.is_available():
                self.config.whisper.device = 'cuda'
                logger.info("Using CUDA for computation")
            else:
                self.config.whisper.device = 'cpu'
                logger.info("Using CPU for computation")
        else:
            logger.info(f"Using specified device: {self.config.whisper.device}")


# Global configuration instance
config_manager = ConfigurationManager()