# File: transcription_engine/utils/config.py
"""Configuration management for the Open Transcription Engine.

Handles loading and validation of configuration settings from YAML files,
with support for different environments and GPU configurations.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Type variable for the ConfigurationManager class
T = TypeVar("T", bound="ConfigurationManager")

VALID_MODELS = {
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
}


@dataclass
class WhisperConfig:
    """Configuration for Whisper model settings."""

    model_size: str  # tiny, base, small, medium, large
    device: str  # cpu, cuda, mps
    language: str  # en, auto, etc.
    batch_size: int
    compute_type: str  # float32, float16, int8
    backend: str = "transformers"  # transformers or fast
    attn_implementation: str = "flash_attention_2"  # sdpa or flash_attention_2
    chunk_length_s: int = 30
    progress_bar: bool = True
    use_cache: bool = True
    cache_dir: str = "~/.cache/whisper"


@dataclass
class AudioConfig:
    """Configuration for audio recording and processing."""

    sample_rate: int
    channels: int
    chunk_size: int
    format: str  # wav, mp3, etc.
    device_index: int | None


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
    encryption_key_file: Path | None
    audit_logging: bool


@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization."""

    enabled: bool = True
    use_pyannote: bool = True
    auth_token: str | None = None
    device: str = "auto"  # cpu, cuda, mps, auto


@dataclass
class SystemConfig:
    """Main configuration class containing all settings."""

    whisper: WhisperConfig
    audio: AudioConfig
    redaction: RedactionConfig
    security: SecurityConfig
    diarization: DiarizationConfig  # Add diarization config
    output_dir: Path
    temp_dir: Path
    max_audio_length: int  # maximum length in seconds to process at once


class ConfigurationManager:
    """Manages loading and accessing configuration settings."""

    DEFAULT_CONFIG_PATH = Path("config/default.yml")

    def __init__(self: T, config_path: Path | None = None) -> None:
        """Initialize configuration manager."""
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config: SystemConfig | None = None

    def load_config(self: T) -> SystemConfig:
        """Load and validate configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(
                "Config file not found at %s, creating default config",
                self.config_path,
            )
            self._create_default_config()

        with open(self.config_path) as f:
            config_dict = yaml.safe_load(f)

        self.config = self._parse_config(config_dict)
        self._validate_config()
        self._setup_device()
        return self.config

    def _create_default_config(self: T) -> None:
        """Create default configuration file if none exists."""
        default_config: dict[str, Any] = {
            "whisper": {
                "model_size": "base",
                "device": "auto",
                "language": "en",
                "batch_size": 16,
                "compute_type": "float16",
            },
            "audio": {
                "sample_rate": 16000,
                "channels": 1,
                "chunk_size": 1024,
                "format": "wav",
                "device_index": None,
            },
            "redaction": {
                "sensitive_phrases_file": "config/sensitive_phrases.txt",
                "redaction_char": "*",
                "min_phrase_length": 2,
                "fuzzy_threshold": 0.85,
            },
            "security": {
                "encrypt_audio": True,
                "encrypt_transcripts": True,
                "encryption_key_file": None,
                "audit_logging": True,
            },
            "output_dir": "output",
            "temp_dir": "temp",
            "max_audio_length": 3600,  # 1 hour
        }

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.safe_dump(default_config, f)

    def _parse_config(self: T, config_dict: dict[str, Any]) -> SystemConfig:
        """Parse configuration dictionary into dataclass instances."""
        whisper_config = WhisperConfig(**config_dict["whisper"])
        audio_config = AudioConfig(**config_dict["audio"])
        redaction_config = RedactionConfig(
            **{
                **config_dict["redaction"],
                "sensitive_phrases_file": Path(
                    config_dict["redaction"]["sensitive_phrases_file"],
                ),
            },
        )
        security_config = SecurityConfig(
            **{
                **config_dict["security"],
                "encryption_key_file": Path(
                    config_dict["security"]["encryption_key_file"],
                )
                if config_dict["security"]["encryption_key_file"]
                else None,
            },
        )

        # Create diarization config with environment variable override for auth token
        diarization_config = DiarizationConfig(
            **{
                **config_dict["diarization"],
                "auth_token": os.getenv("HUGGINGFACE_TOKEN")
                or config_dict["diarization"].get("auth_token"),
            }
        )

        return SystemConfig(
            whisper=whisper_config,
            audio=audio_config,
            redaction=redaction_config,
            security=security_config,
            diarization=diarization_config,
            output_dir=Path(config_dict["output_dir"]),
            temp_dir=Path(config_dict["temp_dir"]),
            max_audio_length=config_dict["max_audio_length"],
        )

    def _validate_config(self: T) -> None:
        """Validate configuration settings."""
        if self.config is None:
            error_message = "Configuration not loaded"
            raise ValueError(error_message)

        # Validate directories exist or create them
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)

        # Validate Whisper settings
        if self.config.whisper.model_size not in VALID_MODELS:
            error_message = f"Invalid model size. Must be one of {VALID_MODELS}"
            raise ValueError(error_message)

        # Validate audio settings
        if self.config.audio.sample_rate not in {8000, 16000, 22050, 44100, 48000}:
            error_message = "Invalid sample rate"
            raise ValueError(error_message)

        if self.config.audio.channels < 1:
            error_message = "Number of channels must be positive"
            raise ValueError(error_message)

        # Validate security settings
        if (
            self.config.security.encrypt_audio
            or self.config.security.encrypt_transcripts
        ):
            if not self.config.security.encryption_key_file:
                error_message = "Encryption key file must be specified when enabled."
                raise ValueError(error_message)

    def _setup_device(self: T) -> None:
        """Configure the computation device (CPU, CUDA, or MPS)."""
        if self.config is None:
            error_message = "Configuration not loaded"
            raise ValueError(error_message)

        if self.config.whisper.device == "auto":
            if torch.backends.mps.is_available():
                self.config.whisper.device = "mps"
                logger.info("Using MPS (Metal Performance Shaders) for computation")
            elif torch.cuda.is_available():
                self.config.whisper.device = "cuda"
                logger.info("Using CUDA for computation")
            else:
                self.config.whisper.device = "cpu"
                logger.info("Using CPU for computation")
        else:
            logger.info(f"Using specified device: {self.config.whisper.device}")


# Global configuration instance
config_manager = ConfigurationManager()
