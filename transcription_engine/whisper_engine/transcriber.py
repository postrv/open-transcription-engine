# File: transcription_engine/whisper_engine/transcriber.py
"""Whisper integration module for the Open Transcription Engine.

Handles model loading, chunking, and transcription with GPU support. Supports both
standard Whisper and insanely-fast-whisper backends for optimal performance.
"""

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import psutil
import torch
from transformers import Pipeline, pipeline
from transformers.utils import is_flash_attn_2_available

from ..audio_input.recorder import AudioSegment
from ..utils.config import WhisperConfig, config_manager

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for the WhisperManager class
T = TypeVar("T", bound="WhisperManager")


@dataclass
class TranscriptionSegment:
    """Container for transcribed text and its metadata."""

    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    speaker_id: str | None = None
    confidence: float = 0.0
    diarization_data: dict[str, Any] = field(default_factory=dict)


class WhisperManager:
    """Manages Whisper model lifecycle and transcription."""

    # Mapping of model sizes to their memory requirements (in GB)
    MODEL_MEMORY_REQUIREMENTS = {
        "tiny": 1,
        "base": 1,
        "small": 2,
        "medium": 5,
        "large": 10,
        "large-v1": 10,
        "large-v2": 10,
        "large-v3": 10,
    }

    def __init__(self: T, config: WhisperConfig | None = None) -> None:
        """Initialize the Whisper manager with configuration."""
        self.config = config or config_manager.load_config().whisper
        self.model: Pipeline | None = None
        self.device = self._setup_device()
        self._validate_memory()

    def _setup_device(self: T) -> torch.device:
        """Configure the computation device based on availability and config."""
        if self.config.device == "auto":
            # First try MPS on Apple Silicon
            if torch.backends.mps.is_available():
                try:
                    # Test MPS with a small tensor operation
                    test_tensor = torch.zeros(1).to("mps")
                    del test_tensor
                    logger.info("Using MPS backend")
                    return torch.device("mps")
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    logger.warning(
                        "MPS available but test failed, falling back to CPU: %s",
                        e,
                    )

            # Try CUDA if available
            if torch.cuda.is_available():
                logger.info("Using CUDA backend")
                return torch.device("cuda")

            # CPU fallback
            logger.info("Using CPU backend")
            return torch.device("cpu")

        # If specific device requested, try it with fallback
        try:
            device = torch.device(self.config.device)
            # Test the device
            test_tensor = torch.zeros(1).to(device)
            del test_tensor
            return device
        except (RuntimeError, torch.cuda.OutOfMemoryError, ValueError) as e:
            logger.warning(
                "Requested device %s failed, falling back to CPU: %s",
                self.config.device,
                e,
            )
            return torch.device("cpu")

    def _validate_memory(self: T) -> None:
        """Validate if system has enough memory for the chosen model."""
        required_memory = self.MODEL_MEMORY_REQUIREMENTS[self.config.model_size]

        if self.device.type == "mps":
            # For Apple Silicon / MPS, assume adequate memory but warn about batch size
            if self.config.batch_size > 8:
                warnings.warn(
                    "Batch size > 8 may cause OOM on MPS. Reduce batch_size.",
                    UserWarning,
                    stacklevel=2,
                )
            return
        elif self.device.type == "cuda":
            try:
                available_memory = (
                    torch.cuda.get_device_properties(0).total_memory / 1e9
                )
            except (AssertionError, RuntimeError):
                # If CUDA is unavailable or fails, fall back to system memory check
                available_memory = psutil.virtual_memory().total / 1e9
        else:
            # CPU fallback
            available_memory = psutil.virtual_memory().total / 1e9

        if available_memory < required_memory:
            warnings.warn(
                f"Available memory ({available_memory:.1f}GB) may be insufficient "
                f"for {self.config.model_size} model ({required_memory}GB required)",
                UserWarning,
                stacklevel=2,
            )

    def load_model(self: T) -> bool:
        """Load the Whisper model into memory."""
        try:
            logger.info("Loading Whisper model: %s", self.config.model_size)

            # Handle legacy model names for compatibility
            model_name = self.config.model_size
            if model_name in ["large-v1", "large-v2", "large-v3"]:
                model_id = f"openai/whisper-{model_name}"
            else:
                model_id = f"openai/whisper-{model_name}"

            cache_dir = Path(self.config.cache_dir).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Setup torch dtype based on compute type and device
            if self.config.compute_type == "float16" and self.device.type != "cpu":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
                if self.config.compute_type == "float16":
                    logger.info(
                        "float16 not supported on %s, using float32",
                        self.device.type,
                    )

            # Set up model kwargs based on device and availability
            model_kwargs: dict[str, Any] = {}
            if (
                is_flash_attn_2_available()
                and self.device.type != "cpu"
                and self.config.attn_implementation == "flash_attention_2"
            ):
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            else:
                model_kwargs["attn_implementation"] = "sdpa"
                if self.config.attn_implementation == "flash_attention_2":
                    logger.info("Flash Attention 2 not available, using SDPA")

            # Initialize pipeline with optimized settings
            self.model = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=torch_dtype,
                device=self.device,
                model_kwargs=model_kwargs,
            )

            logger.info("Model loaded successfully")
            return True

        except (RuntimeError, ValueError) as e:
            logger.error("Error loading Whisper model: %s", e)
            return False

    def unload_model(self: T) -> None:
        """Unload the model to free up memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info("Model unloaded")

    def _prepare_audio(self: T, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Prepare audio data for Whisper model.

        Args:
            audio_data: Input audio data array
            sample_rate: Sample rate of the input audio

        Returns:
            Preprocessed audio data array
        """
        # Convert to float32 and normalize
        audio_data = audio_data.astype(np.float32)
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            from scipy import signal

            audio_data = signal.resample(
                audio_data,
                int(len(audio_data) * 16000 / sample_rate),
            )

        return audio_data

    def _chunk_audio(self: T, audio_data: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Split long audio into chunks for processing.

        Args:
            audio_data: Input audio data

        Returns:
            List of (chunk, timestamp) tuples
        """
        chunk_length = int(self.config.chunk_length_s * 16000)  # 16kHz sampling
        chunks = []

        for i in range(0, len(audio_data), chunk_length):
            chunk = audio_data[i : i + chunk_length]
            timestamp = i / 16000.0
            chunks.append((chunk, timestamp))

        return chunks

    def transcribe(
        self: T,
        audio_data: np.ndarray,
        sample_rate: int,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[TranscriptionSegment]:
        """Transcribe audio data into text with timestamps."""
        if self.model is None:
            msg = "Model not loaded. Call load_model() first."
            raise RuntimeError(msg)

        if len(audio_data) == 0:
            msg = "Empty audio data provided"
            raise ValueError(msg)

        try:
            # Prepare audio
            audio_data = self._prepare_audio(audio_data, sample_rate)

            # Run transcription with optimized settings
            result = self.model(
                audio_data,
                chunk_length_s=self.config.chunk_length_s,
                batch_size=self.config.batch_size,
                return_timestamps=True,  # Changed from "word" to True
                generate_kwargs={
                    "language": self.config.language
                    if self.config.language != "auto"
                    else None,
                    "task": "transcribe",
                },
            )

            # Process segments
            segments = []
            if isinstance(result, dict) and "chunks" in result:
                total_chunks = len(result["chunks"])
                for i, chunk in enumerate(result["chunks"]):
                    # Get timestamp from either format
                    timestamps = chunk.get("timestamp", [])
                    if isinstance(timestamps, list | tuple) and len(timestamps) == 2:
                        start, end = timestamps
                        if end is None:
                            logger.warning("Missing end timestamp in chunk: %s", chunk)
                            continue
                    else:
                        logger.warning("Invalid timestamp format in chunk: %s", chunk)
                        continue

                    segments.append(
                        TranscriptionSegment(
                            text=chunk["text"].strip(),
                            start=float(start),
                            end=float(end),
                            confidence=chunk.get("confidence", 0.0),
                        ),
                    )

                    # Calculate progress and call the callback
                    if progress_callback:
                        progress = (i + 1) / total_chunks * 100
                        progress_callback(progress)

            # Apply timestamp post-processing
            segments = self._process_timestamps(segments)

            logger.info("Transcription completed: %d segments", len(segments))
            return segments

        except Exception as e:
            logger.error("Error during transcription: %s", e)
            raise

    def _process_timestamps(
        self: T,
        segments: list[TranscriptionSegment],
    ) -> list[TranscriptionSegment]:
        """Process and clean up segment timestamps.

        Args:
            segments: List of transcription segments to process

        Returns:
            Processed segments with cleaned up timestamps
        """
        if not segments:
            return segments

        # Sort segments by start time
        segments.sort(key=lambda x: x.start)

        # Merge very close segments (threshold of 0.05s)
        merged_segments = []
        current_segment = None
        merge_threshold = 0.05

        for segment in segments:
            if current_segment is None:
                current_segment = segment
            elif (
                segment.start - current_segment.end <= merge_threshold
                and segment.speaker_id == current_segment.speaker_id
            ):
                # Merge very close segments
                current_segment.text += " " + segment.text
                current_segment.end = segment.end
                current_segment.confidence = (
                    current_segment.confidence + segment.confidence
                ) / 2
            else:
                merged_segments.append(current_segment)
                current_segment = segment

        if current_segment is not None:
            merged_segments.append(current_segment)

        return merged_segments

    def transcribe_stream(
        self: T,
        audio_stream: list[AudioSegment],
    ) -> list[TranscriptionSegment]:
        """Transcribe streaming audio data.

        Args:
            audio_stream: List of AudioSegment objects

        Returns:
            List of TranscriptionSegment objects
        """
        if not audio_stream:
            return []

        # Concatenate audio segments
        audio_data = np.concatenate([segment.data for segment in audio_stream])
        sample_rate = audio_stream[0].sample_rate

        return self.transcribe(audio_data, sample_rate)

    def __del__(self: T) -> None:
        """Cleanup when the object is deleted."""
        self.unload_model()
