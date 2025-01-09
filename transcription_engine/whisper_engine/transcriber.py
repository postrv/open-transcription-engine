# File: transcription_engine/whisper_engine/transcriber.py
"""
Whisper integration module for the Open Transcription Engine.
Handles model loading, chunking, and transcription with GPU support.
"""

import torch
import whisper
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta

from ..utils.config import config_manager, WhisperConfig
from ..audio_input.recorder import AudioSegment

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """Container for transcribed text and its metadata."""
    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    speaker_id: Optional[str] = None
    confidence: float = 0.0


class WhisperManager:
    """Manages Whisper model lifecycle and transcription."""

    # Mapping of model sizes to their memory requirements (in GB)
    MODEL_MEMORY_REQUIREMENTS = {
        'tiny': 1,
        'base': 1,
        'small': 2,
        'medium': 5,
        'large': 10
    }

    def __init__(self, config: Optional[WhisperConfig] = None):
        """Initialize the Whisper manager with configuration."""
        self.config = config or config_manager.load_config().whisper
        self.model = None
        self.device = self._setup_device()
        self._validate_memory()

    def _setup_device(self) -> torch.device:
        """Configure the computation device based on availability and config."""
        if self.config.device == 'auto':
            if torch.backends.mps.is_available():
                logger.info("Using MPS backend")
                return torch.device('mps')
            elif torch.cuda.is_available():
                logger.info("Using CUDA backend")
                return torch.device('cuda')
            else:
                logger.info("Using CPU backend")
                return torch.device('cpu')
        return torch.device(self.config.device)

    def _validate_memory(self):
        """Validate if system has enough memory for the chosen model."""
        required_memory = self.MODEL_MEMORY_REQUIREMENTS[self.config.model_size]

        if self.device.type == 'cuda':
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        elif self.device.type == 'mps':
            # For M1 Macs, we'll assume they have enough memory as they share system RAM
            available_memory = 16  # Conservative estimate
        else:
            import psutil
            available_memory = psutil.virtual_memory().total / 1e9

        if available_memory < required_memory:
            logger.warning(
                f"Available memory ({available_memory:.1f}GB) may be insufficient "
                f"for {self.config.model_size} model ({required_memory}GB required)"
            )

    def load_model(self):
        """Load the Whisper model into memory."""
        try:
            logger.info(f"Loading Whisper model: {self.config.model_size}")

            # Handle compute type
            compute_type = self.config.compute_type
            if self.device.type == 'mps' and compute_type == 'float16':
                # MPS doesn't support float16, fallback to float32
                compute_type = 'float32'
                logger.info("MPS detected, using float32 instead of float16")

            self.model = whisper.load_model(
                self.config.model_size,
                device=self.device,
                download_root=Path.home() / '.cache' / 'whisper'
            )

            # Set the compute type
            if compute_type == 'float16' and self.device.type != 'cpu':
                self.model = self.model.half()

            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            return False

    def unload_model(self):
        """Unload the model to free up memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            logger.info("Model unloaded")

    def _prepare_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Prepare audio data for Whisper model."""
        # Whisper expects mono audio at 16kHz
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # Convert to mono

        if sample_rate != 16000:
            # Resample to 16kHz
            from scipy import signal
            audio_data = signal.resample(
                audio_data,
                int(len(audio_data) * 16000 / sample_rate)
            )

        return audio_data

    def _chunk_audio(self, audio_data: np.ndarray,
                     chunk_duration: int = 30) -> List[Tuple[np.ndarray, float]]:
        """Split audio into chunks with timestamps."""
        sample_rate = 16000  # Whisper's expected sample rate
        chunk_size = chunk_duration * sample_rate
        chunks = []

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < sample_rate:  # Skip chunks shorter than 1 second
                continue
            start_time = i / sample_rate
            chunks.append((chunk, start_time))

        return chunks

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> List[TranscriptionSegment]:
        """
        Transcribe audio data into text with timestamps.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            List of TranscriptionSegment objects
        """
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Failed to load Whisper model")

        try:
            # Prepare audio
            audio_data = self._prepare_audio(audio_data, sample_rate)

            # Split into chunks for long audio
            chunks = self._chunk_audio(audio_data)
            segments = []

            for chunk, start_time in chunks:
                # Ensure audio is in the correct format for Whisper
                if self.device.type == 'mps':
                    # MPS requires contiguous tensors
                    chunk = torch.tensor(chunk).contiguous().to(self.device)
                else:
                    chunk = torch.tensor(chunk).to(self.device)

                # Run transcription
                result = self.model.transcribe(
                    chunk,
                    language=self.config.language,
                    task='transcribe',
                    batch_size=self.config.batch_size,
                    fp16=(self.config.compute_type == 'float16')
                )

                # Process segments
                for segment in result['segments']:
                    segments.append(TranscriptionSegment(
                        text=segment['text'].strip(),
                        start=start_time + segment['start'],
                        end=start_time + segment['end'],
                        confidence=segment.get('confidence', 0.0)
                    ))

            # Sort segments by start time
            segments.sort(key=lambda x: x.start)

            # Merge consecutive segments with same speaker
            merged_segments = []
            current_segment = None

            for segment in segments:
                if current_segment is None:
                    current_segment = segment
                elif (segment.start - current_segment.end <= 0.3 and  # Close in time
                      segment.speaker_id == current_segment.speaker_id):  # Same speaker
                    current_segment.text += ' ' + segment.text
                    current_segment.end = segment.end
                    current_segment.confidence = (current_segment.confidence + segment.confidence) / 2
                else:
                    merged_segments.append(current_segment)
                    current_segment = segment

            if current_segment is not None:
                merged_segments.append(current_segment)

            logger.info(f"Transcription completed: {len(merged_segments)} segments")
            return merged_segments

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise

    def transcribe_stream(self, audio_stream: List[AudioSegment]) -> List[TranscriptionSegment]:
        """
        Transcribe streaming audio data.

        Args:
            audio_stream: List of AudioSegment objects

        Returns:
            List of TranscriptionSegment objects
        """
        # Concatenate audio segments
        audio_data = np.concatenate([segment.data for segment in audio_stream])
        sample_rate = audio_stream[0].sample_rate if audio_stream else 16000

        return self.transcribe(audio_data, sample_rate)

    def __del__(self):
        """Cleanup when the object is deleted."""
        self.unload_model()