# File: transcription_engine/speaker_id/diarizer.py
"""
Speaker Diarization module for the Open Transcription Engine.
Supports both multi-channel (direct mapping) and single-channel (ML-based) diarization.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation
from collections import defaultdict

from ..utils.config import config_manager
from ..whisper_engine.transcriber import TranscriptionSegment

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """Container for speaker-labeled audio segments."""
    start: float
    end: float
    speaker_id: str
    channel: Optional[int] = None
    score: float = 1.0


class DiarizationManager:
    """Manages speaker diarization processes."""

    def __init__(self, use_pyannote: bool = True, auth_token: Optional[str] = None):
        """
        Initialize the diarization manager.

        Args:
            use_pyannote: Whether to use PyAnnote for single-channel diarization
            auth_token: HuggingFace token for PyAnnote (if using PyAnnote)
        """
        self.config = config_manager.load_config()
        self.device = self._setup_device()
        self.pipeline = None
        self.use_pyannote = use_pyannote
        self.auth_token = auth_token

    def _setup_device(self) -> torch.device:
        """Configure the computation device based on availability."""
        if torch.backends.mps.is_available():
            logger.info("Using MPS backend for diarization")
            return torch.device('mps')
        elif torch.cuda.is_available():
            logger.info("Using CUDA backend for diarization")
            return torch.device('cuda')
        else:
            logger.info("Using CPU backend for diarization")
            return torch.device('cpu')

    def _load_pyannote(self) -> bool:
        """Load the PyAnnote pipeline."""
        if self.pipeline is not None:
            return True

        try:
            if not self.auth_token:
                raise ValueError(
                    "PyAnnote requires a HuggingFace auth token. "
                    "Please set auth_token in the configuration."
                )

            # Initialize pipeline with authentication
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=self.auth_token
            )

            # Move to appropriate device
            self.pipeline = self.pipeline.to(self.device)

            logger.info("PyAnnote pipeline loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load PyAnnote pipeline: {e}")
            self.pipeline = None
            return False

    def process_multichannel(self, audio_data: np.ndarray,
                             sample_rate: int) -> List[SpeakerSegment]:
        """
        Process multi-channel audio where each channel represents a different speaker.

        Args:
            audio_data: Multi-channel audio data
            sample_rate: Audio sample rate

        Returns:
            List of SpeakerSegment objects
        """
        if len(audio_data.shape) != 2:
            raise ValueError("Expected multi-channel audio data")

        num_channels = audio_data.shape[1]
        segments: List[SpeakerSegment] = []

        # Process each channel to detect speech segments
        for channel in range(num_channels):
            channel_data = audio_data[:, channel]

            # Use voice activity detection to find speech segments
            speech_segments = self._detect_speech(channel_data, sample_rate)

            # Create speaker segments for this channel
            for start, end in speech_segments:
                segments.append(SpeakerSegment(
                    start=start,
                    end=end,
                    speaker_id=f"speaker_{channel + 1}",
                    channel=channel,
                    score=1.0  # High confidence for direct channel mapping
                ))

        # Sort segments by start time
        segments.sort(key=lambda x: x.start)
        return segments

    def process_singlechannel(self, audio_data: np.ndarray,
                              sample_rate: int) -> List[SpeakerSegment]:
        """
        Process single-channel audio using PyAnnote for speaker diarization.

        Args:
            audio_data: Single-channel audio data
            sample_rate: Audio sample rate

        Returns:
            List of SpeakerSegment objects
        """
        if not self.use_pyannote:
            raise ValueError("PyAnnote diarization is disabled")

        if not self._load_pyannote():
            raise RuntimeError("Failed to load PyAnnote pipeline")

        try:
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Run diarization
            diarization = self.pipeline({
                "waveform": torch.tensor(audio_data).unsqueeze(0),
                "sample_rate": sample_rate
            })

            # Convert PyAnnote output to our format
            segments: List[SpeakerSegment] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker_id=speaker,
                    score=0.95  # PyAnnote doesn't provide confidence scores
                ))

            # Sort segments by start time
            segments.sort(key=lambda x: x.start)
            return segments

        except Exception as e:
            logger.error(f"Error during PyAnnote diarization: {e}")
            raise

    def _detect_speech(self, audio_data: np.ndarray,
                       sample_rate: int) -> List[Tuple[float, float]]:
        """
        Detect speech segments in audio using energy-based VAD.

        Args:
            audio_data: Single channel audio data
            sample_rate: Audio sample rate

        Returns:
            List of (start, end) tuples in seconds
        """
        # Parameters for voice activity detection
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        frame_step = int(0.010 * sample_rate)  # 10ms step
        energy_threshold = 0.1  # Adjusted based on normalized audio
        min_speech_duration = 0.3  # Minimum speech segment duration in seconds

        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Calculate frame energies
        frames = []
        for i in range(0, len(audio_data) - frame_length, frame_step):
            frame = audio_data[i:i + frame_length]
            energy = np.sum(frame ** 2) / frame_length
            frames.append(energy)

        frames = np.array(frames)

        # Find speech segments
        is_speech = frames > energy_threshold

        # Convert frame indices to time segments
        segments = []
        start_frame = None

        for i in range(len(is_speech)):
            if is_speech[i] and start_frame is None:
                start_frame = i
            elif not is_speech[i] and start_frame is not None:
                duration = (i - start_frame) * frame_step / sample_rate
                if duration >= min_speech_duration:
                    start_time = start_frame * frame_step / sample_rate
                    end_time = i * frame_step / sample_rate
                    segments.append((start_time, end_time))
                start_frame = None

        # Handle the last segment
        if start_frame is not None:
            duration = (len(is_speech) - start_frame) * frame_step / sample_rate
            if duration >= min_speech_duration:
                start_time = start_frame * frame_step / sample_rate
                end_time = len(audio_data) / sample_rate
                segments.append((start_time, end_time))

        return segments

    def assign_speaker_ids(self, transcription_segments: List[TranscriptionSegment],
                           diarization_segments: List[SpeakerSegment]) -> List[TranscriptionSegment]:
        """
        Assign speaker IDs to transcription segments based on diarization results.

        Args:
            transcription_segments: List of transcription segments
            diarization_segments: List of speaker segments

        Returns:
            Updated list of transcription segments with speaker IDs
        """
        # Create a map of time ranges to speaker IDs
        speaker_map = {}
        for seg in diarization_segments:
            speaker_map[(seg.start, seg.end)] = seg.speaker_id

        # Assign speakers to transcription segments
        for trans_seg in transcription_segments:
            # Find overlapping diarization segments
            matching_speakers = defaultdict(float)

            for (start, end), speaker_id in speaker_map.items():
                overlap_start = max(trans_seg.start, start)
                overlap_end = min(trans_seg.end, end)

                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    matching_speakers[speaker_id] += overlap_duration

            # Assign the speaker with maximum overlap
            if matching_speakers:
                trans_seg.speaker_id = max(
                    matching_speakers.items(),
                    key=lambda x: x[1]
                )[0]

        return transcription_segments

    def __del__(self):
        """Cleanup when the object is deleted."""
        if self.pipeline is not None:
            del self.pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()