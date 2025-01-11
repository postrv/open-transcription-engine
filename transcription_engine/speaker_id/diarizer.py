# File: transcription_engine/speaker_id/diarizer.py
# transcription_engine/speaker_id/diarizer.py
"""Speaker diarization module for the Open Transcription Engine.

Provides functionality for identifying and separating different speakers
in audio recordings, supporting both multi-channel and single-channel inputs.
"""

# Use importlib.util to check for pyannote availability
import importlib.util
import logging
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

# Move these up before other imports
from ..utils.config import DiarizationConfig, config_manager
from ..whisper_engine.transcriber import TranscriptionSegment

PYANNOTE_AVAILABLE = importlib.util.find_spec("pyannote") is not None

if PYANNOTE_AVAILABLE:
    from pyannote.audio import Pipeline
    # Remove the unused imports
else:
    Pipeline = None

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="DiarizationManager")


@dataclass
class SpeakerSegment:
    """Container for speaker-labeled audio segments."""

    start: float
    end: float
    speaker_id: str
    channel: int | None = None
    score: float = 1.0


class DiarizationManager:
    """Manages speaker diarization processes."""

    def __init__(
        self: "DiarizationManager", config: DiarizationConfig | None = None
    ) -> None:
        """Initialize the diarization manager.

        Args:
            config: Optional DiarizationConfig instance
        """
        self.auth_token = None
        self.config = config or config_manager.load_config().diarization
        self.pipeline: Pipeline | None = None
        self.device = self._setup_device()
        self.use_pyannote = self.config.use_pyannote

        # Initialize pipeline if using pyannote
        if self.use_pyannote:
            self._load_pyannote()

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
        except (RuntimeError, ValueError) as e:
            logger.warning(
                "Requested device %s failed, falling back to CPU: %s",
                self.config.device,
                e,
            )
            return torch.device("cpu")

    def _load_pyannote(self: "DiarizationManager") -> bool:
        """Load the PyAnnote pipeline."""
        if not PYANNOTE_AVAILABLE:
            logger.warning("PyAnnote not available.")
            return False

        if self.pipeline is not None:
            # Already loaded
            return True

        try:
            if not self.auth_token:
                msg = (
                    "PyAnnote requires a HuggingFace auth token. "
                    "Please set auth_token in the configuration."
                )
                raise ValueError(msg)

            pipeline_obj = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=self.auth_token,
            )

            pipeline_obj = pipeline_obj.to(self.device)
            self.pipeline = pipeline_obj

            logger.info("PyAnnote pipeline loaded successfully")
            return True

        except (ImportError, ValueError, RuntimeError) as e:
            # Catch narrower exceptions
            logger.error("Failed to load PyAnnote pipeline: %s", e)
            self.pipeline = None
            return False

    def process_multichannel(
        self: "DiarizationManager",
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> list[SpeakerSegment]:
        """Process multi-channel audio where each channel is a different speaker.

        Args:
            audio_data: Multi-channel audio data
            sample_rate: Audio sample rate

        Returns:
            List of SpeakerSegment objects

        Raises:
            ValueError: If audio data is not multi-channel
        """
        if len(audio_data.shape) != 2:
            msg = "Expected multi-channel audio data"
            raise ValueError(msg)

        num_channels = audio_data.shape[1]
        segments: list[SpeakerSegment] = []

        # Process each channel to detect speech segments
        for channel in range(num_channels):
            channel_data = audio_data[:, channel]

            # Use voice activity detection to find speech segments
            speech_segments = self._detect_speech(channel_data, sample_rate)

            # Create speaker segments for this channel
            for start, end in speech_segments:
                segments.append(
                    SpeakerSegment(
                        start=start,
                        end=end,
                        speaker_id=f"speaker_{channel + 1}",
                        channel=channel,
                        score=1.0,  # High confidence for direct channel mapping
                    ),
                )

        # Sort segments by start time
        segments.sort(key=lambda x: x.start)
        return segments

    def process_singlechannel(
        self: "DiarizationManager",
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> list[SpeakerSegment]:
        """Process single-channel audio using available diarization method.

        Args:
            audio_data: Single channel audio data
            sample_rate: Audio sample rate

        Returns:
            List of SpeakerSegment objects
        """
        if self.use_pyannote and self._load_pyannote():
            return self._process_with_pyannote(audio_data, sample_rate)

        # Fallback to basic energy-based segmentation
        logger.warning(
            "Using basic energy-based segmentation (no speaker identification)",
        )
        segments = self._detect_speech(audio_data, sample_rate)
        return [
            SpeakerSegment(
                start=start,
                end=end,
                speaker_id="speaker_unknown",
                score=0.5,
            )
            for start, end in segments
        ]

    def _process_with_pyannote(
        self: "DiarizationManager",
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> list[SpeakerSegment]:
        """Process audio using PyAnnote pipeline."""
        try:
            if self.pipeline is None:
                error_message = "PyAnnote pipeline must be loaded first"
                raise AssertionError(error_message)

            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            diarization = self.pipeline(
                {
                    "waveform": torch.tensor(audio_data).unsqueeze(0),
                    "sample_rate": sample_rate,
                },
            )

            # Convert PyAnnote output to our format
            segments: list[SpeakerSegment] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    SpeakerSegment(
                        start=turn.start,
                        end=turn.end,
                        speaker_id=speaker,
                        score=0.95,  # PyAnnote doesn't provide confidence scores
                    ),
                )

            # Sort segments by start time
            segments.sort(key=lambda x: x.start)
            return segments

        except Exception as e:  # Possibly narrower: (RuntimeError, ValueError, etc.)
            logger.error("Error during PyAnnote diarization: %s", e)

            raise

    def _detect_speech(
        self: "DiarizationManager",
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> list[tuple[float, float]]:
        """Energy-based VAD."""
        frame_length = int(0.025 * sample_rate)
        frame_step = int(0.010 * sample_rate)
        energy_threshold = 0.1
        min_speech_duration = 0.3

        audio_data = audio_data / np.max(np.abs(audio_data))

        frame_energies: list[float] = []
        for i in range(0, len(audio_data) - frame_length, frame_step):
            frame = audio_data[i : i + frame_length]
            energy = float(np.sum(frame**2) / frame_length)
            frame_energies.append(energy)

        frames_array: NDArray[np.float64] = np.array(frame_energies, dtype=np.float64)
        is_speech = frames_array > energy_threshold

        # Convert bool array to segments
        segments: list[tuple[float, float]] = []
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

    def assign_speaker_ids(
        self: "DiarizationManager",
        transcription_segments: list[TranscriptionSegment],
        diarization_segments: list[SpeakerSegment],
    ) -> list[TranscriptionSegment]:
        """Assign speaker IDs to transcription segments.

        Based on diarization segments.
        """
        from collections import defaultdict

        # specify dict types
        speaker_map: dict[tuple[float, float], str] = {}
        for seg in diarization_segments:
            speaker_map[(seg.start, seg.end)] = seg.speaker_id

        for trans_seg in transcription_segments:
            matching_speakers: defaultdict[str, float] = defaultdict(float)

            for (start, end), speaker_id in speaker_map.items():
                overlap_start = max(trans_seg.start, start)
                overlap_end = min(trans_seg.end, end)

                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    matching_speakers[speaker_id] += overlap_duration

            if matching_speakers:
                trans_seg.speaker_id = max(
                    matching_speakers.items(),
                    key=lambda x: x[1],
                )[0]

        return transcription_segments

    def __del__(self: "DiarizationManager") -> None:
        """Cleanup when the object is deleted."""
        if self.pipeline is not None:
            del self.pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
