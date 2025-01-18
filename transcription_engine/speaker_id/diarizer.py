# File: transcription_engine/speaker_id/diarizer.py
"""Manages speaker diarization processes with enhanced accuracy, confidence metrics."""

import importlib.util
import logging
import os
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

from ..utils.config import DiarizationConfig, config_manager
from ..whisper_engine.transcriber import TranscriptionSegment

# Configure logging
logger = logging.getLogger(__name__)

PYANNOTE_AVAILABLE = importlib.util.find_spec("pyannote") is not None

if PYANNOTE_AVAILABLE:
    from pyannote.audio import Pipeline
else:
    Pipeline = None

T = TypeVar("T", bound="DiarizationManager")


@dataclass
class SpeakerSegment:
    """Container for speaker-labeled audio segments with enhanced metadata."""

    start: float
    end: float
    speaker_id: str
    channel: int | None = None
    score: float = 1.0
    overlap_detected: bool = False
    energy_score: float = 0.0  # Voice activity confidence
    embedding_similarity: float = 0.0  # Speaker embedding similarity score


@dataclass
class DiarizationMetrics:
    """Container for diarization quality metrics."""

    total_segments: int = 0
    average_confidence: float = 0.0
    overlap_segments: int = 0
    silence_ratio: float = 0.0
    speaker_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self: "DiarizationMetrics") -> None:
        """Initialize empty speaker counts if none provided."""
        if self.speaker_counts is None:
            self.speaker_counts = {}


class DiarizationManager:
    """Manages speaker diarization with enhanced accuracy and metrics."""

    def __init__(self: T, config: DiarizationConfig | None = None) -> None:
        """Initialize the diarization manager with configuration."""
        self.auth_token: str | None = None
        self.config = config or config_manager.load_config().diarization
        self.pipeline: "Pipeline" | None = None
        self.device = self._setup_device()
        self.use_pyannote = self.config.use_pyannote
        self.metrics: DiarizationMetrics | None = None

        # Initialize metrics
        self.reset_metrics()

        # Initialize pipeline if using pyannote
        if self.use_pyannote:
            self._load_pyannote()

    def reset_metrics(self: T) -> None:
        """Reset diarization metrics."""
        self.metrics = DiarizationMetrics()

    def _validate_audio_energy(self: T, audio_data: NDArray[np.float64]) -> float:
        """Validate audio energy levels for more accurate VAD.

        Args:
            audio_data: Audio samples to analyze

        Returns:
            float: Energy confidence score
        """
        if len(audio_data) == 0:
            return 0.0

        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Calculate RMS energy
        frame_length = 1024
        hop_length = 512

        frames = [
            audio_data[i : i + frame_length]
            for i in range(0, len(audio_data) - frame_length, hop_length)
        ]

        if not frames:
            return 0.0

        frame_rms = np.array([np.sqrt(np.mean(frame**2)) for frame in frames])

        # Calculate confidence based on energy distribution
        energy_mean = np.mean(frame_rms)
        energy_std = np.std(frame_rms)

        # Higher score for consistent energy levels
        energy_confidence = 1.0 - min(1.0, energy_std / (energy_mean + 1e-6))

        return float(energy_confidence)

    # Fix the _detect_speaker_overlap method
    def _detect_speaker_overlap(
        self: T,
        audio_data: NDArray[np.float64],
        sample_rate: int,
        segment: SpeakerSegment,
    ) -> bool:
        """Detect potential speaker overlap in segment."""
        # Convert times to samples
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)

        if start_sample >= end_sample or end_sample > len(audio_data):
            return False

        segment_audio = audio_data[start_sample:end_sample]

        # Calculate spectral flatness
        spec = np.abs(np.fft.rfft(segment_audio))
        geometric_mean = float(
            np.exp(np.mean(np.log(spec + 1e-6)))
        )  # Explicit float casting
        arithmetic_mean = float(np.mean(spec))  # Explicit float casting

        flatness = geometric_mean / (arithmetic_mean + 1e-6)

        # Lower flatness often indicates multiple speakers
        return bool(flatness < 0.1)  # Explicit boolean return

    def _calculate_segment_confidence(
        self: T,
        segment: SpeakerSegment,
        metrics: DiarizationMetrics,
    ) -> float:
        """Calculate overall confidence score for a segment.

        Args:
            segment: Speaker segment to evaluate
            metrics: Current diarization metrics

        Returns:
            float: Combined confidence score
        """
        # Base score from diarization
        confidence = segment.score

        # Penalize if overlap detected
        if segment.overlap_detected:
            confidence *= 0.8

        # Factor in energy score
        confidence *= 0.5 + 0.5 * segment.energy_score

        # Consider speaker frequency
        speaker_count = metrics.speaker_counts.get(segment.speaker_id, 0)
        total_segments = metrics.total_segments
        if total_segments > 0:
            frequency = speaker_count / total_segments
            # Penalize very rare or overly frequent speakers
            if frequency < 0.1 or frequency > 0.9:
                confidence *= 0.9

        # Ensure score is in [0,1]
        return float(max(0.0, min(1.0, confidence)))

    def process_singlechannel(
        self: "DiarizationManager",
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> list[SpeakerSegment]:
        """Process single-channel audio with enhanced speaker detection.

        Args:
            audio_data: Audio samples to process
            sample_rate: Audio sample rate

        Returns:
            List of speaker segments with metadata
        """
        # Reset metrics
        self.reset_metrics()

        try:
            if self.use_pyannote and self._load_pyannote():
                # Convert to mono if needed
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)

                # Normalize audio
                audio_data = audio_data / np.max(np.abs(audio_data))

                # Create torch tensor
                waveform = torch.tensor(audio_data).unsqueeze(0)

                # Run diarization
                try:
                    if self.pipeline is not None:
                        diarization = self.pipeline(
                            {"waveform": waveform, "sample_rate": sample_rate},
                            num_speakers=None,  # Let the model determine speakers
                            min_speakers=self.pipeline_params.get("min_speakers", 1),
                            max_speakers=self.pipeline_params.get("max_speakers", 6),
                        )

                        segments: list[SpeakerSegment] = []

                        # Process each segment with enhanced metadata
                        for turn, _, speaker in diarization.itertracks(
                            yield_label=True
                        ):
                            # Get segment audio
                            start_sample = int(turn.start * sample_rate)
                            end_sample = min(
                                int(turn.end * sample_rate), len(audio_data)
                            )

                            if start_sample >= end_sample:
                                continue

                            segment_audio = audio_data[start_sample:end_sample]

                            # Calculate energy score
                            energy_score = self._validate_audio_energy(segment_audio)

                            # Create enhanced segment
                            segment = SpeakerSegment(
                                start=turn.start,
                                end=turn.end,
                                speaker_id=speaker,
                                score=0.95,  # Base confidence score
                                overlap_detected=False,
                                energy_score=energy_score,
                            )

                            # Check for speaker overlap
                            segment.overlap_detected = self._detect_speaker_overlap(
                                audio_data, sample_rate, segment
                            )

                            # Calculate final confidence
                            segment.score = self._calculate_segment_confidence(
                                segment,
                                self.metrics
                                if self.metrics is not None
                                else DiarizationMetrics(),
                            )

                            segments.append(segment)

                            # Update metrics
                            if self.metrics is not None:
                                self.metrics.total_segments += 1
                                if segment.overlap_detected:
                                    self.metrics.overlap_segments += 1
                                self.metrics.speaker_counts[speaker] = (
                                    self.metrics.speaker_counts.get(speaker, 0) + 1
                                )

                        # Sort segments by start time
                        segments.sort(key=lambda x: x.start)

                        # Update metrics
                        if segments and self.metrics is not None:
                            self.metrics.average_confidence = sum(
                                s.score for s in segments
                            ) / len(segments)

                        return segments

                except (RuntimeError, ValueError, ImportError) as e:
                    logger.error(f"Error during diarization: {str(e)}")
                    # Fall back to basic segmentation
                    logger.warning("Falling back to basic segmentation")
                    speech_segments = self._detect_speech(audio_data, sample_rate)
                    return self._convert_to_speaker_segments(
                        speech_segments, audio_data, sample_rate
                    )

            # No pyannote available
            logger.warning("Using basic segmentation (no speaker identification)")
            speech_segments = self._detect_speech(audio_data, sample_rate)
            return self._convert_to_speaker_segments(
                speech_segments, audio_data, sample_rate
            )

        except Exception as e:
            logger.error("Error in speaker diarization: %s", e)
            raise

    def _convert_to_speaker_segments(
        self: "DiarizationManager",
        time_segments: list[tuple[float, float]],
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> list[SpeakerSegment]:
        """Convert time segments to speaker segments with basic detection.

        Args:
            time_segments: List of (start, end) time tuples
            audio_data: Audio data array
            sample_rate: Audio sample rate

        Returns:
            List of speaker segments
        """
        return [
            SpeakerSegment(
                start=start,
                end=end,
                speaker_id="SPEAKER_00",  # Default speaker ID for basic detection
                score=0.5,  # Medium confidence for basic detection
                energy_score=self._validate_audio_energy(
                    audio_data[int(start * sample_rate) : int(end * sample_rate)]
                ),
            )
            for start, end in time_segments
        ]

    def get_metrics(self: T) -> DiarizationMetrics:
        """Get current diarization metrics.

        Returns:
            DiarizationMetrics object with current stats
        """
        return self.metrics if self.metrics is not None else DiarizationMetrics()

    def _load_pyannote(self: T) -> bool:
        """Load the PyAnnote pipeline with optimized settings."""
        if not PYANNOTE_AVAILABLE:
            logger.warning("PyAnnote not available - please install pyannote.audio")
            return False

        if self.pipeline is not None:
            # Already loaded
            return True

        try:
            # Check auth token
            if not self.auth_token:
                auth_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
                if auth_token:
                    self.auth_token = auth_token
                    logger.info("Using HuggingFace token from environment")
                else:
                    msg = (
                        "PyAnnote requires a HuggingFace auth token. "
                        "Please set HUGGINGFACE_TOKEN in .env or auth_token in config"
                    )
                    raise ValueError(msg)

            logger.info("Loading PyAnnote pipeline with device: %s", self.device)

            if Pipeline is None:
                msg = "PyAnnote Pipeline is not available"
                raise ImportError(msg)

            # Create pipeline with optimized parameters
            try:
                model_name = "pyannote/speaker-diarization"
                pipeline_obj = Pipeline.from_pretrained(
                    model_name, use_auth_token=self.auth_token
                )

                # Configure parameters compatible with pyannote.audio 3.x
                self.pipeline_params = {
                    # Segmentation parameters
                    "segmentation": {
                        "min_duration_on": 0.1,  # Minimum speech duration
                        "min_duration_off": 0.1,  # Minimum silence duration
                    },
                    # Speaker clustering parameters
                    "clustering": {
                        "method": "centroid",  # Clustering method
                        "min_cluster_size": 6,  # Minimum segments per speaker
                        "threshold": 0.715,  # Clustering threshold
                    },
                    # Other parameters
                    "min_speakers": 2,  # Set for court context
                    "max_speakers": 6,
                }

            except RuntimeError as e:
                msg = (
                    "Failed to create PyAnnote pipeline. "
                    f"Error: {str(e)}. Please verify model access at "
                    f"https://huggingface.co/{model_name}"
                )
                logger.error(msg)
                raise ValueError(msg) from e

            # Handle version warnings
            logger.warning(
                "Note: Running PyAnnote 3.x with newer PyTorch version. "
                "This is expected and should work despite version warnings."
            )

            if pipeline_obj is None:
                msg = "Failed to initialize PyAnnote pipeline."
                raise ValueError(msg)

            # Move to device after creation
            try:
                pipeline_obj = pipeline_obj.to(self.device)
                self.pipeline = pipeline_obj
            except (RuntimeError, ValueError, ImportError, AttributeError) as e:
                msg = f"Failed to move pipeline to device {self.device}: {str(e)}"
                logger.error(msg)
                raise RuntimeError(msg) from e

            logger.info("PyAnnote pipeline loaded successfully")
            return True

        except (ImportError, RuntimeError, ValueError, AttributeError) as e:
            logger.error("Error loading PyAnnote: %s", e)
            self.pipeline = None
            return False

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
            for segment in speech_segments:
                start, end = segment
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

    # Update the _detect_speech method to return proper types
    def _detect_speech(
        self: "DiarizationManager",
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> list[tuple[float, float]]:  # Return type updated to be explicit
        """Energy-based voice activity detection.

        Args:
            audio_data: Audio data array
            sample_rate: Audio sample rate

        Returns:
            List of tuples containing (start_time, end_time)
        """
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

        # Create a list of speech segments as tuples
        segments: list[tuple[float, float]] = []
        start_frame: int | None = None

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
        """Assign speaker IDs to transcription segments."""
        if not diarization_segments:
            return transcription_segments

        speaker_map: dict[tuple[float, float], dict[str, Any]] = {}
        for seg in diarization_segments:
            start = round(seg.start, 2)
            end = round(seg.end, 2)
            speaker_map[(start, end)] = {
                "speaker_id": seg.speaker_id,
                "confidence": float(seg.score),  # Explicit float
                "overlap_detected": bool(seg.overlap_detected),  # Explicit bool
                "energy_score": float(seg.energy_score),  # Explicit float
            }

        for trans_seg in transcription_segments:
            trans_start = round(trans_seg.start, 2)
            trans_end = round(trans_seg.end, 2)

            matching_speakers: dict[str, dict[str, float | bool]] = {}
            total_overlap = 0.0

            for (start, end), speaker_info in speaker_map.items():
                overlap_start = max(trans_start, start)
                overlap_end = min(trans_end, end)

                if overlap_end > overlap_start:
                    overlap_duration = float(
                        overlap_end - overlap_start
                    )  # Explicit float
                    speaker_id = str(speaker_info["speaker_id"])  # Explicit string

                    if speaker_id not in matching_speakers:
                        matching_speakers[speaker_id] = {
                            "duration": 0.0,
                            "confidence": 0.0,
                            "overlap_detected": False,
                            "energy_score": 0.0,
                        }

                    info = matching_speakers[speaker_id]
                    info["duration"] = float(info["duration"]) + overlap_duration
                    info["confidence"] = max(
                        float(info["confidence"]), float(speaker_info["confidence"])
                    )
                    info["overlap_detected"] = bool(info["overlap_detected"]) | bool(
                        speaker_info["overlap_detected"]
                    )
                    info["energy_score"] = max(
                        float(info["energy_score"]), float(speaker_info["energy_score"])
                    )

                    total_overlap += overlap_duration

            if matching_speakers:
                # Find the dominant speaker with proper type annotations
                dominant_speaker = max(
                    matching_speakers.items(),
                    key=lambda x: float(
                        x[1]["duration"]
                    ),  # Explicit float for comparison
                )

                # Only assign if the speaker covers a significant portion
                if float(dominant_speaker[1]["duration"]) / total_overlap > 0.5:
                    trans_seg.speaker_id = str(dominant_speaker[0])
                    trans_seg.confidence = float(dominant_speaker[1]["confidence"])

                    # Add diarization metadata with proper types
                    trans_seg.diarization_data = {
                        "overlap_detected": bool(
                            dominant_speaker[1]["overlap_detected"]
                        ),
                        "energy_score": float(dominant_speaker[1]["energy_score"]),
                        "coverage": float(dominant_speaker[1]["duration"])
                        / total_overlap,
                    }
                else:
                    # Multiple speakers detected
                    trans_seg.speaker_id = "MULTIPLE_SPEAKERS"
                    trans_seg.confidence = 0.5
                    trans_seg.diarization_data = {
                        "overlap_detected": True,
                        "energy_score": max(
                            float(s["energy_score"]) for s in matching_speakers.values()
                        ),
                        "coverage": 1.0,
                    }
            else:
                # No speaker found for this segment
                trans_seg.speaker_id = "UNKNOWN"
                trans_seg.confidence = 0.3
                trans_seg.diarization_data = {
                    "overlap_detected": False,
                    "energy_score": 0.0,
                    "coverage": 0.0,
                }

        return transcription_segments

    def __del__(self: "DiarizationManager") -> None:
        """Cleanup when the object is deleted."""
        if self.pipeline is not None:
            del self.pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
