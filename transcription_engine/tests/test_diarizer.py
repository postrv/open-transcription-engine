# File: transcription_engine/tests/test_diarizer.py
"""Tests for the speaker diarization module."""

import os
from collections.abc import Generator
from typing import Self
from unittest.mock import Mock, patch

import numpy as np
import pytest
from pytest import MonkeyPatch

from ..speaker_id.diarizer import PYANNOTE_AVAILABLE, DiarizationManager, SpeakerSegment
from ..whisper_engine.transcriber import TranscriptionSegment


@pytest.fixture(scope="function", autouse=False)  # type: ignore[misc]
def sample_audio() -> tuple[np.ndarray, int]:
    """Create sample audio data for testing.

    Returns:
        Tuple containing audio data array and sample rate
    """
    # Create 5 seconds of audio at 16kHz
    duration = 5
    sample_rate = 16000
    t = np.linspace(0, duration, duration * sample_rate)

    # Generate two channels with different frequencies
    channel1 = np.sin(2 * np.pi * 440 * t)  # 440 Hz
    channel2 = np.sin(2 * np.pi * 880 * t)  # 880 Hz

    # Combine into stereo
    audio_data = np.vstack((channel1, channel2)).T
    return audio_data, sample_rate


@pytest.mark.skipif(not PYANNOTE_AVAILABLE, reason="pyannote.audio not installed")
class TestPyannoteDiarization:
    """Test suite for PyAnnote-specific functionality."""

    @pytest.fixture(scope="function")  # type: ignore[misc]
    def mock_pyannote(self: Self) -> Generator[Mock, None, None]:
        """Provide a mocked PyAnnote pipeline.

        Returns:
            Mock object for the PyAnnote pipeline
        """
        with patch("pyannote.audio.Pipeline") as mock:
            # Create a mock pipeline
            mock_pipeline = Mock()
            mock_pipeline.to.return_value = mock_pipeline

            # Mock diarization output
            mock_output = Mock()
            mock_output.itertracks.return_value = [
                (Mock(start=0.0, end=2.0), None, "SPEAKER_1"),
                (Mock(start=2.1, end=4.0), None, "SPEAKER_2"),
            ]

            mock_pipeline.return_value = mock_output
            mock.from_pretrained.return_value = mock_pipeline

            yield mock

    @pytest.mark.skipif(  # type: ignore[misc]
        not PYANNOTE_AVAILABLE or "HF_TOKEN" not in os.environ,
        reason="pyannote.audio not installed or token not configured",
    )
    def test_pyannote_processing(
        self: Self,
        mock_pyannote: Mock,
        sample_audio: tuple[np.ndarray, int],
    ) -> None:
        """Test processing using PyAnnote.

        Args:
            mock_pyannote: Mock PyAnnote pipeline
            sample_audio: Sample audio data fixture
        """
        audio_data, sample_rate = sample_audio
        mono_audio = audio_data.mean(axis=1)

        manager = DiarizationManager(
            use_pyannote=True,
            auth_token=os.environ.get("HF_TOKEN", "dummy_token"),
        )
        segments = manager.process_singlechannel(mono_audio, sample_rate)

        # Verify segment count and speaker IDs
        pytest.assume(
            len(segments) == 2,
            "Expected 2 segments",
        )
        pytest.assume(
            segments[0].speaker_id == "SPEAKER_1",
            "Wrong speaker ID for first segment",
        )
        pytest.assume(
            segments[1].speaker_id == "SPEAKER_2",
            "Wrong speaker ID for second segment",
        )


class TestBasicDiarization:
    """Test suite for basic diarization functionality."""

    def test_device_selection(self: Self, monkeypatch: MonkeyPatch) -> None:
        """Test computation device selection logic.

        Args:
            monkeypatch: PyTest fixture for mocking
        """
        # Test CPU fallback
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "torch.backends.mps.is_available",
                return_value=False,
            ),
        ):
            manager = DiarizationManager(use_pyannote=False)
            pytest.assume(manager.device.type == "cpu", "Should default to CPU")

        # Test MPS selection
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "torch.backends.mps.is_available",
                return_value=True,
            ),
        ):
            manager = DiarizationManager(use_pyannote=False)
            pytest.assume(manager.device.type == "mps", "Should select MPS")

        # Test CUDA selection
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "torch.backends.mps.is_available",
                return_value=False,
            ),
        ):
            manager = DiarizationManager(use_pyannote=False)
            pytest.assume(manager.device.type == "cuda", "Should select CUDA")

    def test_multichannel_processing(
        self: Self,
        sample_audio: tuple[np.ndarray, int],
    ) -> None:
        """Test processing of multi-channel audio.

        Args:
            sample_audio: Sample audio data fixture
        """
        audio_data, sample_rate = sample_audio
        manager = DiarizationManager(use_pyannote=False)

        segments = manager.process_multichannel(audio_data, sample_rate)

        # Verify segment properties
        pytest.assume(len(segments) > 0, "Should produce at least one segment")
        pytest.assume(
            all(isinstance(seg, SpeakerSegment) for seg in segments),
            "All outputs should be SpeakerSegments",
        )
        pytest.assume(
            all(seg.channel in [0, 1] for seg in segments),
            "Channels should be 0 or 1",
        )
        pytest.assume(
            all(seg.speaker_id in ["speaker_1", "speaker_2"] for seg in segments),
            "Speaker IDs should match expected format",
        )

    def test_basic_singlechannel(
        self: Self,
        sample_audio: tuple[np.ndarray, int],
    ) -> None:
        """Test basic single-channel processing without PyAnnote.

        Args:
            sample_audio: Sample audio data fixture
        """
        audio_data, sample_rate = sample_audio
        mono_audio = audio_data.mean(axis=1)

        manager = DiarizationManager(use_pyannote=False)
        segments = manager.process_singlechannel(mono_audio, sample_rate)

        # Verify basic properties
        pytest.assume(len(segments) > 0, "Should produce at least one segment")
        pytest.assume(
            all(isinstance(seg, SpeakerSegment) for seg in segments),
            "All outputs should be SpeakerSegments",
        )
        pytest.assume(
            all(seg.speaker_id == "speaker_unknown" for seg in segments),
            "Speaker should be unknown without PyAnnote",
        )

    def test_speaker_assignment(self: Self) -> None:
        """Test assignment of speaker IDs to transcription segments."""
        manager = DiarizationManager(use_pyannote=False)

        # Create sample transcription segments
        transcription_segments = [
            TranscriptionSegment(
                text="Hello there",
                start=0.0,
                end=2.0,
                confidence=0.9,
            ),
            TranscriptionSegment(
                text="General Kenobi",
                start=2.5,
                end=4.0,
                confidence=0.85,
            ),
        ]

        # Create sample diarization segments
        diarization_segments = [
            SpeakerSegment(
                start=0.0,
                end=2.2,
                speaker_id="SPEAKER_1",
                score=0.95,
            ),
            SpeakerSegment(
                start=2.3,
                end=4.1,
                speaker_id="SPEAKER_2",
                score=0.92,
            ),
        ]

        # Test speaker assignment
        updated_segments = manager.assign_speaker_ids(
            transcription_segments,
            diarization_segments,
        )

        # Verify assignments
        pytest.assume(len(updated_segments) == 2, "Should preserve segment count")
        pytest.assume(
            updated_segments[0].speaker_id == "SPEAKER_1",
            "First segment should be assigned to SPEAKER_1",
        )
        pytest.assume(
            updated_segments[1].speaker_id == "SPEAKER_2",
            "Second segment should be assigned to SPEAKER_2",
        )

    def test_error_handling(self: Self) -> None:
        """Test error handling scenarios."""
        # Test invalid audio input for multichannel
        manager = DiarizationManager(use_pyannote=False)
        with pytest.raises(ValueError, match="Expected multi-channel audio"):
            manager.process_multichannel(np.array([1, 2, 3]), 16000)
