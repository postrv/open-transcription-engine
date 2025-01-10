# File: transcription_engine/tests/test_transcriber.py
"""Tests for the Whisper transcription engine module."""

from collections.abc import Generator
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from pytest import FixtureRequest

from ..utils.config import WhisperConfig
from ..whisper_engine.transcriber import (
    TranscriptionSegment,
    WhisperManager,
)


@pytest.fixture  # type: ignore[misc]
def whisper_config(request: FixtureRequest) -> WhisperConfig:
    """Provide test Whisper configuration.

    Args:
        request: Pytest fixture request

    Returns:
        WhisperConfig: Test configuration instance
    """
    return WhisperConfig(
        model_size="tiny",
        device="cpu",  # Force CPU for consistent testing
        language="en",
        batch_size=16,
        compute_type="float32",
    )


@pytest.fixture  # type: ignore[misc]
def mock_whisper(request: FixtureRequest) -> Generator[Mock, None, None]:
    """Provide a mocked Whisper model.

    Args:
        request: Pytest fixture request

    Returns:
        Mock: Mock object simulating Whisper model
    """
    with patch("whisper.load_model") as mock:
        # Create a mock model with segments that are clearly separate in time
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "segments": [
                {
                    "text": " Test transcript one.",
                    "start": 0.0,
                    "end": 2.0,
                    "confidence": 0.95,
                },
                {
                    "text": " Test transcript two.",
                    "start": 2.5,  # Increased gap between segments
                    "end": 4.0,
                    "confidence": 0.92,
                },
            ]
        }
        mock.return_value = mock_model
        yield mock


class TestWhisperManager:
    """Test suite for WhisperManager class."""

    def test_device_selection(
        self: "TestWhisperManager",
        whisper_config: WhisperConfig,
    ) -> None:
        """Test computation device selection logic.

        Args:
            whisper_config: Test configuration fixture
        """
        # Test CPU fallback
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            manager = WhisperManager(whisper_config)
            pytest.assume(
                manager.device.type == "cpu",
                "Should default to CPU when no GPU available",
            )

        # Test MPS selection
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            whisper_config.device = "auto"
            manager = WhisperManager(whisper_config)
            pytest.assume(
                manager.device.type == "mps",
                "Should select MPS when available",
            )

        # Skip CUDA test on Mac
        if not torch.backends.mps.is_available():
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.backends.mps.is_available", return_value=False),
            ):
                whisper_config.device = "auto"
                manager = WhisperManager(whisper_config)
                pytest.assume(
                    manager.device.type == "cuda",
                    "Should select CUDA when available",
                )

    def test_model_loading(
        self: "TestWhisperManager",
        whisper_config: WhisperConfig,
        mock_whisper: Mock,
    ) -> None:
        """Test Whisper model loading and unloading.

        Args:
            whisper_config: Test configuration fixture
            mock_whisper: Mock Whisper model
        """
        manager = WhisperManager(whisper_config)
        pytest.assume(manager.load_model(), "Model should load successfully")
        pytest.assume(
            manager.model is not None, "Model should be available after loading"
        )

        # Test unloading
        manager.unload_model()
        pytest.assume(manager.model is None, "Model should be None after unloading")

    def test_audio_preparation(
        self: "TestWhisperManager",
        whisper_config: WhisperConfig,
    ) -> None:
        """Test audio preparation for Whisper.

        Args:
            whisper_config: Test configuration fixture
        """
        manager = WhisperManager(whisper_config)

        # Test stereo to mono conversion
        stereo_audio = np.random.rand(1000, 2)
        mono_audio = manager._prepare_audio(stereo_audio, 16000)
        pytest.assume(
            len(mono_audio.shape) == 1,
            "Should convert stereo to mono",
        )

        # Test resampling
        audio_44k = np.random.rand(44100)
        audio_16k = manager._prepare_audio(audio_44k, 44100)
        pytest.assume(
            len(audio_16k) == 16000,
            "Should resample to 16kHz",
        )

    def test_audio_chunking(
        self: "TestWhisperManager",
        whisper_config: WhisperConfig,
    ) -> None:
        """Test audio chunking for long recordings.

        Args:
            whisper_config: Test configuration fixture
        """
        manager = WhisperManager(whisper_config)

        # Create 60 seconds of audio at 16kHz
        audio_data = np.random.rand(16000 * 60)
        chunks = manager._chunk_audio(audio_data)

        pytest.assume(len(chunks) == 2, "Should split into 2 30-second chunks")
        for chunk, _ in chunks:  # Ignore timestamp in loop var
            pytest.assume(
                len(chunk) <= 16000 * 30,
                "Each chunk should be ≤ 30 seconds",
            )

    def test_transcription(
        self: "TestWhisperManager",
        whisper_config: WhisperConfig,
        mock_whisper: Mock,
    ) -> None:
        """Test transcription functionality.

        Args:
            whisper_config: Test configuration fixture
            mock_whisper: Mock Whisper model
        """
        manager = WhisperManager(whisper_config)
        manager.load_model()

        # Create test audio data
        audio_data = np.random.rand(16000 * 5).astype(np.float32)
        segments = manager.transcribe(audio_data, 16000)

        pytest.assume(len(segments) == 2, "Should produce 2 segments")
        pytest.assume(
            segments[0].text.strip() == "Test transcript one.",
            "First segment text mismatch",
        )
        pytest.assume(
            segments[1].text.strip() == "Test transcript two.",
            "Second segment text mismatch",
        )
        pytest.assume(segments[0].start == 0.0, "First segment start time mismatch")
        pytest.assume(segments[0].end == 2.0, "First segment end time mismatch")
        pytest.assume(segments[1].start == 2.5, "Second segment start time mismatch")
        pytest.assume(segments[1].end == 4.0, "Second segment end time mismatch")

    def test_error_handling(
        self: "TestWhisperManager",
        whisper_config: WhisperConfig,
    ) -> None:
        """Test error handling during transcription.

        Args:
            whisper_config: Test configuration fixture
        """
        manager = WhisperManager(whisper_config)

        # Test transcription without loading model
        with pytest.raises(RuntimeError, match="Model not loaded"):
            audio_data = np.random.rand(16000)
            manager.transcribe(audio_data, 16000)

        # Test with empty audio data
        manager.load_model()
        with pytest.raises(ValueError, match="Empty audio data"):
            manager.transcribe(np.array([]), 16000)

    def test_stream_transcription(
        self: "TestWhisperManager",
        whisper_config: WhisperConfig,
        mock_whisper: Mock,
    ) -> None:
        """Test streaming transcription functionality.

        Args:
            whisper_config: Test configuration fixture
            mock_whisper: Mock Whisper model
        """
        from ..audio_input.recorder import AudioSegment

        manager = WhisperManager(whisper_config)
        manager.load_model()

        # Create mock audio segments
        segments = [
            AudioSegment(
                data=np.random.rand(16000).astype(np.float32),
                sample_rate=16000,
                channels=1,
                timestamp=i,
            )
            for i in range(3)
        ]

        results = manager.transcribe_stream(segments)

        pytest.assume(len(results) == 2, "Should produce 2 segments")
        pytest.assume(
            all(isinstance(seg, TranscriptionSegment) for seg in results),
            "All results should be TranscriptionSegments",
        )
        pytest.assume(
            results[0].text == "Test transcript one.",
            "First segment text mismatch",
        )
        pytest.assume(
            results[1].text == "Test transcript two.",
            "Second segment text mismatch",
        )

    def test_memory_validation(
        self: "TestWhisperManager",
        whisper_config: WhisperConfig,
    ) -> None:
        """Test memory requirement validation.

        Args:
            whisper_config: Test configuration fixture
        """
        whisper_config.model_size = "large"
        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value.total = 8 * 1e9  # 8GB total memory

            # Capture warnings
            with pytest.warns(UserWarning, match="Available memory") as record:
                WhisperManager(whisper_config)

            pytest.assume(len(record) == 1, "Should emit one warning")
