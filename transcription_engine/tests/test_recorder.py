# File: transcription_engine/tests/test_recorder.py
"""Tests for the audio recording and loading functionality."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import FixtureRequest

from ..audio_input.recorder import AudioDevice, AudioLoader, AudioRecorder
from ..utils.config import AudioConfig


@pytest.fixture(scope="function", autouse=False)  # type: ignore[misc]
def audio_config(request: FixtureRequest) -> AudioConfig:
    """Provide test audio configuration."""
    return AudioConfig(
        sample_rate=16000,
        channels=1,
        chunk_size=1024,
        format="wav",
        device_index=None,
    )


@pytest.fixture(scope="function", autouse=False)  # type: ignore[misc]
def mock_pyaudio() -> Generator[MagicMock, None, None]:
    """Provide a mocked PyAudio instance.

    Yields:
        MagicMock: Mock object simulating PyAudio
    """
    with patch("pyaudio.PyAudio") as mock:
        mock.return_value.get_device_count.return_value = 2
        mock.return_value.get_device_info_by_index.side_effect = [
            {
                "maxInputChannels": 2,
                "name": "Test Device 1",
                "index": 0,
                "defaultSampleRate": 44100,
            },
            {
                "maxInputChannels": 1,
                "name": "Test Device 2",
                "index": 1,
                "defaultSampleRate": 48000,
            },
        ]
        yield mock


class TestAudioDevice:
    """Test suite for AudioDevice class."""

    def test_list_devices(self: "TestAudioDevice", mock_pyaudio: MagicMock) -> None:
        """Test listing available audio devices.

        Args:
            mock_pyaudio: Mock PyAudio instance
        """
        device = AudioDevice()
        devices = device.list_devices()

        pytest.assume(len(devices) == 2, "Should find two devices")
        pytest.assume(
            devices[0]["name"] == "Test Device 1",
            "First device name mismatch",
        )
        pytest.assume(
            devices[0]["channels"] == 2,
            "First device channel count mismatch",
        )
        pytest.assume(
            devices[1]["name"] == "Test Device 2",
            "Second device name mismatch",
        )
        pytest.assume(
            devices[1]["channels"] == 1,
            "Second device channel count mismatch",
        )

    def test_get_default_device(
        self: "TestAudioDevice",
        mock_pyaudio: MagicMock,
    ) -> None:
        """Test getting default audio device.

        Args:
            mock_pyaudio: Mock PyAudio instance
        """
        mock_pyaudio.return_value.get_default_input_device_info.return_value = {
            "index": 0,
            "name": "Default Device",
            "maxInputChannels": 2,
            "defaultSampleRate": 44100,
        }

        device = AudioDevice()
        default_device = device.get_default_device()

        pytest.assume(
            default_device is not None and default_device["name"] == "Default Device",
            "Default device name mismatch",
        )
        pytest.assume(
            default_device is not None and default_device["channels"] == 2,
            "Default device channel count mismatch",
        )
        pytest.assume(
            default_device is not None and default_device["sample_rate"] == 44100,
            "Default device sample rate mismatch",
        )


class TestAudioRecorder:
    """Test suite for AudioRecorder class."""

    @pytest.fixture  # type: ignore[misc]
    def mock_stream(self: "TestAudioRecorder") -> Generator[MagicMock, None, None]:
        """Provide a mocked audio stream.

        Yields:
            MagicMock: Mock object simulating audio stream
        """
        with patch("pyaudio.PyAudio") as mock_pa:
            mock_stream = MagicMock()
            mock_pa.return_value.open.return_value = mock_stream
            yield mock_stream

    def test_start_stop_recording(
        self: "TestAudioRecorder",
        audio_config: AudioConfig,
        mock_stream: MagicMock,
    ) -> None:
        """Test recording start and stop functionality.

        Args:
            audio_config: Audio configuration fixture
            mock_stream: Mock audio stream
        """
        # Create fake audio data
        fake_audio = np.random.rand(audio_config.chunk_size).astype(np.float32)
        mock_stream.read.return_value = fake_audio.tobytes()

        recorder = AudioRecorder(audio_config)
        recorder.start_recording()

        # Let it record a few chunks
        import time

        time.sleep(0.1)

        audio_data = recorder.stop_recording()

        pytest.assume(isinstance(audio_data, np.ndarray), "Should return numpy array")
        if audio_data is not None:  # Type guard for mypy
            pytest.assume(len(audio_data) > 0, "Should have recorded data")
        pytest.assume(not recorder.recording, "Should not be recording after stop")

    def test_recording_error_handling(
        self: "TestAudioRecorder",
        audio_config: AudioConfig,
        mock_stream: MagicMock,
    ) -> None:
        """Test error handling during recording.

        Args:
            audio_config: Audio configuration fixture
            mock_stream: Mock audio stream
        """
        mock_stream.read.side_effect = Exception("Test error")

        recorder = AudioRecorder(audio_config)
        recorder.start_recording()

        import time

        time.sleep(0.1)

        audio_data = recorder.stop_recording()
        pytest.assume(audio_data is None, "Should return None on error")


class TestAudioLoader:
    """Test suite for AudioLoader class."""

    def test_load_wav_file(self: "TestAudioLoader") -> None:
        """Test loading a WAV file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Create a simple test WAV file
            sample_rate = 16000
            duration = 1  # seconds
            t = np.linspace(0, duration, sample_rate * duration)
            audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

            AudioLoader.save_wav(tmp_file.name, audio_data, sample_rate, channels=1)

            # Load and verify
            loaded_data, loaded_sr = AudioLoader.load_file(tmp_file.name)
            pytest.assume(loaded_sr == sample_rate, "Sample rate mismatch")
            pytest.assume(
                len(loaded_data) == len(audio_data),
                "Audio length mismatch",
            )
            np.testing.assert_array_almost_equal(loaded_data, audio_data, decimal=4)

        # Clean up
        Path(tmp_file.name).unlink()

    def test_unsupported_format(self: "TestAudioLoader") -> None:
        """Test handling of unsupported file formats."""
        with pytest.raises(ValueError, match="Unsupported audio format"):
            AudioLoader.load_file("test.xyz")

    def test_nonexistent_file(self: "TestAudioLoader") -> None:
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            AudioLoader.load_file("nonexistent.wav")


if __name__ == "__main__":
    pytest.main([__file__])
