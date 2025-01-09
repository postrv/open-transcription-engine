# File: transcription_engine/tests/test_recorder.py
"""
Unit tests for the audio recording and loading functionality.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ..audio_input.recorder import (
    AudioDevice,
    AudioRecorder,
    AudioLoader,
    AudioSegment
)
from ..utils.config import AudioConfig


@pytest.fixture
def audio_config():
    """Fixture providing test audio configuration."""
    return AudioConfig(
        sample_rate=16000,
        channels=1,
        chunk_size=1024,
        format='wav',
        device_index=None
    )


@pytest.fixture
def mock_pyaudio():
    """Fixture providing a mocked PyAudio instance."""
    with patch('pyaudio.PyAudio') as mock:
        mock.return_value.get_device_count.return_value = 2
        mock.return_value.get_device_info_by_index.side_effect = [
            {'maxInputChannels': 2, 'name': 'Test Device 1', 'index': 0, 'defaultSampleRate': 44100},
            {'maxInputChannels': 1, 'name': 'Test Device 2', 'index': 1, 'defaultSampleRate': 48000}
        ]
        yield mock


class TestAudioDevice:
    """Test suite for AudioDevice class."""

    def test_list_devices(self, mock_pyaudio):
        """Test listing available audio devices."""
        device = AudioDevice()
        devices = device.list_devices()

        assert len(devices) == 2
        assert devices[0]['name'] == 'Test Device 1'
        assert devices[0]['channels'] == 2
        assert devices[1]['name'] == 'Test Device 2'
        assert devices[1]['channels'] == 1

    def test_get_default_device(self, mock_pyaudio):
        """Test getting default audio device."""
        mock_pyaudio.return_value.get_default_input_device_info.return_value = {
            'index': 0,
            'name': 'Default Device',
            'maxInputChannels': 2,
            'defaultSampleRate': 44100
        }

        device = AudioDevice()
        default_device = device.get_default_device()

        assert default_device['name'] == 'Default Device'
        assert default_device['channels'] == 2
        assert default_device['sample_rate'] == 44100


class TestAudioRecorder:
    """Test suite for AudioRecorder class."""

    @pytest.fixture
    def mock_stream(self):
        """Fixture providing a mocked audio stream."""
        with patch('pyaudio.PyAudio') as mock_pa:
            mock_stream = MagicMock()
            mock_pa.return_value.open.return_value = mock_stream
            yield mock_stream

    def test_start_stop_recording(self, audio_config, mock_stream):
        """Test recording start and stop functionality."""
        # Create fake audio data
        fake_audio = np.random.rand(audio_config.chunk_size).astype(np.float32)
        mock_stream.read.return_value = fake_audio.tobytes()

        recorder = AudioRecorder(audio_config)
        recorder.start_recording()

        # Let it record a few chunks
        import time
        time.sleep(0.1)

        audio_data = recorder.stop_recording()

        assert isinstance(audio_data, np.ndarray)
        assert len(audio_data) > 0
        assert not recorder.recording

    def test_recording_error_handling(self, audio_config, mock_stream):
        """Test error handling during recording."""
        mock_stream.read.side_effect = Exception("Test error")

        recorder = AudioRecorder(audio_config)
        recorder.start_recording()

        import time
        time.sleep(0.1)

        audio_data = recorder.stop_recording()
        assert audio_data is None


class TestAudioLoader:
    """Test suite for AudioLoader class."""

    def test_load_wav_file(self):
        """Test loading a WAV file."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Create a simple test WAV file
            sample_rate = 16000
            duration = 1  # seconds
            t = np.linspace(0, duration, sample_rate * duration)
            audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

            AudioLoader.save_wav(
                tmp_file.name,
                audio_data,
                sample_rate,
                channels=1
            )

            # Load and verify
            loaded_data, loaded_sr = AudioLoader.load_file(tmp_file.name)
            assert loaded_sr == sample_rate
            assert len(loaded_data) == len(audio_data)
            np.testing.assert_array_almost_equal(loaded_data, audio_data, decimal=4)

        # Clean up
        Path(tmp_file.name).unlink()

    def test_unsupported_format(self):
        """Test handling of unsupported file formats."""
        with pytest.raises(ValueError):
            AudioLoader.load_file('test.xyz')

    def test_nonexistent_file(self):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            AudioLoader.load_file('nonexistent.wav')