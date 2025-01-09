# File: transcription_engine/tests/test_transcriber.py
"""
Unit tests for the Whisper integration module.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from ..whisper_engine.transcriber import (
    WhisperManager,
    TranscriptionSegment,
)
from ..utils.config import WhisperConfig


@pytest.fixture
def whisper_config():
    """Fixture providing test Whisper configuration."""
    return WhisperConfig(
        model_size='tiny',
        device='cpu',
        language='en',
        batch_size=16,
        compute_type='float32'
    )


@pytest.fixture
def mock_whisper():
    """Fixture providing a mocked Whisper model."""
    with patch('whisper.load_model') as mock:
        # Create a mock model with transcribe method
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'segments': [
                {
                    'text': 'Test transcript one.',
                    'start': 0.0,
                    'end': 2.0,
                    'confidence': 0.95
                },
                {
                    'text': 'Test transcript two.',
                    'start': 2.1,
                    'end': 4.0,
                    'confidence': 0.92
                }
            ]
        }
        mock.return_value = mock_model
        yield mock


class TestWhisperManager:
    """Test suite for WhisperManager class."""

    def test_device_selection(self, whisper_config):
        """Test computation device selection logic."""
        # Test CPU fallback
        with patch('torch.cuda.is_available', return_value=False), \
                patch('torch.backends.mps.is_available', return_value=False):
            manager = WhisperManager(whisper_config)
            assert manager.device.type == 'cpu'

        # Test MPS selection
        with patch('torch.cuda.is_available', return_value=False), \
                patch('torch.backends.mps.is_available', return_value=True):
            whisper_config.device = 'auto'
            manager = WhisperManager(whisper_config)
            assert manager.device.type == 'mps'

        # Test CUDA selection
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.backends.mps.is_available', return_value=False):
            whisper_config.device = 'auto'
            manager = WhisperManager(whisper_config)
            assert manager.device.type == 'cuda'

    def test_model_loading(self, whisper_config, mock_whisper):
        """Test Whisper model loading."""
        manager = WhisperManager(whisper_config)
        assert manager.load_model()
        assert manager.model is not None

        # Test unloading
        manager.unload_model()
        assert manager.model is None

    def test_audio_preparation(self, whisper_config):
        """Test audio preparation for Whisper."""
        manager = WhisperManager(whisper_config)

        # Test stereo to mono conversion
        stereo_audio = np.random.rand(1000, 2)
        mono_audio = manager._prepare_audio(stereo_audio, 16000)
        assert len(mono_audio.shape) == 1

        # Test resampling
        audio_44k = np.random.rand(44100)
        audio_16k = manager._prepare_audio(audio_44k, 44100)
        assert len(audio_16k) == 16000

    def test_audio_chunking(self, whisper_config):
        """Test audio chunking for long recordings."""
        manager = WhisperManager(whisper_config)

        # Create 60 seconds of audio at 16kHz
        audio_data = np.random.rand(16000 * 60)
        chunks = manager._chunk_audio(audio_data, chunk_duration=30)

        assert len(chunks) == 2  # Should split into 2 30-second chunks
        for chunk, timestamp in chunks:
            assert len(chunk) <= 16000 * 30  # Each chunk should be ≤ 30 seconds

    def test_transcription(self, whisper_config, mock_whisper):
        """Test transcription functionality."""
        manager = WhisperManager(whisper_config)
        manager.load_model()

        # Create test audio data
        audio_data = np.random.rand(16000 * 5)  # 5 seconds of audio
        segments = manager.transcribe(audio_data, 16000)

        assert len(segments) == 2
        assert isinstance(segments[0], TranscriptionSegment)
        assert segments[0].text == 'Test transcript one.'
        assert segments[0].start == 0.0
        assert segments[0].end == 2.0
        assert segments[0].confidence == 0.95

    def test_error_handling(self, whisper_config):
        """Test error handling during transcription."""
        manager = WhisperManager(whisper_config)

        # Test transcription without loading model
        with pytest.raises(RuntimeError):
            audio_data = np.random.rand(16000)
            manager.transcribe(audio_data, 16000)

        # Test with invalid audio data
        manager.load_model()
        with pytest.raises(ValueError):
            manager.transcribe(np.array([]), 16000)

        # Test with invalid sample rate
        with pytest.raises(ValueError):
            manager.transcribe(np.random.rand(16000), -1)

    def test_stream_transcription(self, whisper_config, mock_whisper):
        """Test streaming transcription functionality."""
        from ..audio_input.recorder import AudioSegment

        manager = WhisperManager(whisper_config)
        manager.load_model()

        # Create mock audio segments
        segments = [
            AudioSegment(
                data=np.random.rand(16000).astype(np.float32),
                sample_rate=16000,
                channels=1,
                timestamp=i
            )
            for i in range(3)
        ]

        results = manager.transcribe_stream(segments)

        assert len(results) == 2  # Based on our mock_whisper fixture
        assert all(isinstance(seg, TranscriptionSegment) for seg in results)
        assert results[0].text == 'Test transcript one.'
        assert results[1].text == 'Test transcript two.'

    def test_memory_validation(self, whisper_config):
        """Test memory requirement validation."""
        # Test with large model on system with limited memory
        whisper_config.model_size = 'large'
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.total = 8 * 1e9  # 8GB total memory

            with pytest.warns(UserWarning, match="Available memory"):
                WhisperManager(whisper_config)


if __name__ == '__main__':
    pytest.main([__file__])