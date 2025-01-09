# File: transcription_engine/tests/test_diarizer.py
"""
Unit tests for the speaker diarization module.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from pathlib import Path

from ..speaker_id.diarizer import (
    DiarizationManager,
    SpeakerSegment,
    PYANNOTE_AVAILABLE
)
from ..whisper_engine.transcriber import TranscriptionSegment


@pytest.fixture
def sample_audio():
    """Fixture providing sample audio data."""
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

    @pytest.fixture
    def mock_pyannote(self):
        """Fixture providing a mocked PyAnnote pipeline."""
        with patch('pyannote.audio.Pipeline') as mock:
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

    def test_pyannote_processing(self, mock_pyannote, sample_audio):
        """Test processing using PyAnnote."""
        audio_data, sample_rate = sample_audio
        mono_audio = audio_data.mean(axis=1)

        manager = DiarizationManager(use_pyannote=True, auth_token="dummy_token")
        segments = manager.process_singlechannel(mono_audio, sample_rate)

        assert len(segments) == 2
        assert segments[0].speaker_id == "SPEAKER_1"
        assert segments[1].speaker_id == "SPEAKER_2"
        assert segments[0].start == 0.0
        assert segments[0].end == 2.0


class TestBasicDiarization:
    """Test suite for basic diarization functionality."""

    def test_device_selection(self):
        """Test computation device selection logic."""
        # Test CPU fallback
        with patch('torch.cuda.is_available', return_value=False), \
                patch('torch.backends.mps.is_available', return_value=False):
            manager = DiarizationManager(use_pyannote=False)
            assert manager.device.type == 'cpu'

        # Test MPS selection
        with patch('torch.cuda.is_available', return_value=False), \
                patch('torch.backends.mps.is_available', return_value=True):
            manager = DiarizationManager(use_pyannote=False)
            assert manager.device.type == 'mps'

        # Test CUDA selection
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.backends.mps.is_available', return_value=False):
            manager = DiarizationManager(use_pyannote=False)
            assert manager.device.type == 'cuda'

    def test_multichannel_processing(self, sample_audio):
        """Test processing of multi-channel audio."""
        audio_data, sample_rate = sample_audio
        manager = DiarizationManager(use_pyannote=False)

        segments = manager.process_multichannel(audio_data, sample_rate)

        assert len(segments) > 0
        assert all(isinstance(seg, SpeakerSegment) for seg in segments)
        assert all(seg.channel in [0, 1] for seg in segments)
        assert all(seg.speaker_id in ['speaker_1', 'speaker_2'] for seg in segments)

    def test_basic_singlechannel(self, sample_audio):
        """Test basic single-channel processing without PyAnnote."""
        audio_data, sample_rate = sample_audio
        mono_audio = audio_data.mean(axis=1)

        manager = DiarizationManager(use_pyannote=False)
        segments = manager.process_singlechannel(mono_audio, sample_rate)

        assert len(segments) > 0
        assert all(isinstance(seg, SpeakerSegment) for seg in segments)
        assert all(seg.speaker_id == "speaker_unknown" for seg in segments)

    def test_speaker_assignment(self):
        """Test assignment of speaker IDs to transcription segments."""
        manager = DiarizationManager(use_pyannote=False)

        # Create sample transcription segments
        transcription_segments = [
            TranscriptionSegment(
                text="Hello there",
                start=0.0,
                end=2.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="General Kenobi",
                start=2.5,
                end=4.0,
                confidence=0.85
            )
        ]

        # Create sample diarization segments
        diarization_segments = [
            SpeakerSegment(
                start=0.0,
                end=2.2,
                speaker_id="SPEAKER_1",
                score=0.95
            ),
            SpeakerSegment(
                start=2.3,
                end=4.1,
                speaker_id="SPEAKER_2",
                score=0.92
            )
        ]

        # Test speaker assignment
        updated_segments = manager.assign_speaker_ids(
            transcription_segments,
            diarization_segments
        )

        assert len(updated_segments) == 2
        assert updated_segments[0].speaker_id == "SPEAKER_1"
        assert updated_segments[1].speaker_id == "SPEAKER_2"

    def test_error_handling(self):
        """Test error handling scenarios."""
        # Test invalid audio input for multichannel
        manager = DiarizationManager(use_pyannote=False)
        with pytest.raises(ValueError, match="Expected multi-channel audio"):
            manager.process_multichannel(np.array([1, 2, 3]), 16000)


if __name__ == '__main__':
    pytest.main([__file__])