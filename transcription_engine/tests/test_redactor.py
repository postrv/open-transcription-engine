# File: transcription_engine/tests/test_redactor.py
"""Tests for the redaction module."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pytest import FixtureRequest

from ..redaction.redactor import RedactionZone, TranscriptRedactor
from ..utils.config import RedactionConfig
from ..whisper_engine.transcriber import TranscriptionSegment


@pytest.fixture  # type: ignore[misc]
def mock_config(request: FixtureRequest) -> RedactionConfig:
    """Provide test redaction configuration."""
    return RedactionConfig(
        sensitive_phrases_file=Path("../data/test_phrases.txt"),
        redaction_char="*",
        min_phrase_length=2,
        fuzzy_threshold=0.85,
    )


@pytest.fixture  # type: ignore[misc]
def sample_transcript() -> list[TranscriptionSegment]:
    """Provide sample transcript segments."""
    return [
        TranscriptionSegment(
            text="John Smith spoke about the project.",
            start=0.0,
            end=2.0,
            confidence=0.95,
        ),
        TranscriptionSegment(
            text="His phone number is +44 7700 900123.",
            start=2.5,
            end=4.5,
            confidence=0.92,
        ),
    ]


class TestTranscriptRedactor:
    """Test suite for TranscriptRedactor class."""

    def test_auto_redaction(
        self: "TestTranscriptRedactor",
        mock_config: RedactionConfig,
        sample_transcript: list[TranscriptionSegment],
    ) -> None:
        """Test automatic redaction of sensitive information."""
        redactor = TranscriptRedactor()

        # Mock the fuzzy checker to return some matches
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            redacted_segments, matches = redactor.auto_redact(sample_transcript)

            pytest.assume(len(redacted_segments) == 2, "Should maintain segment count")
            pytest.assume(
                "***" in redacted_segments[0]["text"],
                "Should contain redaction markers",
            )
            pytest.assume(len(matches) > 0, "Should find sensitive terms")

    def test_manual_redaction(
        self: "TestTranscriptRedactor",
        mock_config: RedactionConfig,
        sample_transcript: list[TranscriptionSegment],
    ) -> None:
        """Test manual redaction zones."""
        redactor = TranscriptRedactor()

        user_zones = [
            RedactionZone(
                start_time=1.0,
                end_time=2.0,
                reason="Test redaction",
                redaction_type="manual",
            )
        ]

        # Convert transcript segments to dict format
        segments = [
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "confidence": seg.confidence,
            }
            for seg in sample_transcript
        ]

        redacted = redactor.manual_redact(segments, user_zones)

        pytest.assume(len(redacted) == len(segments), "Should maintain segment count")
        pytest.assume(
            any("*" in seg["text"] for seg in redacted),
            "Should contain redaction markers",
        )
        pytest.assume(
            any(len(seg.get("redaction_zones", [])) > 0 for seg in redacted),
            "Should add redaction zone metadata",
        )

    def test_save_redactions(
        self: "TestTranscriptRedactor", mock_config: RedactionConfig, tmp_path: Path
    ) -> None:
        """Test saving redacted transcript."""
        redactor = TranscriptRedactor()

        segments = [
            {
                "text": "Redacted *** text",
                "start": 0.0,
                "end": 2.0,
                "redaction_zones": [
                    {
                        "start_time": 0.5,
                        "end_time": 1.5,
                        "reason": "Test",
                        "redaction_type": "auto",
                        "confidence": 0.9,
                    }
                ],
            }
        ]

        output_path = tmp_path / "redacted.json"
        redactor.save_redactions(segments, output_path)

        pytest.assume(output_path.exists(), "Should create output file")

        with open(output_path) as f:
            saved_data = json.load(f)
            pytest.assume("segments" in saved_data, "Should contain segments key")
            pytest.assume("redaction_stats" in saved_data, "Should contain stats")

    def test_error_handling(
        self: "TestTranscriptRedactor", mock_config: RedactionConfig
    ) -> None:
        """Test error handling for invalid inputs."""
        redactor = TranscriptRedactor()

        # Test with empty segments
        with pytest.raises(ValueError):
            redactor.save_redactions([], Path("test.json"))

        # Test with invalid redaction zones
        segments = [{"text": "test", "start": 0, "end": 1}]
        zones = [
            RedactionZone(
                start_time=2.0,  # Outside segment bounds
                end_time=3.0,
                reason="Invalid",
                redaction_type="manual",
            )
        ]

        redacted = redactor.manual_redact(segments, zones)
        pytest.assume(
            all("*" not in seg["text"] for seg in redacted),
            "Should not apply out-of-bounds redactions",
        )
