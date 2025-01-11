# File: transcription_engine/redaction/redactor.py
"""
Automated and manual redaction module for sensitive information in transcripts.

Supports both pattern-based redaction and user-defined redaction zones.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..fuzzy_matching.fuzzy_checker import FuzzyChecker, FuzzyMatch
from ..utils.config import config_manager
from ..whisper_engine.transcriber import TranscriptionSegment

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RedactionZone:
    """Represents a section of text to be redacted."""

    start_time: float
    end_time: float
    reason: str
    redaction_type: str  # 'auto' or 'manual'
    confidence: float = 1.0

    def to_dict(self: "RedactionZone") -> dict[str, Any]:
        """Convert the RedactionZone to a dictionary for JSON serialization.

        Returns:
            Dictionary representation of the RedactionZone
        """
        return asdict(self)


class TranscriptRedactor:
    """Handles automatic and manual redaction of sensitive information."""

    def __init__(self: "TranscriptRedactor") -> None:
        """Initialize the redactor with configuration."""
        self.config = config_manager.load_config().redaction
        self.fuzzy_checker = FuzzyChecker(self.config)
        self.redaction_char = self.config.redaction_char

    def _prepare_segment_for_json(
        self: "TranscriptRedactor",
        segment: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare a segment for JSON serialization.

        Args:
            segment: Segment dictionary to prepare

        Returns:
            JSON-serializable segment dictionary
        """
        # Create a copy to avoid modifying the original
        processed_segment = segment.copy()

        # Convert RedactionZone objects to dictionaries
        if "redaction_zones" in processed_segment:
            processed_segment["redaction_zones"] = [
                zone.to_dict() if isinstance(zone, RedactionZone) else zone
                for zone in processed_segment["redaction_zones"]
            ]

        return processed_segment

    def auto_redact(
        self: "TranscriptRedactor",
        segments: list[TranscriptionSegment],
    ) -> tuple[list[dict[str, Any]], list[FuzzyMatch]]:
        """Automatically redact sensitive information from transcript segments."""
        # Convert segments to format expected by fuzzy checker
        segment_dicts = [{"text": seg.text} for seg in segments]

        # Find potential matches using fuzzy matching
        potential_matches = self.fuzzy_checker.find_similar_terms(segment_dicts)

        # Create redacted version of segments
        redacted_segments = []
        for segment in segments:
            redacted_text = segment.text
            redaction_zones = []

            # Apply redactions from matches
            matches_for_segment = [
                m for m in potential_matches if m.original_text == segment.text
            ]

            for match in matches_for_segment:
                if match.confidence >= self.config.fuzzy_threshold:
                    # Create redaction mask
                    redaction = self.redaction_char * (match.end_pos - match.start_pos)
                    redacted_text = (
                        redacted_text[: match.start_pos]
                        + redaction
                        + redacted_text[match.end_pos :]
                    )

                    # Record redaction zone
                    zone_length = match.end_pos - match.start_pos
                    time_per_char = (segment.end - segment.start) / len(segment.text)
                    zone_start = segment.start + (match.start_pos * time_per_char)
                    zone_end = zone_start + (zone_length * time_per_char)

                    redaction_zones.append(
                        RedactionZone(
                            start_time=zone_start,
                            end_time=zone_end,
                            reason=f"Matched sensitive term: {match.matched_term}",
                            redaction_type="auto",
                            confidence=match.confidence,
                        ),
                    )

            redacted_segments.append(
                {
                    "text": redacted_text,
                    "start": segment.start,
                    "end": segment.end,
                    "speaker_id": segment.speaker_id,
                    "confidence": segment.confidence,
                    "redaction_zones": redaction_zones,
                },
            )

        logger.info("Applied %d automatic redactions", len(potential_matches))
        return redacted_segments, potential_matches

    def manual_redact(
        self: "TranscriptRedactor",
        segments: list[dict[str, Any]],
        user_zones: list[RedactionZone],
    ) -> list[dict[str, Any]]:
        """Apply manual redactions based on user-defined zones."""
        redacted_segments = []

        for segment in segments:
            redacted_text = segment.get("text", "")
            segment_zones = segment.get("redaction_zones", [])

            # Find zones that overlap with this segment
            relevant_zones = [
                zone
                for zone in user_zones
                if (
                    zone.start_time < segment["end"]
                    and zone.end_time > segment["start"]
                )
            ]

            for zone in relevant_zones:
                # Calculate character positions based on time
                time_per_char = (segment["end"] - segment["start"]) / len(redacted_text)
                start_pos = max(
                    0,
                    int((zone.start_time - segment["start"]) / time_per_char),
                )
                end_pos = min(
                    len(redacted_text),
                    int((zone.end_time - segment["start"]) / time_per_char),
                )

                # Apply redaction
                redaction = self.redaction_char * (end_pos - start_pos)
                redacted_text = (
                    redacted_text[:start_pos] + redaction + redacted_text[end_pos:]
                )

                # Add to redaction zones
                segment_zones.append(zone)

            redacted_segments.append(
                {
                    **segment,
                    "text": redacted_text,
                    "redaction_zones": segment_zones,
                },
            )

        logger.info("Applied %d manual redactions", len(user_zones))
        return redacted_segments

    def save_redactions(
        self: "TranscriptRedactor",
        segments: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Save redacted transcript with metadata to file.

        Args:
            segments: List of transcript segments with redactions
            output_path: Path where to save the JSON file

        Raises:
            ValueError: If no segments provided
            OSError: If there are filesystem-related errors
            json.JSONDecodeError: If there are JSON serialization errors
        """
        if not segments:
            msg = "No segments provided to save"
            raise ValueError(msg)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Process segments for JSON serialization
            processed_segments = [
                self._prepare_segment_for_json(segment) for segment in segments
            ]

            output_data = {
                "segments": processed_segments,
                "redaction_stats": {
                    "total_segments": len(segments),
                    "redacted_segments": len(
                        [s for s in segments if s.get("redaction_zones")],
                    ),
                },
            }

            # Write atomically by first converting to string
            json_str = json.dumps(output_data, indent=2)
            with open(output_path, "w") as f:
                f.write(json_str)
                f.flush()  # Ensure content is written

            logger.info("Saved redacted transcript to %s", output_path)

        except OSError as e:
            logger.error("Error saving redacted transcript: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error saving transcript: %s", e)
            raise
