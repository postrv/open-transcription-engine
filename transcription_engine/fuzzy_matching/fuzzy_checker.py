# File: transcription_engine/fuzzy_matching/fuzzy_checker.py
"""Fuzzy/Phoneme Matching module for detecting approximate matches of sensitive terms.

Uses rapidfuzz for efficient fuzzy string matching and phonetics for name matching.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import jellyfish  # For phonetic matching
from rapidfuzz import fuzz, process

from ..utils.config import RedactionConfig, config_manager

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for the FuzzyChecker class
T = TypeVar("T", bound="FuzzyChecker")
# Type variable for dataclasses
D = TypeVar("D", bound="FuzzyMatch")


@dataclass
class FuzzyMatch:
    """Container for detected fuzzy matches."""

    original_text: str
    matched_term: str
    matched_phrase: str
    confidence: float
    start_pos: int
    end_pos: int
    match_type: str  # 'fuzzy', 'phonetic', or 'partial'

    def to_dict(self: D) -> dict[str, Any]:
        """Convert the FuzzyMatch to a dictionary for JSON serialization.

        Returns:
            Dictionary representation of the FuzzyMatch
        """
        return asdict(self)


class FuzzyChecker:
    """Handles fuzzy matching of sensitive terms with configurable thresholds."""

    def __init__(self: T, config: RedactionConfig | None = None) -> None:
        """Initialize the fuzzy checker with configuration."""
        self.config = config or config_manager.load_config().redaction
        self.sensitive_terms = self._load_sensitive_terms()
        # Fix line length by breaking into multiple lines
        self.name_terms = {
            term for term in self.sensitive_terms if len(term.split()) <= 2
        }  # Likely names
        self.fuzzy_threshold = self.config.fuzzy_threshold
        self.min_length = self.config.min_phrase_length

    def _load_sensitive_terms(self: T) -> set[str]:
        """Load sensitive terms from configured file.

        Returns:
            Set of sensitive terms loaded from file.
        """
        try:
            path = Path(self.config.sensitive_phrases_file)
            if not path.exists():
                logger.warning("Sensitive phrases file not found: %s", path)
                return set()

            with open(path) as f:
                terms = {line.strip() for line in f if line.strip()}
            logger.info("Loaded %d sensitive terms", len(terms))
            return terms
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Error loading sensitive terms: %s", e)
            return set()

    def _get_phonetic_matches(
        self: T,
        text: str,
        phrase: str,
    ) -> list[FuzzyMatch]:
        """Find phonetic matches of names and similar sounding terms.

        Args:
            text: Original text containing the phrase
            phrase: Specific phrase to check for matches

        Returns:
            List of phonetic matches found
        """
        matches = []
        phrase_words = phrase.split()

        if len(phrase_words) <= 2:  # Only match potential names
            for term in self.name_terms:
                term_words = term.split()
                if len(phrase_words) == len(term_words):
                    # Compare each word using Soundex
                    if all(
                        jellyfish.soundex(pw) == jellyfish.soundex(tw)
                        for pw, tw in zip(phrase_words, term_words, strict=False)
                    ):
                        matches.append(
                            FuzzyMatch(
                                original_text=text,
                                matched_term=term,
                                matched_phrase=phrase,
                                confidence=0.85,
                                start_pos=text.find(phrase),
                                end_pos=text.find(phrase) + len(phrase),
                                match_type="phonetic",
                            ),
                        )
        return matches

    def _process_segment(
        self: T,
        text: str,
        words: list[str],
        start_idx: int,
        window_size: int,
    ) -> list[FuzzyMatch]:
        """Process a single segment of text for potential matches.

        Args:
            text: Full text being analyzed
            words: List of words in the text
            start_idx: Starting index for the current window
            window_size: Size of the current phrase window

        Returns:
            List of matches found in this segment
        """
        matches: list[FuzzyMatch] = []
        end_idx = min(start_idx + window_size, len(words))
        phrase = " ".join(words[start_idx:end_idx])

        # Skip very short phrases
        if len(phrase) < self.min_length:
            return matches

        # First try phonetic matching for names
        phonetic_matches = self._get_phonetic_matches(text, phrase)
        if phonetic_matches:
            matches.extend(phonetic_matches)
            return matches

        # Then try fuzzy matching
        fuzzy_matches = process.extract(
            phrase,
            self.sensitive_terms,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=self.fuzzy_threshold * 80,
            limit=3,
        )

        for term, score, _ in fuzzy_matches:
            matches.append(
                FuzzyMatch(
                    original_text=text,
                    matched_term=term,
                    matched_phrase=phrase,
                    confidence=score / 100.0,
                    start_pos=text.find(phrase),
                    end_pos=text.find(phrase) + len(phrase),
                    match_type="fuzzy",
                ),
            )

        return matches

    def find_similar_terms(
        self: T,
        transcript_segments: list[dict[str, str]],
    ) -> list[FuzzyMatch]:
        """Find potential matches for sensitive terms in transcript segments.

        Args:
            transcript_segments: List of transcript segments to check

        Returns:
            List of FuzzyMatch objects for review
        """
        if not transcript_segments:
            return []

        all_matches = []
        for segment in transcript_segments:
            text = segment.get("text", "")
            if not text:
                continue

            # Check each word/phrase in the text
            words = text.split()
            for start_idx in range(len(words)):
                for window_size in range(2, min(6, len(words) - start_idx + 1)):
                    matches = self._process_segment(text, words, start_idx, window_size)
                    all_matches.extend(matches)

        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for match in all_matches:
            key = (match.matched_term, match.start_pos, match.end_pos)
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)

        logger.info("Found %d potential sensitive term matches", len(unique_matches))
        return unique_matches

    def save_matches(self: T, matches: list[FuzzyMatch], output_path: Path) -> None:
        """Save matches to a JSON file for review.

        Args:
            matches: List of FuzzyMatch objects to save
            output_path: Path where to save the JSON file

        Raises:
            OSError: If there are filesystem-related errors
            json.JSONDecodeError: If there are JSON serialization errors
        """
        try:
            matches_dict = [match.to_dict() for match in matches]

            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to string first for atomic write
            json_str = json.dumps(matches_dict, indent=2)

            # Write content
            with open(output_path, "w") as f:
                f.write(json_str)
                f.flush()  # Ensure content is written

            logger.info("Saved %d matches to %s", len(matches), output_path)

        except OSError as e:
            logger.error("Error saving matches: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error saving matches: %s", e)
            raise
